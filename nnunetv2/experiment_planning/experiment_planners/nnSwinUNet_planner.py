import numpy as np
from copy import deepcopy
from typing import Union, List, Tuple

from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet
from dynamic_network_architectures.building_blocks.helper import convert_dim_to_conv_op, get_matching_instancenorm
from torch import nn

from nnunetv2.experiment_planning.experiment_planners.default_experiment_planner import ExperimentPlanner

from nnunetv2.experiment_planning.experiment_planners.network_topology import get_pool_and_conv_props

from nnunetv2.nnSwinUNet import nnSwinUNet


class SwinUNetPlanner(ExperimentPlanner):
    def __init__(self, dataset_name_or_id, gpu_memory_target_in_gb=8,
                 preprocessor_name='DefaultPreprocessor',
                 plans_name='nnUNetSwinUNetPlans',
                 overwrite_target_spacing=None,
                 suppress_transpose=False):
        super().__init__(dataset_name_or_id, gpu_memory_target_in_gb,
                         preprocessor_name, plans_name,
                         overwrite_target_spacing, suppress_transpose)
        
        # 你自己的SwinUNet类
        self.UNet_class = nnSwinUNet  
        # 需要参考或调试的基准VRAM参考值（建议先粗略估计）
        self.UNet_reference_val_3d = 500000000  # 你可以根据模型计算量估计
        self.UNet_reference_val_2d = 120000000
        # 例如，SwinTransformer里block数等超参数，可按stage设定
        self.UNet_blocks_per_stage_encoder = (2, 2, 6, 2)
        self.UNet_blocks_per_stage_decoder = (1, 1, 1, 1)

    def generate_data_identifier(self, configuration_name: str) -> str:
        if configuration_name in ['2d', '3d_fullres']:
            return 'nnUNetPlans' + '_' + configuration_name
        else:
            return self.plans_identifier + '_' + configuration_name

    def get_plans_for_configuration(self, spacing, median_shape, data_identifier,
                                    approximate_n_voxels_dataset, _cache):
        
        
        # 计算初始patch_size，参考ResEncUNetPlanner
        # ...
        num_classes = len(self.dataset_json['labels'].keys())

        # 计算输入通道数
        num_input_channels = len(self.dataset_json.get('channel_names', self.dataset_json.get('modality', {})))
        
        # 根据2D/3D确定最大特征数
        max_num_features = self.UNet_max_features_2d if len(spacing) == 2 else self.UNet_max_features_3d
        
        # 选择卷积操作
        conv_op = convert_dim_to_conv_op(len(spacing))

        norm_op = get_matching_instancenorm(conv_op)
        deep_supervision = True

        def features_per_stage(num_stages, max_num_features) -> Tuple[int, ...]:
            return tuple([min(max_num_features, self.UNet_base_num_features * 2 ** i) for
                          i in range(num_stages)])

        def _keygen(patch_size, strides):
            return str(patch_size) + '_' + str(strides)

        assert all([i > 0 for i in spacing]), f"Spacing must be > 0! Spacing: {spacing}"

        # 计算网络结构和层数，可能需要自定义函数适配SwinUNet的stage/block数
        # 你可以用get_pool_and_conv_props，但SwinUNet不一定使用普通卷积池化，这里可自定义
        
        tmp = 1 / np.array(spacing)
        
        if len(spacing) == 3:
            initial_patch_size = [round(i) for i in tmp * (256 ** 3 / np.prod(tmp)) ** (1 / 3)]
        elif len(spacing) == 2:
            initial_patch_size = [round(i) for i in tmp * (2048 ** 2 / np.prod(tmp)) ** (1 / 2)]
        else:
            raise RuntimeError()
        
        initial_patch_size = np.minimum(initial_patch_size, median_shape[:len(spacing)])
        network_num_pool_per_axis, pool_op_kernel_sizes, conv_kernel_sizes, patch_size, shape_must_be_divisible_by = get_pool_and_conv_props(spacing, initial_patch_size,
                                                             self.UNet_featuremap_min_edge_length,
                                                             999999)
        num_stages = len(pool_op_kernel_sizes)

        norm = get_matching_instancenorm(conv_op)
        # 构造architecture_kwargs字典，示例：
        architecture_kwargs = {
            'network_class_name': self.UNet_class.__module__ + '.' + self.UNet_class.__name__,
            'arch_kwargs': {
                'n_stages': num_stages,
                'features_per_stage': features_per_stage,
                'embed_dim': 96,             # 例如SwinUNet默认embed_dim
                'num_heads': [3,6,12,24],   # 每stage头数
                'window_size': 7,
                'mlp_ratio': 4.0,
                'conv_op': conv_op.__module__ + '.' + conv_op.__name__,
                'norm_op': get_matching_instancenorm(conv_op).__module__ + '.' + get_matching_instancenorm(conv_op).__name__,
                'dropout_op': None,
                'nonlin': 'torch.nn.GELU',
                'num_classes': len(self.dataset_json['labels'].keys()),
                # 更多SwinUNet自定义参数
            },
            '_kw_requires_import': ('conv_op', 'norm_op', 'dropout_op', 'nonlin'),
        }
        
        # 显存估算逻辑和patch_size动态调整（可以复用ResEncUNetPlanner逻辑）
        # ...
        # now estimate vram consumption
        if _keygen(patch_size, pool_op_kernel_sizes) in _cache.keys():
            estimate = _cache[_keygen(patch_size, pool_op_kernel_sizes)]
        else:
            estimate = self.static_estimate_VRAM_usage(patch_size,
                                                       num_input_channels,
                                                       len(self.dataset_json['labels'].keys()),
                                                       architecture_kwargs['network_class_name'],
                                                       architecture_kwargs['arch_kwargs'],
                                                       architecture_kwargs['_kw_requires_import'],
                                                       )
            _cache[_keygen(patch_size, pool_op_kernel_sizes)] = estimate

        # how large is the reference for us here (batch size etc)?
        # adapt for our vram target
        reference = (self.UNet_reference_val_2d if len(spacing) == 2 else self.UNet_reference_val_3d) * \
                    (self.UNet_vram_target_GB / self.UNet_reference_val_corresp_GB)

        while estimate > reference:
            # print(patch_size)
            # patch size seems to be too large, so we need to reduce it. Reduce the axis that currently violates the
            # aspect ratio the most (that is the largest relative to median shape)
            axis_to_be_reduced = np.argsort([i / j for i, j in zip(patch_size, median_shape[:len(spacing)])])[-1]

            # we cannot simply reduce that axis by shape_must_be_divisible_by[axis_to_be_reduced] because this
            # may cause us to skip some valid sizes, for example shape_must_be_divisible_by is 64 for a shape of 256.
            # If we subtracted that we would end up with 192, skipping 224 which is also a valid patch size
            # (224 / 2**5 = 7; 7 < 2 * self.UNet_featuremap_min_edge_length(4) so it's valid). So we need to first
            # subtract shape_must_be_divisible_by, then recompute it and then subtract the
            # recomputed shape_must_be_divisible_by. Annoying.
            patch_size = list(patch_size)
            tmp = deepcopy(patch_size)
            tmp[axis_to_be_reduced] -= shape_must_be_divisible_by[axis_to_be_reduced]
            _, _, _, _, shape_must_be_divisible_by = \
                get_pool_and_conv_props(spacing, tmp,
                                        self.UNet_featuremap_min_edge_length,
                                        999999)
            patch_size[axis_to_be_reduced] -= shape_must_be_divisible_by[axis_to_be_reduced]

            # now recompute topology
            network_num_pool_per_axis, pool_op_kernel_sizes, conv_kernel_sizes, patch_size, \
            shape_must_be_divisible_by = get_pool_and_conv_props(spacing, patch_size,
                                                                 self.UNet_featuremap_min_edge_length,
                                                                 999999)

            num_stages = len(pool_op_kernel_sizes)
            architecture_kwargs['arch_kwargs'].update({
                'n_stages': num_stages,
                'kernel_sizes': conv_kernel_sizes,
                'strides': pool_op_kernel_sizes,
                'features_per_stage': features_per_stage(num_stages, max_num_features),
                'n_blocks_per_stage': self.UNet_blocks_per_stage_encoder[:num_stages],
                'n_conv_per_stage_decoder': self.UNet_blocks_per_stage_decoder[:num_stages - 1],
            })
            if _keygen(patch_size, pool_op_kernel_sizes) in _cache.keys():
                estimate = _cache[_keygen(patch_size, pool_op_kernel_sizes)]
            else:
                estimate = self.static_estimate_VRAM_usage(
                    patch_size,
                    num_input_channels,
                    len(self.dataset_json['labels'].keys()),
                    architecture_kwargs['network_class_name'],
                    architecture_kwargs['arch_kwargs'],
                    architecture_kwargs['_kw_requires_import'],
                )
                _cache[_keygen(patch_size, pool_op_kernel_sizes)] = estimate

        # alright now let's determine the batch size. This will give self.UNet_min_batch_size if the while loop was
        # executed. If not, additional vram headroom is used to increase batch size
        ref_bs = self.UNet_reference_val_corresp_bs_2d if len(spacing) == 2 else self.UNet_reference_val_corresp_bs_3d
        batch_size = round((reference / estimate) * ref_bs)

        # we need to cap the batch size to cover at most 5% of the entire dataset. Overfitting precaution. We cannot
        # go smaller than self.UNet_min_batch_size though
        bs_corresponding_to_5_percent = round(
            approximate_n_voxels_dataset * self.max_dataset_covered / np.prod(patch_size, dtype=np.float64))
        batch_size = max(min(batch_size, bs_corresponding_to_5_percent), self.UNet_min_batch_size)

        resampling_data, resampling_data_kwargs, resampling_seg, resampling_seg_kwargs = self.determine_resampling()
        resampling_softmax, resampling_softmax_kwargs = self.determine_segmentation_softmax_export_fn()

        normalization_schemes, mask_is_used_for_norm = \
            self.determine_normalization_scheme_and_whether_mask_is_used_for_norm()
        
        # 返回plan字典
        plan = {
            'data_identifier': data_identifier,
            'preprocessor_name': self.preprocessor_name,
            'batch_size': batch_size,
            'patch_size': patch_size,
            'median_image_size_in_voxels': median_shape,
            'spacing': spacing,
            'normalization_schemes': normalization_schemes,
            'use_mask_for_norm': mask_is_used_for_norm,
            'resampling_fn_data': resampling_data.__name__,
            'resampling_fn_seg': resampling_seg.__name__,
            'resampling_fn_data_kwargs': resampling_data_kwargs,
            'resampling_fn_seg_kwargs': resampling_seg_kwargs,
            'resampling_fn_probabilities': resampling_softmax.__name__,
            'resampling_fn_probabilities_kwargs': resampling_softmax_kwargs,
            'architecture': architecture_kwargs
        }
        return plan
