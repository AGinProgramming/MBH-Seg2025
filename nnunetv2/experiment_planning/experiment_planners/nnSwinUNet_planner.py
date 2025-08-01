import numpy as np
from copy import deepcopy
from typing import Union, List, Tuple

from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet
from dynamic_network_architectures.building_blocks.helper import convert_dim_to_conv_op, get_matching_instancenorm
from torch import nn

from nnunetv2.experiment_planning.experiment_planners.default_experiment_planner import ExperimentPlanner

from nnunetv2.experiment_planning.experiment_planners.network_topology import get_pool_and_conv_props


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
        # 计算输入通道数
        num_input_channels = len(self.dataset_json.get('channel_names', self.dataset_json.get('modality', {})))
        
        # 根据2D/3D确定最大特征数
        max_num_features = self.UNet_max_features_2d if len(spacing) == 2 else self.UNet_max_features_3d
        
        # 选择卷积操作
        conv_op = convert_dim_to_conv_op(len(spacing))
        
        # 计算初始patch_size，参考ResEncUNetPlanner
        # ...
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
        network_num_pool_per_axis, pool_op_kernel_sizes, conv_kernel_sizes, patch_size, \
        shape_must_be_divisible_by = get_pool_and_conv_props(spacing, initial_patch_size,
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
