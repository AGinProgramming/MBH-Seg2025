from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.nnSwinUNet import nnSwinEncoder, nnSwinUNetDecoder
import torch.nn as nn

# class SwinUNetTrainer(nnUNetTrainer):
#     def build_network(self):
#         self.network = SwinUNet(
#             in_channels=self.plans['num_input_channels'],
#             out_channels=self.plans['num_classes'],
#             img_size=self.plans['patch_size'],
#         )
#         self.network.to(self.device)

class nnSwinUNetTrainer(nnUNetTrainer):
    def build_network_architecture(self):
        num_input_channels = self.configuration_manager.determine_num_input_channels()
        num_output_channels = self.label_manager.num_segmentation_heads

        encoder = nnSwinEncoder(in_chans=num_input_channels)
        decoder = nnSwinUNetDecoder(encoder_channels=encoder.out_channels, out_channels=num_output_channels,
                              deep_supervision=self.configuration_manager.use_deep_supervision)

        class HybridModel(nn.Module):
            def __init__(self, encoder, decoder):
                super().__init__()
                self.encoder = encoder
                self.decoder = decoder

            def forward(self, x):
                features = self.encoder(x)
                out = self.decoder(features)
                return out

        return HybridModel(encoder, decoder)