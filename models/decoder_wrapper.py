import torch
import torch.nn as nn
import torch.nn.functional as F

class MedSAMDecoderWrapper(nn.Module):
    def __init__(self, config, medsam_decoder):
        super().__init__()
        self.decoder = medsam_decoder
        self.feat_dim = config['model']['embedding_dim']
        
        # Determine grid dimensions from token count (2048 = 16 * 16 * 8)
        # These should match the patch-grid used by the MedSAM ViT encoder
        self.grid_d = 8
        self.grid_h = 16
        self.grid_w = 16

    def forward(self, z_final, original_size):
        """
        Args:
            z_final (torch.Tensor): (B, 2048, 768) - Fused Graph-Visual tokens
            original_size (tuple): (D, H, W) - Original input volume dimensions
        Returns:
            torch.Tensor: (B, 1, D, H, W) - Final segmentation logits
        """
        B, N, C = z_final.shape

        # 1. Reshape flat tokens back to a 3D feature volume
        # (B, 2048, 768) -> (B, 768, 8, 16, 16)
        x = z_final.transpose(1, 2).view(B, C, self.grid_d, self.grid_h, self.grid_w)

        # 2. Pass through the MedSAM Mask Decoder
        # Note: MedSAM's decoder usually expects image embeddings and 
        # potentially sparse/dense prompt embeddings (which we keep empty 
        # or use as default learnable queries in the base MedSAM).
        low_res_masks, iou_predictions = self.decoder(
            image_embeddings=x,
            image_pe=None,        # Positional encodings are usually baked or passed here
            sparse_prompt_embeddings=None,
            dense_prompt_embeddings=None,
            multimask_output=False,
        )

        # 3. Upsample to original volume resolution
        # MedSAM outputs at a lower resolution (e.g., 1/4 of input)
        final_logits = F.interpolate(
            low_res_masks,
            size=original_size,
            mode='trilinear',
            align_corners=False
        )

        return final_logits