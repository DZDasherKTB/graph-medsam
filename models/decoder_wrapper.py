import torch
import torch.nn as nn
import torch.nn.functional as F

class MedSAMDecoderWrapper(nn.Module):
    def __init__(self, config, medsam_decoder):
        super().__init__()
        self.decoder = medsam_decoder
        self.feat_dim = config['model']['embedding_dim']
        
        # Grid dimensions: 16*16*8 = 2048 tokens
        self.grid_size = (8, 16, 16) 

    def forward(self, z_final, original_size):
        """
        z_final: (B, 2048, 768)
        original_size: (D, H, W)
        """
        B, N, C = z_final.shape

        # 1. Reshape to 3D Spatial Feature Map
        # (B, 2048, 768) -> (B, 768, 8, 16, 16)
        x = z_final.transpose(1, 2).view(B, C, *self.grid_size)

        # 2. Reconstruct Positional Encodings (Issue 2 Fix)
        # Most MedSAM/SAM decoders have a pe_layer to generate 3D spatial priors
        # We generate the PE for the current grid size (8, 16, 16)
        image_pe = self.decoder.pe_layer(self.grid_size).unsqueeze(0).expand(B, -1, -1, -1, -1)

        # 3. Handle Prompt Embeddings (Issue 3 Fix)
        # We use the decoder's default learnable 'not-a-point' or 'empty' tokens
        # to maintain the expected input format without manual zero-tensors.
        sparse_embeddings, dense_embeddings = self.decoder.prompt_encoder(
            points=None,
            boxes=None,
            masks=None,
        )

        # 4. Decoder Pass
        # low_res_masks shape: (B, num_masks, d, h, w)
        low_res_masks, iou_predictions = self.decoder(
            image_embeddings=x,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        # 5. Output Selection & Upsampling (Issue 4 Fix)
        # Select the first mask (index 0) if multimask_output is False
        # Resulting shape: (B, 1, d, h, w)
        if low_res_masks.shape[1] > 1:
            low_res_masks = low_res_masks[:, 0:1, :, :, :]

        final_logits = F.interpolate(
            low_res_masks,
            size=original_size,
            mode='trilinear',
            align_corners=False
        )

        return final_logits