import torch
import random

class RandomFlip3D:
    def __call__(self, image, mask, sdf):
        # Flip along Horizontal (W) and Vertical (H) axes
        if random.random() > 0.5:
            # dims 1, 2, 3 correspond to D, H, W in (C, D, H, W)
            axis = random.choice([2, 3]) 
            image = torch.flip(image, dims=[axis])
            mask = torch.flip(mask, dims=[axis])
            sdf = torch.flip(sdf, dims=[axis])

        return image, mask, sdf

class IntensityScale:
    def __call__(self, image, mask, sdf):
        # Only scale the image, never the mask or SDF
        scale = 0.9 + 0.2 * torch.rand(1)
        image = image * scale
        return image, mask, sdf

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, image, mask, sdf):
        for t in self.transforms:
            image, mask, sdf = t(image, mask, sdf)
        return image, mask, sdf
    
class RandomNoise:
    def __init__(self, std=0.05):
        self.std = std
    def __call__(self, image, mask, sdf):
        if random.random() > 0.8: # 20% chance
            noise = torch.randn_like(image) * self.std
            image = image + noise
        return image, mask, sdf

class IntensityScale:
    def __call__(self, image, mask, sdf):
        scale = 0.9 + 0.2 * torch.rand(1)
        return image * scale, mask, sdf