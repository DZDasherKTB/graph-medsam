import torch
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np
from scipy.ndimage import distance_transform_edt

class MedicalDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        # Issue 1 Fix: Pre-generate coords once
        self.coords = self.generate_grid_coords()

    def load_volume(self, path):
        vol = nib.load(path).get_fdata().astype(np.float32)
        # Issue 3 Fix: Basic Z-score normalization
        vol = (vol - np.mean(vol)) / (np.std(vol) + 1e-6)
        return vol

    def compute_sdf(self, mask):
        mask = mask.astype(bool)
        dist_out = distance_transform_edt(~mask)
        dist_in = distance_transform_edt(mask)
        sdf = dist_out - dist_in
        # Issue 2 Fix: Normalize SDF to prevent loss dominance
        if np.max(np.abs(sdf)) > 0:
            sdf = sdf / np.max(np.abs(sdf))
        return sdf.astype(np.float32)

    def generate_grid_coords(self):
        d, h, w = 8, 16, 16
        z = torch.linspace(0, 1, d)
        y = torch.linspace(0, 1, h)
        x = torch.linspace(0, 1, w)
        grid_z, grid_y, grid_x = torch.meshgrid(z, y, x, indexing='ij')
        return torch.stack([grid_z, grid_y, grid_x], dim=-1).view(-1, 3)

    def __getitem__(self, idx):
        image = self.load_volume(self.image_paths[idx])
        mask = nib.load(self.mask_paths[idx]).get_fdata().astype(np.float32)
        sdf = self.compute_sdf(mask)

        image = torch.from_numpy(image).unsqueeze(0)
        mask = torch.from_numpy(mask).unsqueeze(0)
        sdf = torch.from_numpy(sdf).unsqueeze(0)

        if self.transform:
            image, mask, sdf = self.transform(image, mask, sdf)

        return {
            "image": image,
            "mask": mask,
            "sdf": sdf,
            "coords": self.coords # Reused static tensor
        }