import torch
from torch.utils.data import DataLoader
import yaml

from segment_anything import sam_model_registry

# Model
from models.graph_medsam import GraphMedSAM

# Dataset
from datasets.dataset_loader import MedicalDataset
from datasets.transforms import Compose, RandomFlip3D, IntensityScale, RandomNoise

# Trainer
from training.trainer import VesselTrainer

# Losses
from losses.dice_loss import DiceLoss
from losses.boundary_loss import BoundaryLoss
from losses.geometry_loss import GeometryLoss
from losses.topology_loss import TopologyLoss
import glob

def load_vessel12():

    imgs = sorted(glob.glob("data/vessel12/images/*.nii.gz"))
    masks = sorted(glob.glob("data/vessel12/masks/*.nii.gz"))

    print("Loaded scans:", len(imgs))

    return imgs, masks

def load_config(path="configs/default.yaml"):
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config


def build_model(config):

    print("Loading MedSAM backbone...")

    sam_model = sam_model_registry["vit_b"](
        checkpoint=config["model"]["medsam_checkpoint"]
    )

    model = GraphMedSAM(config, sam_model)

    return model


def build_dataloaders(config):

    transforms = Compose([
        RandomFlip3D(),
        IntensityScale(),
        RandomNoise()
    ])

    train_dataset = MedicalDataset(
        config["dataset"]["train_images"],
        config["dataset"]["train_masks"],
        transform=transforms
    )

    val_dataset = MedicalDataset(
        config["dataset"]["val_images"],
        config["dataset"]["val_masks"],
        transform=None
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2
    )

    return train_loader, val_loader


def build_losses(config):

    criterion_dict = {
        "dice": DiceLoss(),
        "boundary": BoundaryLoss(config),
        "geometry": GeometryLoss(config),
        "topology": TopologyLoss()
    }

    return criterion_dict


def main():

    print("Loading config...")
    config = load_config()

    print("Building model...")
    model = build_model(config)

    print("Preparing datasets...")
    train_loader, val_loader = build_dataloaders(config)

    print("Initializing losses...")
    criterion_dict = build_losses(config)

    print("Initializing trainer...")
    trainer = VesselTrainer(
        model,
        train_loader,
        val_loader,
        criterion_dict,
        config
    )

    epochs = config["training"]["epochs"]

    print("Starting training...")

    for epoch in range(epochs):

        train_loss = trainer.train_epoch(epoch)

        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f}")

        if val_loader is not None:
            val_loss = trainer.validate_epoch(epoch)
            print(f"Epoch {epoch} | Val Loss: {val_loss:.4f}")


if __name__ == "__main__":
    main()