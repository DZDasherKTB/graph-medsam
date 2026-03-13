import torch
import torch.optim as optim
from tqdm import tqdm
import os


class VesselTrainer:

    def __init__(self, model, train_loader, val_loader, criterion_dict, config):

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Loss functions
        self.dice_loss = criterion_dict["dice"]
        self.boundary_loss = criterion_dict["boundary"]
        self.geometry_loss = criterion_dict["geometry"]
        self.topology_loss = criterion_dict["topology"]

        # Training state
        self.current_phase = 1

        # Checkpoint management
        self.checkpoint_dir = "checkpoints"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.best_val = float("inf")

        self.setup_optimizer()


    def setup_optimizer(self):

        params = []

        # -------- Encoder --------
        if self.current_phase <= 2:
            for p in self.model.encoder.parameters():
                p.requires_grad = False

        else:
            for i, block in enumerate(self.model.encoder.blocks):

                trainable = (
                    i >= self.config["model"]["encoder_frozen_layers"]
                )

                for p in block.parameters():
                    p.requires_grad = trainable

            params.append({
                "params": [p for p in self.model.encoder.parameters() if p.requires_grad],
                "lr": float(self.config["training"]["lr_encoder"])
            })

        # -------- Graph module --------
        graph_active = self.current_phase >= 2

        for p in self.model.graph_module.parameters():
            p.requires_grad = graph_active

        if graph_active:
            params.append({
                "params": self.model.graph_module.parameters(),
                "lr": float(self.config["training"]["lr_graph"])
            })

        # -------- Decoder + Fusion --------
        params.append({
            "params": list(self.model.decoder.parameters()) +
                      list(self.model.fusion.parameters()),
            "lr": float(self.config["training"]["lr_decoder"])
        })

        self.optimizer = optim.AdamW(
            params,
            weight_decay=self.config["training"]["weight_decay"]
        )


    def compute_loss(self, logits, target, sdf, epoch):

        weights = self.config["loss_weights"]

        losses = {}

        losses["dice"] = self.dice_loss(logits, target) * weights["dice"]

        losses["boundary"] = (
            self.boundary_loss(logits, sdf) * weights["boundary"]
            if self.current_phase >= 2
            else torch.tensor(0.0, device=self.device)
        )

        losses["geometry"] = (
            self.geometry_loss(logits) * weights["geometry"]
            if self.current_phase >= 3
            else torch.tensor(0.0, device=self.device)
        )

        losses["topology"] = (
            self.topology_loss(logits, target, current_epoch=epoch)
            * weights["topology"]
            if self.current_phase >= 4
            else torch.tensor(0.0, device=self.device)
        )

        total_loss = sum(losses.values())

        return total_loss, losses


    def train_epoch(self, epoch):

        self.model.train()

        self.update_phase(epoch)

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Phase {self.current_phase}]")

        for batch in pbar:

            images = batch["image"].to(self.device)
            masks = batch["mask"].to(self.device)
            sdfs = batch["sdf"].to(self.device)
            coords = batch["coords"].to(self.device)

            self.optimizer.zero_grad()

            logits = self.model(images, coords)

            loss, loss_dict = self.compute_loss(logits, masks, sdfs, epoch)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self.optimizer.step()

            pbar.set_postfix({
                "L": f"{loss.item():.3f}",
                "D": f"{loss_dict['dice'].item():.3f}",
                "G": f"{self.model.fusion.gamma.item():.3f}"
            })

        # Save latest checkpoint
        torch.save(
            self.model.state_dict(),
            os.path.join(self.checkpoint_dir, "last_model.pth")
        )


    def validate(self):

        if self.val_loader is None:
            return None

        self.model.eval()

        total_loss = 0

        with torch.no_grad():

            for batch in self.val_loader:

                images = batch["image"].to(self.device)
                masks = batch["mask"].to(self.device)
                sdfs = batch["sdf"].to(self.device)
                coords = batch["coords"].to(self.device)

                logits = self.model(images, coords)

                loss, _ = self.compute_loss(logits, masks, sdfs, epoch=999)

                total_loss += loss.item()

        return total_loss / len(self.val_loader)


    def save_best(self, val_loss):

        if val_loss is None:
            return

        if val_loss < self.best_val:

            self.best_val = val_loss

            torch.save(
                self.model.state_dict(),
                os.path.join(self.checkpoint_dir, "best_model.pth")
            )

            print("Saved best model")


    def update_phase(self, epoch):

        prev_phase = self.current_phase

        if epoch < 10:
            self.current_phase = 1
        elif epoch < 25:
            self.current_phase = 2
        elif epoch < 60:
            self.current_phase = 3
        else:
            self.current_phase = 4

        if self.current_phase != prev_phase:

            print(f"\n>>> Transitioning to Phase {self.current_phase}")

            self.setup_optimizer()