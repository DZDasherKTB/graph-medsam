import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

class VesselTrainer:
    def __init__(self, model, train_loader, val_loader, criterion_dict, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.dice_loss = criterion_dict['dice']
        self.boundary_loss = criterion_dict['boundary']
        self.geometry_loss = criterion_dict['geometry']
        self.topology_loss = criterion_dict['topology']
        
        self.current_phase = 1
        self.setup_optimizer()

    def setup_optimizer(self):
        """Dynamic parameter management with parameter-level freezing."""
        params = []
        
        # 1. Encoder Management (The Critical Fix)
        if self.current_phase <= 2:
            for p in self.model.encoder.parameters():
                p.requires_grad = False
        else:
            # Unfreeze layers 9-12 by iterating through parameters within blocks
            for i, block in enumerate(self.model.encoder.blocks):
                is_trainable = (i >= self.config['model']['encoder_frozen_layers'])
                for p in block.parameters():
                    p.requires_grad = is_trainable
            
            params.append({
                'params': [p for p in self.model.encoder.parameters() if p.requires_grad],
                'lr': float(self.config['training']['lr_encoder'])
            })

        # 2. Graph Module Management
        graph_active = (self.current_phase >= 2)
        for p in self.model.graph_module.parameters():
            p.requires_grad = graph_active
        
        if graph_active:
            params.append({
                'params': self.model.graph_module.parameters(),
                'lr': float(self.config['training']['lr_graph'])
            })

        # 3. Decoder & Fusion (Always Trainable)
        params.append({
            'params': list(self.model.decoder.parameters()) + list(self.model.fusion.parameters()),
            'lr': float(self.config['training']['lr_decoder'])
        })

        self.optimizer = optim.AdamW(params, weight_decay=self.config['training']['weight_decay'])

    def compute_loss(self, logits, target, sdf, epoch):
        weights = self.config['loss_weights']
        losses = {}
        
        # Base Dice Loss
        losses['dice'] = self.dice_loss(logits, target) * weights['dice']
        
        # Conditional Losses based on Phase
        losses['boundary'] = self.boundary_loss(logits, sdf) * weights['boundary'] if self.current_phase >= 2 else torch.tensor(0.0, device=self.device)
        losses['geometry'] = self.geometry_loss(logits) * weights['geometry'] if self.current_phase >= 3 else torch.tensor(0.0, device=self.device)
        losses['topology'] = self.topology_loss(logits, target, current_epoch=epoch) * weights['topology'] if self.current_phase >= 4 else torch.tensor(0.0, device=self.device)
        
        total_loss = sum(losses.values())
        return total_loss, losses

    def train_epoch(self, epoch):
        self.model.train()
        self.update_phase(epoch)
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Phase {self.current_phase}]")
        for batch in pbar:
            images, masks, sdfs, coords = batch['image'].to(self.device), batch['mask'].to(self.device), batch['sdf'].to(self.device), batch['coords'].to(self.device)
            
            self.optimizer.zero_grad()
            logits = self.model(images, coords)
            
            loss, loss_dict = self.compute_loss(logits, masks, sdfs, epoch)
            loss.backward()
            
            # Stability Trick: Gradient Clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.optimizer.step()
            
            # Enhanced Logging for Debugging
            pbar.set_postfix({
                'L': f"{loss.item():.3f}",
                'D': f"{loss_dict['dice'].item():.3f}",
                'G': f"{self.model.fusion.gamma.item():.3f}"
            })

    def update_phase(self, epoch):
        prev_phase = self.current_phase
        if epoch < 10: self.current_phase = 1
        elif epoch < 25: self.current_phase = 2
        elif epoch < 60: self.current_phase = 3
        else: self.current_phase = 4
            
        if self.current_phase != prev_phase:
            print(f"\n>>> Transitioning to Phase {self.current_phase}")
            self.setup_optimizer()