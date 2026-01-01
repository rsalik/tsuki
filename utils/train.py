"""
Training loop for World Model with hybrid losses.

Losses:
    1. Latent consistency: predicted latent vs encoded(actual next frame)
    2. Reconstruction: decoded prediction vs actual next frame
    3. Reward prediction: predicted reward vs actual reward
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
from tqdm import tqdm


class WorldModelTrainer:
    """Trainer for World Model with hybrid losses."""

    def __init__(
        self,
        model,
        device="cuda",
        lr=1e-4,
        weight_decay=1e-5,
        latent_loss_weight=1.0,
        recon_loss_weight=0.5,
        reward_loss_weight=1.0,
    ):
        """
        Args:
            model: WorldModel instance
            device: 'cuda' or 'cpu'
            lr: Learning rate
            weight_decay: Weight decay for AdamW
            latent_loss_weight: Weight for latent consistency loss
            recon_loss_weight: Weight for reconstruction loss
            reward_loss_weight: Weight for reward prediction loss
        """
        self.model = model.to(device)
        self.device = device

        self.latent_loss_weight = latent_loss_weight
        self.recon_loss_weight = recon_loss_weight
        self.reward_loss_weight = reward_loss_weight

        self.optimizer = AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        self.scheduler = None  # Set in train()

    def compute_losses(self, batch):
        """
        Compute all losses for a batch.

        Args:
            batch: dict with 'observations', 'actions', 'rewards', 'next_observations'

        Returns:
            dict with individual losses and total loss
        """
        observations = batch["observations"].to(self.device)  # [B, T, C, H, W]
        actions = batch["actions"].to(self.device)  # [B, T]
        rewards = batch["rewards"].to(self.device)  # [B, T]
        next_observations = batch["next_observations"].to(
            self.device
        )  # [B, T, C, H, W]

        # Forward pass
        outputs = self.model(observations, actions)

        predicted_latents = outputs["predicted_latents"]  # [B, T, C, H', W']
        predicted_observations = outputs["predicted_observations"]  # [B, T, C, H, W]
        predicted_rewards = outputs["predicted_rewards"]  # [B, T, 1]

        # Encode actual next observations (with stop gradient for target)
        with torch.no_grad():
            target_latents = self.model.encode(next_observations)  # [B, T, C, H', W']

        # 1. Latent consistency loss
        latent_loss = F.mse_loss(predicted_latents, target_latents)

        # 2. Reconstruction loss
        recon_loss = F.mse_loss(predicted_observations, next_observations)

        # 3. Reward prediction loss
        reward_loss = F.mse_loss(predicted_rewards.squeeze(-1), rewards)

        # Total loss
        total_loss = (
            self.latent_loss_weight * latent_loss
            + self.recon_loss_weight * recon_loss
            + self.reward_loss_weight * reward_loss
        )

        return {
            "total_loss": total_loss,
            "latent_loss": latent_loss,
            "recon_loss": recon_loss,
            "reward_loss": reward_loss,
        }

    def train_step(self, batch):
        """Single training step."""
        self.model.train()
        self.optimizer.zero_grad()

        losses = self.compute_losses(batch)
        losses["total_loss"].backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()

        return {k: v.item() for k, v in losses.items()}

    @torch.no_grad()
    def validate(self, val_dataloader):
        """Validate on validation set."""
        self.model.eval()
        total_losses = {
            "total_loss": 0,
            "latent_loss": 0,
            "recon_loss": 0,
            "reward_loss": 0,
        }
        num_batches = 0

        for batch in val_dataloader:
            losses = self.compute_losses(batch)
            for k, v in losses.items():
                total_losses[k] += v.item()
            num_batches += 1

        return {k: v / num_batches for k, v in total_losses.items()}

    def save_checkpoint(self, path, epoch, best_loss):
        """Save model checkpoint."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": (
                    self.scheduler.state_dict() if self.scheduler else None
                ),
                "best_loss": best_loss,
            },
            path,
        )
        print(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler and checkpoint["scheduler_state_dict"]:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        return checkpoint["epoch"], checkpoint["best_loss"]


def train(
    model,
    dataloader,
    optimizer=None,  # Unused, kept for API compatibility
    loss_fn=None,  # Unused, kept for API compatibility
    device="cuda",
    num_epochs=100,
    save_path="checkpoints/world_model.pt",
    save_interval=10,
    print_interval=100,
    val_dataloader=None,
    lr=1e-4,
    latent_loss_weight=1.0,
    recon_loss_weight=0.5,
    reward_loss_weight=1.0,
):
    """
    Train the World Model.

    Args:
        model: WorldModel instance
        dataloader: Training DataLoader
        device: 'cuda' or 'cpu'
        num_epochs: Number of training epochs
        save_path: Path to save checkpoints
        save_interval: Save checkpoint every N epochs
        print_interval: Print losses every N batches
        val_dataloader: Optional validation DataLoader
        lr: Learning rate
        latent_loss_weight: Weight for latent consistency loss
        recon_loss_weight: Weight for reconstruction loss
        reward_loss_weight: Weight for reward prediction loss
    """
    trainer = WorldModelTrainer(
        model=model,
        device=device,
        lr=lr,
        latent_loss_weight=latent_loss_weight,
        recon_loss_weight=recon_loss_weight,
        reward_loss_weight=reward_loss_weight,
    )

    # Learning rate scheduler
    trainer.scheduler = CosineAnnealingLR(trainer.optimizer, T_max=num_epochs)

    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 50)

        # Training
        epoch_losses = {
            "total_loss": 0,
            "latent_loss": 0,
            "recon_loss": 0,
            "reward_loss": 0,
        }
        num_batches = 0

        pbar = tqdm(dataloader, desc="Training")
        for batch_idx, batch in enumerate(pbar):
            losses = trainer.train_step(batch)

            for k, v in losses.items():
                epoch_losses[k] += v
            num_batches += 1

            if (batch_idx + 1) % print_interval == 0:
                avg_loss = epoch_losses["total_loss"] / num_batches
                pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

        # Average epoch losses
        avg_losses = {k: v / num_batches for k, v in epoch_losses.items()}
        print(
            f"Train - Total: {avg_losses['total_loss']:.4f}, "
            f"Latent: {avg_losses['latent_loss']:.4f}, "
            f"Recon: {avg_losses['recon_loss']:.4f}, "
            f"Reward: {avg_losses['reward_loss']:.4f}"
        )

        # Validation
        if val_dataloader:
            val_losses = trainer.validate(val_dataloader)
            print(
                f"Val   - Total: {val_losses['total_loss']:.4f}, "
                f"Latent: {val_losses['latent_loss']:.4f}, "
                f"Recon: {val_losses['recon_loss']:.4f}, "
                f"Reward: {val_losses['reward_loss']:.4f}"
            )

            # Save best model
            if val_losses["total_loss"] < best_val_loss:
                best_val_loss = val_losses["total_loss"]
                trainer.save_checkpoint(
                    save_path.replace(".pt", "_best.pt"), epoch, best_val_loss
                )

        # Periodic save
        if (epoch + 1) % save_interval == 0:
            trainer.save_checkpoint(save_path, epoch, best_val_loss)

        # Step scheduler
        trainer.scheduler.step()

    # Final save
    trainer.save_checkpoint(save_path, num_epochs - 1, best_val_loss)
    print(f"\nTraining complete! Best val loss: {best_val_loss:.4f}")

    return trainer
