"""
Main entry point for World Model training.

Usage:
    python src/main.py --config configs/pong.yaml
    python src/main.py --game breakout --epochs 100  # CLI overrides
"""

import argparse
import sys
import os
import yaml

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.world_model import WorldModel
from utils.dataset import create_dataloader
from utils.train import train


def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(description="Train World Model on Atari")

    # Config file
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file"
    )

    # Dataset (override config)
    parser.add_argument("--game", type=str, default=None, help="Atari game name")
    parser.add_argument(
        "--dataset_type",
        type=str,
        default=None,
        choices=["mixed", "medium", "expert"],
        help="D4RL dataset type",
    )
    parser.add_argument(
        "--sequence_length", type=int, default=None, help="Sequence length for STU"
    )
    parser.add_argument(
        "--frame_stack", type=int, default=None, help="Number of frames to stack"
    )

    # Model (override config)
    parser.add_argument(
        "--num_channels", type=int, default=None, help="Latent channel dimension"
    )
    parser.add_argument(
        "--num_blocks", type=int, default=None, help="Residual blocks per network"
    )
    parser.add_argument(
        "--action_space_size", type=int, default=None, help="Number of actions"
    )

    # Training (override config)
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")

    # Loss weights (override config)
    parser.add_argument(
        "--latent_weight", type=float, default=None, help="Latent loss weight"
    )
    parser.add_argument(
        "--recon_weight", type=float, default=None, help="Reconstruction loss weight"
    )
    parser.add_argument(
        "--reward_weight", type=float, default=None, help="Reward loss weight"
    )

    # Saving (override config)
    parser.add_argument(
        "--save_path", type=str, default=None, help="Checkpoint save path"
    )
    parser.add_argument(
        "--save_interval", type=int, default=None, help="Save every N epochs"
    )
    parser.add_argument(
        "--print_interval", type=int, default=None, help="Print every N batches"
    )

    return parser.parse_args()


def merge_config(config, args):
    """Merge config file with CLI arguments (CLI takes precedence)."""
    # Dataset
    if args.game is not None:
        config["dataset"]["game"] = args.game
    if args.dataset_type is not None:
        config["dataset"]["dataset_type"] = args.dataset_type
    if args.sequence_length is not None:
        config["dataset"]["sequence_length"] = args.sequence_length
    if args.frame_stack is not None:
        config["dataset"]["frame_stack"] = args.frame_stack

    # Model
    if args.num_channels is not None:
        config["model"]["num_channels"] = args.num_channels
    if args.num_blocks is not None:
        config["model"]["num_blocks"] = args.num_blocks
    if args.action_space_size is not None:
        config["model"]["action_space_size"] = args.action_space_size

    # Training
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.epochs is not None:
        config["training"]["num_epochs"] = args.epochs
    if args.lr is not None:
        config["training"]["learning_rate"] = args.lr
    if args.device is not None:
        config["training"]["device"] = args.device

    # Loss weights
    if args.latent_weight is not None:
        config["loss"]["latent_weight"] = args.latent_weight
    if args.recon_weight is not None:
        config["loss"]["reconstruction_weight"] = args.recon_weight
    if args.reward_weight is not None:
        config["loss"]["reward_weight"] = args.reward_weight

    # Checkpoint
    if args.save_path is not None:
        config["checkpoint"]["save_path"] = args.save_path
    if args.save_interval is not None:
        config["checkpoint"]["save_interval"] = args.save_interval
    if args.print_interval is not None:
        config["checkpoint"]["print_interval"] = args.print_interval

    return config


def get_default_config():
    """Return default configuration."""
    return {
        "dataset": {
            "game": "pong",
            "dataset_type": "mixed",
            "sequence_length": 16,
            "frame_stack": 4,
        },
        "model": {
            "num_channels": 64,
            "num_blocks": 2,
            "action_space_size": 18,
            "spatial_size": 5,
        },
        "training": {
            "batch_size": 32,
            "num_epochs": 100,
            "learning_rate": 1e-4,
            "weight_decay": 1e-5,
            "num_workers": 4,
            "device": "cuda",
        },
        "loss": {
            "latent_weight": 1.0,
            "reconstruction_weight": 0.5,
            "reward_weight": 1.0,
        },
        "checkpoint": {
            "save_path": "checkpoints/world_model.pt",
            "save_interval": 10,
            "print_interval": 100,
        },
        "logging": {
            "log_dir": "logs/",
            "experiment_name": "world_model",
        },
    }


def main():
    args = parse_args()

    # Load config
    if args.config:
        config = load_config(args.config)
    else:
        config = get_default_config()

    # Merge with CLI overrides
    config = merge_config(config, args)

    # Extract settings
    ds = config["dataset"]
    model_cfg = config["model"]
    train_cfg = config["training"]
    loss_cfg = config["loss"]
    ckpt_cfg = config["checkpoint"]

    print("=" * 60)
    print("World Model Training")
    print("=" * 60)
    print(f"Game: {ds['game']}")
    print(f"Device: {train_cfg['device']}")
    print(f"Sequence length: {ds['sequence_length']}")
    print(f"Batch size: {train_cfg['batch_size']}")
    print(f"Epochs: {train_cfg['num_epochs']}")
    print(f"Learning rate: {train_cfg['learning_rate']}")
    print("=" * 60)

    # Observation shape: [frame_stack, height, width]
    observation_shape = (ds["frame_stack"], 84, 84)

    # Create model
    print("\nInitializing World Model...")
    model = WorldModel(
        observation_shape=observation_shape,
        num_channels=model_cfg["num_channels"],
        num_blocks=model_cfg["num_blocks"],
        action_space_size=model_cfg["action_space_size"],
        sequence_length=ds["sequence_length"],
        spatial_size=model_cfg.get("spatial_size", 84 // 16),
    )

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Create dataloader
    print(f"\nLoading {ds['game']} dataset...")
    train_loader = create_dataloader(
        game=ds["game"],
        dataset_type=ds["dataset_type"],
        sequence_length=ds["sequence_length"],
        frame_stack=ds["frame_stack"],
        batch_size=train_cfg["batch_size"],
        num_workers=train_cfg["num_workers"],
        shuffle=True,
    )

    # Create checkpoint directory
    os.makedirs(os.path.dirname(ckpt_cfg["save_path"]), exist_ok=True)

    # Save config alongside checkpoint
    config_save_path = ckpt_cfg["save_path"].replace(".pt", "_config.yaml")
    with open(config_save_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"Saved config to {config_save_path}")

    # Train
    print("\nStarting training...")
    trainer = train(
        model=model,
        dataloader=train_loader,
        device=train_cfg["device"],
        num_epochs=train_cfg["num_epochs"],
        save_path=ckpt_cfg["save_path"],
        save_interval=ckpt_cfg["save_interval"],
        print_interval=ckpt_cfg["print_interval"],
        lr=train_cfg["learning_rate"],
        latent_loss_weight=loss_cfg["latent_weight"],
        recon_loss_weight=loss_cfg["reconstruction_weight"],
        reward_loss_weight=loss_cfg["reward_weight"],
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
