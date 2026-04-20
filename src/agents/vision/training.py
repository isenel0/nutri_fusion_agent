from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainingConfig:

    dataset_root: Path = Path("nutrition5k")
    metadata_file: Path = Path("nutrition5k/metadata/dish_metadata_cafe1_with_headers.csv")
    batch_size: int = 16
    num_epochs: int = 5
    learning_rate: float = 1e-3
    num_classes: int = 1000  
    output_dir: Path = Path("artifacts")


def build_dataloaders(config: TrainingConfig):
    """TODO: Build Nutrition5K train/val datasets and dataloaders."""
    # Step to implement later:
    # 1) Read metadata CSV
    # 2) Map each sample to image path + label
    # 3) Create train/val split
    # 4) Return DataLoader objects
    raise NotImplementedError("Placeholder: dataloader logic not implemented yet.")


def build_model(config: TrainingConfig):
    """TODO: Create EfficientNet-B0 and adapt final layer for Nutrition5K classes."""
    # Step to implement later:
    # 1) Load EfficientNet-B0 weights
    # 2) Replace classifier head with config.num_classes
    # 3) Return initialized model
    raise NotImplementedError("Placeholder: model setup not implemented yet.")


def train_one_epoch(model, train_loader, optimizer, criterion, device):
    """TODO: Implement one epoch of training."""
    # Step to implement later:
    # - Forward pass
    # - Compute loss
    # - Backpropagation
    # - Optimizer step
    raise NotImplementedError("Placeholder: train loop not implemented yet.")


def validate(model, val_loader, criterion, device):
    """TODO: Implement validation pass and metrics."""
    # Step to implement later:
    # - Run model in eval mode
    # - Compute validation loss/accuracy
    # - Return summary metrics
    raise NotImplementedError("Placeholder: validation loop not implemented yet.")


def save_checkpoint(model, config: TrainingConfig, epoch: int):
    """TODO: Save model checkpoint to config.output_dir."""
    # Step to implement later:
    # - Create output directory
    # - Save model/optimizer state dict
    raise NotImplementedError("Placeholder: checkpoint saving not implemented yet.")


def main() -> None:
    """Entry point for future training pipeline."""
    config = TrainingConfig()

    # Placeholder pipeline order only (no actual training yet).
    # train_loader, val_loader = build_dataloaders(config)
    # model = build_model(config)
    # for epoch in range(config.num_epochs):
    #     train_one_epoch(...)
    #     validate(...)
    #     save_checkpoint(...)

    print("training.py is a placeholder scaffold. We will implement each step together.")
    print(f"Dataset root placeholder: {config.dataset_root}")


if __name__ == "__main__":
    main()
