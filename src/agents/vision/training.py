from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import random
from typing import Iterable, Optional

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision import models


@dataclass
class TrainingConfig:

    dataset_root: Path = Path("nutrition5k")
    metadata_file: Path = Path("nutrition5k/metadata/dish_metadata_cafe1_with_headers.csv")
    training_metadata_file: Path = Path("nutrition5k/metadata/normalized/dishes_training.csv")
    realsense_dir: Path = Path("nutrition5k/realsense_overhead")
    image_name: str = "rgb.png"
    mask_dir: Path = Path("nutrition5k/masks")
    image_size: int = 224
    train_split: float = 0.9
    seed: int = 42
    batch_size: int = 16
    num_epochs: int = 5
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    use_pretrained: bool = False
    pin_memory: bool = False
    output_dir: Path = Path("artifacts")


@dataclass(frozen=True)
class MassSample:
    dish_id: str
    image_path: Path
    total_mass: float


class MassRegressionDataset(Dataset):
    def __init__(
        self,
        samples: list[MassSample],
        config: TrainingConfig,
        augment: bool = False,
    ) -> None:
        self.samples = samples
        self.config = config
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]

        with Image.open(sample.image_path) as img:
            rgb = img.convert("RGB")

        mask_path = mask_path_for_dish(self.config, sample.dish_id)
        if mask_path.exists():
            with Image.open(mask_path) as mask_img:
                mask = mask_img.convert("L")
        else:
            mask = Image.new("L", rgb.size, 0)

        target_size = (self.config.image_size, self.config.image_size)
        rgb = TF.resize(rgb, target_size, interpolation=Image.BILINEAR)
        mask = TF.resize(mask, target_size, interpolation=Image.NEAREST)

        if self.augment:
            if torch.rand(1).item() < 0.5:
                rgb = TF.hflip(rgb)
                mask = TF.hflip(mask)

        rgb_tensor = TF.to_tensor(rgb)
        mask_tensor = TF.to_tensor(mask)

        if mask_tensor.shape[0] != 1:
            mask_tensor = mask_tensor[:1]

        input_tensor = torch.cat([rgb_tensor, mask_tensor], dim=0)
        target = torch.tensor([sample.total_mass], dtype=torch.float32)

        return input_tensor, target


def _iter_training_rows(path: Path) -> Iterable[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row:
                continue
            yield row


def build_mass_index(config: TrainingConfig) -> list[MassSample]:
    """Build dish_id -> rgb image path -> total_mass index from Nutrition5k metadata."""
    if not config.training_metadata_file.exists():
        raise FileNotFoundError(
            f"Training metadata not found: {config.training_metadata_file}"
        )

    index: list[MassSample] = []
    for row in _iter_training_rows(config.training_metadata_file):
        dish_id = (row.get("dish_id") or "").strip()
        if not dish_id:
            continue
        total_mass_raw = (row.get("total_mass") or "").strip()
        if not total_mass_raw:
            continue
        try:
            total_mass = float(total_mass_raw)
        except ValueError:
            continue

        image_path = config.realsense_dir / dish_id / config.image_name
        if not image_path.exists():
            continue

        index.append(
            MassSample(
                dish_id=dish_id,
                image_path=image_path,
                total_mass=total_mass,
            )
        )

    if not index:
        raise RuntimeError(
            "No valid dishes found. Check metadata and realsense_overhead paths."
        )

    return index


def mask_path_for_dish(config: TrainingConfig, dish_id: str) -> Path:
    return config.mask_dir / f"{dish_id}.png"


def split_index(
    index: list[MassSample],
    train_split: float,
    seed: int,
) -> tuple[list[MassSample], list[MassSample]]:
    if not 0.0 < train_split < 1.0:
        raise ValueError("train_split must be between 0 and 1.")

    rng = random.Random(seed)
    shuffled = list(index)
    rng.shuffle(shuffled)

    train_size = int(len(shuffled) * train_split)
    train = shuffled[:train_size]
    val = shuffled[train_size:]

    if not train or not val:
        raise RuntimeError(
            f"Split produced empty set (train={len(train)}, val={len(val)})."
        )

    return train, val


def build_dataloaders(config: TrainingConfig):
    """Build Nutrition5K train/val datasets and dataloaders."""
    index = build_mass_index(config)
    train_index, val_index = split_index(index, config.train_split, config.seed)

    train_dataset = MassRegressionDataset(train_index, config, augment=True)
    val_dataset = MassRegressionDataset(val_index, config, augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=config.pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=config.pin_memory,
    )

    return train_loader, val_loader


def build_model(config: TrainingConfig):
    """Create a 4-channel ResNet50 regressor for total mass."""
    weights = models.ResNet50_Weights.DEFAULT if config.use_pretrained else None
    model = models.resnet50(weights=weights)

    old_conv = model.conv1
    new_conv = nn.Conv2d(
        4,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=old_conv.bias is not None,
    )

    with torch.no_grad():
        new_conv.weight[:, :3] = old_conv.weight
        if new_conv.weight.shape[1] > 3:
            mean_weight = old_conv.weight.mean(dim=1, keepdim=True)
            new_conv.weight[:, 3:4] = mean_weight

    model.conv1 = new_conv
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model


def train_one_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_samples = 0

    for inputs, targets in train_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        batch_size = inputs.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / max(total_samples, 1)


def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            batch_size = inputs.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    return total_loss / max(total_samples, 1)


def save_checkpoint(model, config: TrainingConfig, epoch: int):
    """Save model checkpoint to config.output_dir."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = config.output_dir / f"mass_regressor_epoch_{epoch}.pt"
    torch.save({"epoch": epoch, "model_state": model.state_dict()}, checkpoint_path)
    return checkpoint_path


def main() -> None:
    """Entry point for future training pipeline."""
    config = TrainingConfig()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    config.pin_memory = device.type == "cuda"
    print(f"Training device: {device}", flush=True)

    train_loader, val_loader = build_dataloaders(config)

    model = build_model(config).to(device)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    best_val = float("inf")
    for epoch in range(1, config.num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch}/{config.num_epochs} - train MAE: {train_loss:.4f} - val MAE: {val_loss:.4f}",
            flush=True,
        )

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(model, config, epoch)


if __name__ == "__main__":
    main()
