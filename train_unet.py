import argparse
import glob
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from PIL import Image


# ---------------------------------------------------------------------------
# Model: small U-Net with a pretrained ResNet18 encoder
# ---------------------------------------------------------------------------
class UNetResNet18(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = torchvision.models.resnet18(weights=weights)

        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)  # /2, 64ch
        self.pool = resnet.maxpool                                       # /4
        self.layer1 = resnet.layer1                                      # /4, 64ch
        self.layer2 = resnet.layer2                                      # /8, 128ch
        self.layer3 = resnet.layer3                                      # /16, 256ch
        self.layer4 = resnet.layer4                                      # /32, 512ch

        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = nn.Sequential(nn.Conv2d(256 + 256, 256, 3, padding=1), nn.ReLU(inplace=True))

        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = nn.Sequential(nn.Conv2d(128 + 128, 128, 3, padding=1), nn.ReLU(inplace=True))

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = nn.Sequential(nn.Conv2d(64 + 64, 64, 3, padding=1), nn.ReLU(inplace=True))

        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = nn.Sequential(nn.Conv2d(32 + 64, 32, 3, padding=1), nn.ReLU(inplace=True))

        self.up0 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.out_conv = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        # Pad up to a multiple of 32 so the 4 downsample/upsample stages
        # align exactly, then crop back to the original size at the end.
        # This makes the model robust to any input resolution, not just
        # ones you've carefully chosen to divide evenly by 32.
        h, w = x.shape[-2:]
        pad_h = (32 - h % 32) % 32
        pad_w = (32 - w % 32) % 32
        x = nn.functional.pad(x, (0, pad_w, 0, pad_h))

        x0 = self.stem(x)      # /2,  64ch
        x1 = self.pool(x0)
        x1 = self.layer1(x1)   # /4,  64ch
        x2 = self.layer2(x1)   # /8,  128ch
        x3 = self.layer3(x2)   # /16, 256ch
        x4 = self.layer4(x3)   # /32, 512ch

        d4 = self.up4(x4)
        d4 = self.dec4(torch.cat([d4, x3], dim=1))

        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, x2], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, x1], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, x0], dim=1))

        d0 = self.up0(d1)
        out = torch.sigmoid(self.out_conv(d0))
        return out[..., :h, :w]  # crop back off the padding


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class RHEEDHeatmapDataset(Dataset):
    def __init__(self, images_dir, heatmaps_dir, frame_names, size=(275, 344), augment=False):
        self.images_dir = images_dir
        self.heatmaps_dir = heatmaps_dir
        self.frame_names = frame_names
        self.size = size  # (height, width)
        self.augment = augment

    def __len__(self):
        return len(self.frame_names)

    def _load_resized(self, name):
        img = Image.open(os.path.join(self.images_dir, name)).convert("L")
        img = img.resize((self.size[1], self.size[0]), Image.BILINEAR)
        img = np.array(img, dtype=np.float32) / 255.0

        heatmap_path = os.path.join(self.heatmaps_dir, name.replace(".png", ".npy"))
        heatmap = np.load(heatmap_path).astype(np.float32)
        heatmap_img = Image.fromarray((heatmap * 255).astype(np.uint8))
        heatmap_img = heatmap_img.resize((self.size[1], self.size[0]), Image.BILINEAR)
        heatmap = np.array(heatmap_img, dtype=np.float32) / 255.0

        return img, heatmap

    def _augment(self, img, heatmap):
        if random.random() < 0.5:
            shift = random.randint(-15, 15)
            img = np.roll(img, shift, axis=1)
            heatmap = np.roll(heatmap, shift, axis=1)
        if random.random() < 0.5:
            img = np.clip(img * random.uniform(0.8, 1.2), 0, 1)
        if random.random() < 0.5:
            img = np.clip(img + random.uniform(-0.05, 0.05), 0, 1)
        if random.random() < 0.3:
            img = np.clip(img + np.random.normal(0, 0.02, img.shape), 0, 1).astype(np.float32)
        return img, heatmap

    def __getitem__(self, idx):
        name = self.frame_names[idx]
        img, heatmap = self._load_resized(name)
        if self.augment:
            img, heatmap = self._augment(img, heatmap)
        img_t = torch.from_numpy(img.copy()).unsqueeze(0).float()
        heatmap_t = torch.from_numpy(heatmap.copy()).unsqueeze(0).float()
        return img_t, heatmap_t


def weighted_mse_loss(pred, target, pos_weight=80.0):
    """
    Plain MSE fails here: line pixels are <0.5% of the image, so a model
    that predicts all-zero everywhere already gets a very low loss and
    has no gradient pressure to ever predict anything else. This upweights
    the loss wherever the target heatmap is hot, so getting the lines
    wrong actually costs the model something.
    """
    weight = 1.0 + pos_weight * target
    return torch.mean(weight * (pred - target) ** 2)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", required=True,
                         help="Folder of CLEAN frames (no annotation overlay!) matching your labels")
    parser.add_argument("--heatmaps_dir", required=True, help="Output folder from render_heatmaps.py")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--height", type=int, default=275)
    parser.add_argument("--width", type=int, default=344)
    parser.add_argument("--val_frames", default="", help="Comma-separated frame filenames to hold out, e.g. 7.png,8.png")
    parser.add_argument("--checkpoint", default="unet_lines.pt")
    parser.add_argument("--no_pretrained", action="store_true", help="Train encoder from scratch instead")
    parser.add_argument("--pos_weight", type=float, default=80.0,
                         help="How much more heavily to weight line pixels vs. background in the loss")
    args = parser.parse_args()

    all_frames = sorted(os.path.basename(p) for p in glob.glob(os.path.join(args.heatmaps_dir, "*.npy")))
    all_frames = [f.replace(".npy", ".png") for f in all_frames]
    val_frames = set(f for f in args.val_frames.split(",") if f)
    train_frames = [f for f in all_frames if f not in val_frames]

    print(f"Training on {len(train_frames)} frame(s): {train_frames}")
    if val_frames:
        print(f"Holding out {len(val_frames)} frame(s) for validation: {sorted(val_frames)}")

    size = (args.height, args.width)
    train_ds = RHEEDHeatmapDataset(args.images_dir, args.heatmaps_dir, train_frames, size=size, augment=True)
    train_loader = DataLoader(train_ds, batch_size=min(4, len(train_ds)), shuffle=True)

    val_loader = None
    if val_frames:
        val_ds = RHEEDHeatmapDataset(args.images_dir, args.heatmaps_dir, sorted(val_frames), size=size, augment=False)
        val_loader = DataLoader(val_ds, batch_size=len(val_ds), shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = UNetResNet18(pretrained=not args.no_pretrained).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = lambda pred, target: weighted_mse_loss(pred, target, pos_weight=args.pos_weight)

    best_val_loss = float("inf")
    best_epoch = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for imgs, heatmaps in train_loader:
            imgs, heatmaps = imgs.to(device), heatmaps.to(device)
            optimizer.zero_grad()
            preds = model(imgs)
            loss = loss_fn(preds, heatmaps)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
        train_loss /= len(train_ds)

        if epoch % 10 == 0 or epoch == 1:
            msg = f"Epoch {epoch:4d}/{args.epochs}  train_loss={train_loss:.5f}"
            if val_loader is not None:
                model.eval()
                with torch.no_grad():
                    for imgs, heatmaps in val_loader:
                        imgs, heatmaps = imgs.to(device), heatmaps.to(device)
                        val_loss = loss_fn(model(imgs), heatmaps).item()
                msg += f"  val_loss={val_loss:.5f}"
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    torch.save({
                        "model_state_dict": model.state_dict(),
                        "size": size,
                        "epoch": epoch,
                        "val_loss": val_loss,
                        "val_frames": sorted(val_frames),
                    }, args.checkpoint)
                    msg += "  [saved best]"
            print(msg)

    if val_loader is None:
        torch.save({"model_state_dict": model.state_dict(), "size": size}, args.checkpoint)
        print(f"\nSaved final trained model to {args.checkpoint}")
    else:
        print(f"\nSaved best model from epoch {best_epoch} to {args.checkpoint} "
              f"(val_loss={best_val_loss:.5f})")


if __name__ == "__main__":
    main()
