import argparse
import glob
import json
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
    def __init__(self, images_dir, heatmaps_dir, frame_names, size=(275, 344), augment=False,
                 erase_prob=0.0, scale_jitter=0.0):
        self.images_dir = images_dir
        self.heatmaps_dir = heatmaps_dir
        self.frame_names = frame_names
        self.size = size  # (height, width)
        self.augment = augment
        self.erase_prob = erase_prob
        self.scale_jitter = scale_jitter

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

    @staticmethod
    def _translate(array, shift_y, shift_x):
        """Translate without wraparound, which would create false edge features."""
        output = np.zeros_like(array)
        h, w = array.shape
        src_y0 = max(0, -shift_y)
        src_y1 = min(h, h - shift_y)
        src_x0 = max(0, -shift_x)
        src_x1 = min(w, w - shift_x)
        dst_y0 = max(0, shift_y)
        dst_y1 = dst_y0 + (src_y1 - src_y0)
        dst_x0 = max(0, shift_x)
        dst_x1 = dst_x0 + (src_x1 - src_x0)
        output[dst_y0:dst_y1, dst_x0:dst_x1] = array[src_y0:src_y1, src_x0:src_x1]
        return output

    @staticmethod
    def _erase_line(img, heatmap, half_height=7, margin_x=10):
        """
        Remove one labeled line from both image and target, so the network
        sees lattice slots that are empty and cannot assume every expected
        order is present. The band is refilled by interpolating each column
        between the rows just outside it, plus noise matching the local
        texture. At least one line is always kept.
        """
        profile = heatmap.max(axis=1)
        rows = [r for r in range(1, len(profile) - 1)
                if profile[r] >= 0.5 and profile[r] >= profile[r - 1]
                and profile[r] > profile[r + 1]]
        if len(rows) < 2:
            return img, heatmap
        row = random.choice(rows)
        h, w = img.shape
        y0, y1 = row - half_height, row + half_height + 1
        if y0 < 1 or y1 > h - 1:
            return img, heatmap
        columns = np.where(heatmap[max(0, row - 2):row + 3].max(axis=0) >= 0.3)[0]
        x0 = max(0, columns.min() - margin_x)
        x1 = min(w, columns.max() + margin_x + 1)
        img = img.copy()
        heatmap = heatmap.copy()
        above = img[y0 - 1, x0:x1]
        below = img[y1, x0:x1]
        t = np.linspace(0, 1, y1 - y0)[:, None]
        texture = np.concatenate([img[max(0, y0 - 6):y0, x0:x1], img[y1:y1 + 6, x0:x1]])
        fill = (1 - t) * above + t * below
        fill += np.random.normal(0, texture.std() * 0.5, fill.shape)
        img[y0:y1, x0:x1] = np.clip(fill, 0, 1)
        heatmap[y0:y1, x0:x1] = 0.0
        return img, heatmap

    @staticmethod
    def _scale_vertical(array, factor, center):
        """Stretch rows about ``center`` by ``factor`` (bilinear), keeping the size."""
        h = array.shape[0]
        source = center + (np.arange(h) - center) / factor
        low = np.floor(source).astype(int)
        frac = (source - low)[:, None]
        valid = (low >= 0) & (low + 1 < h)
        low = np.clip(low, 0, h - 2)
        out = (1 - frac) * array[low] + frac * array[low + 1]
        out[~valid] = 0.0
        return out.astype(np.float32)

    def _augment(self, img, heatmap):
        if self.erase_prob and random.random() < self.erase_prob:
            img, heatmap = self._erase_line(img, heatmap)
        if self.scale_jitter and random.random() < 0.8:
            factor = random.uniform(1 - self.scale_jitter, 1 + self.scale_jitter)
            center = random.uniform(0.3, 0.7) * img.shape[0]
            img = self._scale_vertical(img, factor, center)
            heatmap = self._scale_vertical(heatmap, factor, center)
        # Vertical motion is important here: without it, a tiny dataset lets
        # the network memorize a fixed y-position grid instead of reading spots.
        if random.random() < 0.8:
            shift_x = random.randint(-20, 20)
            shift_y = random.randint(-20, 20)
            img = self._translate(img, shift_y, shift_x)
            heatmap = self._translate(heatmap, shift_y, shift_x)
        if random.random() < 0.5:
            img = np.clip(img * random.uniform(0.85, 1.15), 0, 1)
        if random.random() < 0.5:
            img = np.clip(img + random.uniform(-0.03, 0.03), 0, 1)
        if random.random() < 0.3:
            img = np.clip(img + np.random.normal(0, 0.015, img.shape), 0, 1).astype(np.float32)
        return img, heatmap

    def __getitem__(self, idx):
        name = self.frame_names[idx]
        img, heatmap = self._load_resized(name)
        if self.augment:
            img, heatmap = self._augment(img, heatmap)
        img_t = torch.from_numpy(img.copy()).unsqueeze(0).float()
        heatmap_t = torch.from_numpy(heatmap.copy()).unsqueeze(0).float()
        return img_t, heatmap_t


def weighted_mse_loss(pred, target, pos_weight=30.0):
    """
    Plain MSE fails here: line pixels are <0.5% of the image, so a model
    that predicts all-zero everywhere already gets a very low loss and
    has no gradient pressure to ever predict anything else. This upweights
    the loss wherever the target heatmap is hot, so getting the lines
    wrong actually costs the model something.
    """
    weight = 1.0 + pos_weight * target
    return torch.mean(weight * (pred - target) ** 2)


def soft_dice_score(pred, target, eps=1e-6):
    dims = tuple(range(1, pred.ndim))
    intersection = torch.sum(pred * target, dim=dims)
    denominator = torch.sum(pred.square(), dim=dims) + torch.sum(target.square(), dim=dims)
    return torch.mean((2.0 * intersection + eps) / (denominator + eps))


def combined_heatmap_loss(pred, target, pos_weight=30.0, dice_weight=0.5):
    mse = weighted_mse_loss(pred, target, pos_weight=pos_weight)
    dice = 1.0 - soft_dice_score(pred, target)
    return mse + dice_weight * dice


def threshold_f1_score(pred, target, threshold=0.3, eps=1e-6):
    pred_mask = pred >= threshold
    target_mask = target >= threshold
    dims = tuple(range(1, pred.ndim))
    true_positive = torch.sum(pred_mask & target_mask, dim=dims).float()
    predicted = torch.sum(pred_mask, dim=dims).float()
    actual = torch.sum(target_mask, dim=dims).float()
    return torch.mean((2.0 * true_positive + eps) / (predicted + actual + eps))


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
    parser.add_argument("--exclude_frames", default="",
                        help="Comma-separated frames to leave out entirely (not used for validation)")
    parser.add_argument("--checkpoint", default="unet_lines.pt")
    parser.add_argument("--no_pretrained", action="store_true", help="Train encoder from scratch instead")
    parser.add_argument("--pos_weight", type=float, default=30.0,
                         help="How much more heavily to weight line pixels vs. background in the loss")
    parser.add_argument("--dice_weight", type=float, default=0.5,
                        help="Weight of soft-Dice loss added to weighted MSE")
    parser.add_argument("--erase_prob", type=float, default=0.0,
                        help="Probability of erasing one labeled line from image and target")
    parser.add_argument("--scale_jitter", type=float, default=0.0,
                        help="Max fractional vertical stretch/squash, e.g. 0.1 for +/-10%%")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--patience", type=int, default=40,
                        help="Stop after this many epochs without validation-Dice improvement; 0 disables")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"],
                        help="Force a device; auto prefers cuda, then mps, then cpu")
    args = parser.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.checkpoint)), exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    all_frames = [os.path.basename(p) for p in glob.glob(os.path.join(args.heatmaps_dir, "*.npy"))]
    all_frames = [f.replace(".npy", ".png") for f in all_frames]
    all_frames.sort(key=lambda f: (0, int(os.path.splitext(f)[0]))
                    if os.path.splitext(f)[0].isdigit() else (1, f))
    val_frames = set(f for f in args.val_frames.split(",") if f)
    missing_val = val_frames.difference(all_frames)
    if missing_val:
        raise ValueError(f"Validation frames have no heatmap: {sorted(missing_val)}")
    excluded = set(f for f in args.exclude_frames.split(",") if f)
    train_frames = [f for f in all_frames if f not in val_frames and f not in excluded]
    if not train_frames:
        raise ValueError("No training frames remain after applying --val_frames")

    print(f"Training on {len(train_frames)} frame(s): {train_frames}")
    if val_frames:
        print(f"Holding out {len(val_frames)} frame(s) for validation: {sorted(val_frames)}")

    size = (args.height, args.width)
    train_ds = RHEEDHeatmapDataset(args.images_dir, args.heatmaps_dir, train_frames, size=size, augment=True,
                                   erase_prob=args.erase_prob, scale_jitter=args.scale_jitter)
    train_loader = DataLoader(train_ds, batch_size=min(args.batch_size, len(train_ds)), shuffle=True)

    val_loader = None
    if val_frames:
        val_ds = RHEEDHeatmapDataset(args.images_dir, args.heatmaps_dir, sorted(val_frames), size=size, augment=False)
        val_loader = DataLoader(val_ds, batch_size=len(val_ds), shuffle=False)

    if args.device != "auto":
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    model = UNetResNet18(pretrained=not args.no_pretrained).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3, min_lr=1e-6)
    loss_fn = lambda pred, target: combined_heatmap_loss(
        pred, target, pos_weight=args.pos_weight, dice_weight=args.dice_weight)

    best_val_loss = float("inf")
    best_val_dice = -1.0
    best_epoch = None
    epochs_without_improvement = 0
    history = []

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

        if epoch % args.eval_every == 0 or epoch == 1:
            msg = f"Epoch {epoch:4d}/{args.epochs}  train_loss={train_loss:.5f}"
            record = {"epoch": epoch, "train_loss": train_loss,
                      "lr": optimizer.param_groups[0]["lr"]}
            if val_loader is not None:
                model.eval()
                with torch.no_grad():
                    for imgs, heatmaps in val_loader:
                        imgs, heatmaps = imgs.to(device), heatmaps.to(device)
                        val_preds = model(imgs)
                        val_loss = loss_fn(val_preds, heatmaps).item()
                        val_dice = soft_dice_score(val_preds, heatmaps).item()
                        val_f1 = threshold_f1_score(val_preds, heatmaps).item()
                scheduler.step(val_dice)
                msg += (f"  val_loss={val_loss:.5f}  val_dice={val_dice:.4f}"
                        f"  val_f1={val_f1:.4f}")
                record.update({"val_loss": val_loss, "val_dice": val_dice,
                               "val_f1": val_f1})
                if val_dice > best_val_dice:
                    best_val_loss = val_loss
                    best_val_dice = val_dice
                    best_epoch = epoch
                    epochs_without_improvement = 0
                    torch.save({
                        "model_state_dict": model.state_dict(),
                        "size": size,
                        "epoch": epoch,
                        "val_loss": val_loss,
                        "val_dice": val_dice,
                        "val_f1": val_f1,
                        "val_frames": sorted(val_frames),
                        "pos_weight": args.pos_weight,
                        "dice_weight": args.dice_weight,
                        "seed": args.seed,
                        "erase_prob": args.erase_prob,
                        "scale_jitter": args.scale_jitter,
                    }, args.checkpoint)
                    msg += "  [saved best]"
                else:
                    epochs_without_improvement += args.eval_every
            history.append(record)
            print(msg)
            if (val_loader is not None and args.patience > 0 and
                    epochs_without_improvement >= args.patience):
                print(f"Early stopping: no validation-Dice improvement for "
                      f"{epochs_without_improvement} epochs")
                break

    if val_loader is None:
        torch.save({"model_state_dict": model.state_dict(), "size": size}, args.checkpoint)
        print(f"\nSaved final trained model to {args.checkpoint}")
    else:
        print(f"\nSaved best model from epoch {best_epoch} to {args.checkpoint} "
              f"(val_dice={best_val_dice:.4f}, val_loss={best_val_loss:.5f})")

    history_path = args.checkpoint + ".history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Saved training history to {history_path}")


if __name__ == "__main__":
    main()
