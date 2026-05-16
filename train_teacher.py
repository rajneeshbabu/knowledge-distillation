"""
Train (fine-tune) the ResNet-50 teacher on CIFAR-10.

Usage:
    python train_teacher.py [--epochs 30] [--lr 0.01] [--batch-size 128]

Saves:
    checkpoints/teacher_best.pth   ← best val-accuracy checkpoint
    checkpoints/teacher_final.pth  ← weights after last epoch
"""

import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim

from models import get_teacher
from utils import get_cifar10_loaders


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune ResNet-50 teacher on CIFAR-10")
    p.add_argument("--epochs",     type=int,   default=30,    help="Number of training epochs")
    p.add_argument("--lr",         type=float, default=0.01,  help="Initial learning rate")
    p.add_argument("--batch-size", type=int,   default=128,   help="Mini-batch size")
    p.add_argument("--data-dir",   type=str,   default="./data")
    p.add_argument("--ckpt-dir",   type=str,   default="./checkpoints")
    p.add_argument("--no-pretrain",action="store_true", help="Train from scratch (no ImageNet weights)")
    return p.parse_args()


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    return 100.0 * correct / total


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / total
    acc = 100.0 * correct / total
    return avg_loss, acc


def main():
    args = parse_args()
    os.makedirs(args.ckpt_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Data ─────────────────────────────────────────────────────────────────
    train_loader, test_loader = get_cifar10_loaders(
        data_dir=args.data_dir, batch_size=args.batch_size
    )

    # ── Model ────────────────────────────────────────────────────────────────
    model = get_teacher(num_classes=10, pretrained=not args.no_pretrain).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Teacher (ResNet-50) — trainable params: {n_params:,}")

    # ── Training setup ───────────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss()
    # Use different LRs: lower for pretrained backbone, higher for new head
    backbone_params = [p for name, p in model.named_parameters() if "fc" not in name]
    head_params     = list(model.fc.parameters())

    optimizer = optim.SGD(
        [{"params": backbone_params, "lr": args.lr * 0.1},
         {"params": head_params,     "lr": args.lr}],
        momentum=0.9, weight_decay=5e-4
    )
    # Cosine annealing — no restarts
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    print(f"\n{'Epoch':>6}  {'Loss':>8}  {'Train%':>8}  {'Val%':>8}  {'Time':>6}")
    print("-" * 46)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_acc = evaluate(model, test_loader, device)
        scheduler.step()

        elapsed = time.time() - t0
        flag = " ★" if val_acc > best_acc else ""
        print(f"{epoch:>6}  {train_loss:>8.4f}  {train_acc:>7.2f}%  {val_acc:>7.2f}%  {elapsed:>5.1f}s{flag}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.ckpt_dir, "teacher_best.pth"))

    torch.save(model.state_dict(), os.path.join(args.ckpt_dir, "teacher_final.pth"))
    print(f"\nBest validation accuracy: {best_acc:.2f}%")
    print(f"Checkpoints saved to: {args.ckpt_dir}/")


if __name__ == "__main__":
    main()
