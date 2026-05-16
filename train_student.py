"""
Train the lightweight student CNN on CIFAR-10 WITHOUT knowledge distillation.

This establishes the baseline accuracy (~70%) before distillation is applied.

Usage:
    python train_student.py [--epochs 50] [--lr 0.1] [--batch-size 128]

Saves:
    checkpoints/student_baseline_best.pth
    checkpoints/student_baseline_final.pth
"""

import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim

from models import StudentCNN
from utils import get_cifar10_loaders


def parse_args():
    p = argparse.ArgumentParser(description="Train student CNN baseline on CIFAR-10 (no distillation)")
    p.add_argument("--epochs",     type=int,   default=50,   help="Number of training epochs")
    p.add_argument("--lr",         type=float, default=0.1,  help="Initial learning rate")
    p.add_argument("--batch-size", type=int,   default=128)
    p.add_argument("--dropout",    type=float, default=0.5,  help="Dropout probability in student")
    p.add_argument("--data-dir",   type=str,   default="./data")
    p.add_argument("--ckpt-dir",   type=str,   default="./checkpoints")
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

    return running_loss / total, 100.0 * correct / total


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
    model = StudentCNN(num_classes=10, dropout=args.dropout).to(device)
    print(f"Student CNN — trainable params: {model.count_parameters():,}")

    # ── Training setup ───────────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4
    )
    # Reduce LR at epoch 30 and 40
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 40], gamma=0.1)

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
            torch.save(model.state_dict(),
                       os.path.join(args.ckpt_dir, "student_baseline_best.pth"))

    torch.save(model.state_dict(),
               os.path.join(args.ckpt_dir, "student_baseline_final.pth"))
    print(f"\nBaseline student accuracy: {best_acc:.2f}%")


if __name__ == "__main__":
    main()
