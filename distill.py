"""
Knowledge Distillation: transfer ResNet-50 teacher knowledge to lightweight student CNN.

Theory
------
Hinton et al. (2015) showed that a small student model can be trained to mimic the
"soft" probability distribution produced by a large teacher rather than just matching
hard one-hot labels. Soft targets carry richer inter-class similarity information —
e.g. the teacher's confidence spread between "cat" and "dog" tells the student more
than a raw 1/0 label.

Loss function
-------------
L_total = α · L_soft + (1 - α) · L_hard

where:
  L_soft = T² · KL( softmax(z_s/T) ‖ softmax(z_t/T) )
           scaled by T² to compensate for gradient magnitude reduction at high T
  L_hard = CrossEntropy(z_s, y)      (standard supervised loss)

Hyperparameters
---------------
  T (temperature): softens both distributions — higher T → softer targets, more
                   information transferred about wrong-class similarities.
                   Typical range: 3–10.
  α (alpha):       weight on soft loss. Typical range: 0.7–0.9.

Usage
-----
    # Step 1 — train teacher (if not already done)
    python train_teacher.py

    # Step 2 — run distillation
    python distill.py [--temperature 4] [--alpha 0.7] [--epochs 60]

Saves
-----
    checkpoints/student_distilled_best.pth
    checkpoints/student_distilled_final.pth
"""

import argparse
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from models import get_teacher, StudentCNN
from utils import get_cifar10_loaders


# ── Distillation loss ─────────────────────────────────────────────────────────

def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 4.0,
    alpha: float = 0.7,
) -> torch.Tensor:
    """
    Combined KL-divergence (soft) + cross-entropy (hard) distillation loss.

    Args:
        student_logits: Raw logits from student [B, C].
        teacher_logits: Raw logits from teacher [B, C].
        labels:         Ground-truth class indices [B].
        temperature:    Softening temperature T (>1 softens distributions).
        alpha:          Weight on soft loss; (1-alpha) on hard loss.

    Returns:
        Scalar loss tensor.
    """
    # Soft loss — KL divergence between temperature-scaled distributions
    soft_student = F.log_softmax(student_logits / temperature, dim=1)
    soft_teacher = F.softmax(teacher_logits / temperature, dim=1)
    # KL(student || teacher) — T² factor restores gradient scale
    L_soft = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (temperature ** 2)

    # Hard loss — standard cross-entropy on ground-truth labels
    L_hard = F.cross_entropy(student_logits, labels)

    return alpha * L_soft + (1.0 - alpha) * L_hard


# ── Training helpers ──────────────────────────────────────────────────────────

def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            _, predicted = model(images).max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    return 100.0 * correct / total


def distill_one_epoch(student, teacher, loader, optimizer, device, temperature, alpha):
    student.train()
    teacher.eval()

    total_loss = correct = total = 0

    with torch.no_grad():
        # We will compute teacher logits inside the loop to save memory
        pass

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        # Teacher forward (no grad needed)
        with torch.no_grad():
            teacher_logits = teacher(images)

        # Student forward + distillation loss
        student_logits = student(images)
        loss = distillation_loss(
            student_logits, teacher_logits, labels,
            temperature=temperature, alpha=alpha
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, predicted = student_logits.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return total_loss / total, 100.0 * correct / total


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Knowledge distillation: ResNet-50 → lightweight CNN")
    p.add_argument("--temperature",  type=float, default=4.0,  help="Softening temperature T")
    p.add_argument("--alpha",        type=float, default=0.7,  help="Weight on soft (KL) loss")
    p.add_argument("--epochs",       type=int,   default=60,   help="Training epochs")
    p.add_argument("--lr",           type=float, default=0.05, help="Initial learning rate for student")
    p.add_argument("--batch-size",   type=int,   default=128)
    p.add_argument("--dropout",      type=float, default=0.3,  help="Dropout for distilled student")
    p.add_argument("--data-dir",     type=str,   default="./data")
    p.add_argument("--ckpt-dir",     type=str,   default="./checkpoints")
    p.add_argument("--teacher-ckpt", type=str,   default="./checkpoints/teacher_best.pth",
                   help="Path to trained teacher weights")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.ckpt_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Temperature T={args.temperature}  |  Alpha α={args.alpha}")

    # ── Data ─────────────────────────────────────────────────────────────────
    train_loader, test_loader = get_cifar10_loaders(
        data_dir=args.data_dir, batch_size=args.batch_size
    )

    # ── Teacher (frozen) ─────────────────────────────────────────────────────
    teacher = get_teacher(num_classes=10, pretrained=False).to(device)
    if not os.path.exists(args.teacher_ckpt):
        raise FileNotFoundError(
            f"Teacher checkpoint not found: {args.teacher_ckpt}\n"
            "Run 'python train_teacher.py' first."
        )
    teacher.load_state_dict(torch.load(args.teacher_ckpt, map_location=device))
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Teacher loaded from: {args.teacher_ckpt}")
    teacher_acc = evaluate(teacher, test_loader, device)
    print(f"Teacher accuracy on test set: {teacher_acc:.2f}%")

    # ── Student ───────────────────────────────────────────────────────────────
    student = StudentCNN(num_classes=10, dropout=args.dropout).to(device)
    print(f"Student CNN — trainable params: {student.count_parameters():,}")

    # ── Optimiser & scheduler ────────────────────────────────────────────────
    optimizer = optim.SGD(
        student.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    print(f"\n{'Epoch':>6}  {'Loss':>9}  {'Train%':>8}  {'Val%':>8}  {'Time':>6}")
    print("-" * 48)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        loss, train_acc = distill_one_epoch(
            student, teacher, train_loader, optimizer, device,
            temperature=args.temperature, alpha=args.alpha
        )
        val_acc = evaluate(student, test_loader, device)
        scheduler.step()

        elapsed = time.time() - t0
        flag = " ★" if val_acc > best_acc else ""
        print(f"{epoch:>6}  {loss:>9.5f}  {train_acc:>7.2f}%  {val_acc:>7.2f}%  {elapsed:>5.1f}s{flag}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(student.state_dict(),
                       os.path.join(args.ckpt_dir, "student_distilled_best.pth"))

    torch.save(student.state_dict(),
               os.path.join(args.ckpt_dir, "student_distilled_final.pth"))

    print(f"\n{'─'*48}")
    print(f"Teacher accuracy:              {teacher_acc:.2f}%")
    print(f"Student (distilled) accuracy:  {best_acc:.2f}%")
    print(f"{'─'*48}")
    print(f"Checkpoints saved to: {args.ckpt_dir}/")


if __name__ == "__main__":
    main()
