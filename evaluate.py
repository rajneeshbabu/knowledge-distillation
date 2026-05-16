"""
Evaluate and compare Teacher, Student Baseline, and Distilled Student on CIFAR-10.

Usage:
    python evaluate.py

Outputs a comparison table and per-class breakdown for all available checkpoints.
"""

import os

import torch
import torch.nn.functional as F

from models import get_teacher, StudentCNN
from utils import get_cifar10_loaders, CIFAR10_CLASSES


CHECKPOINTS = {
    "Teacher (ResNet-50)":         ("teacher",  "./checkpoints/teacher_best.pth"),
    "Student Baseline (no KD)":    ("student",  "./checkpoints/student_baseline_best.pth"),
    "Student Distilled (KD)":      ("student",  "./checkpoints/student_distilled_best.pth"),
}


def load_model(model_type: str, ckpt_path: str, device):
    if model_type == "teacher":
        model = get_teacher(num_classes=10, pretrained=False)
    else:
        model = StudentCNN(num_classes=10)

    if not os.path.exists(ckpt_path):
        return None

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device).eval()
    return model


@torch.no_grad()
def evaluate_full(model, loader, device):
    """Return overall accuracy and per-class accuracy."""
    class_correct = [0] * 10
    class_total   = [0] * 10
    total_correct = total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)

        for c in range(10):
            mask = labels == c
            class_correct[c] += (predicted[mask] == c).sum().item()
            class_total[c]   += mask.sum().item()

        total_correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    overall = 100.0 * total_correct / total
    per_class = [
        100.0 * class_correct[c] / class_total[c] if class_total[c] > 0 else 0.0
        for c in range(10)
    ]
    return overall, per_class


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    _, test_loader = get_cifar10_loaders(batch_size=256)

    results = {}
    for name, (model_type, ckpt) in CHECKPOINTS.items():
        model = load_model(model_type, ckpt, device)
        if model is None:
            print(f"[SKIP] {name} — checkpoint not found: {ckpt}")
            continue
        acc, per_class = evaluate_full(model, test_loader, device)
        n_params = count_params(model)
        results[name] = {"acc": acc, "per_class": per_class, "params": n_params}
        print(f"[OK]   {name}: {acc:.2f}%  ({n_params:,} params)")

    if not results:
        print("\nNo checkpoints found. Run training scripts first.")
        return

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "═" * 65)
    print(f"{'Model':<32}  {'Accuracy':>10}  {'Params':>12}")
    print("═" * 65)
    for name, r in results.items():
        print(f"{name:<32}  {r['acc']:>9.2f}%  {r['params']:>12,}")
    print("═" * 65)

    # ── Accuracy gain from distillation ──────────────────────────────────────
    if "Student Baseline (no KD)" in results and "Student Distilled (KD)" in results:
        gain = (results["Student Distilled (KD)"]["acc"]
                - results["Student Baseline (no KD)"]["acc"])
        print(f"\nKnowledge distillation accuracy gain: +{gain:.2f}%")

    # ── Per-class breakdown ───────────────────────────────────────────────────
    if results:
        print(f"\n{'Class':<14}", end="")
        for name in results:
            short = name.split("(")[0].strip()[:16]
            print(f"  {short:>16}", end="")
        print()
        print("-" * (14 + 18 * len(results)))
        for i, cls in enumerate(CIFAR10_CLASSES):
            print(f"{cls:<14}", end="")
            for r in results.values():
                print(f"  {r['per_class'][i]:>15.1f}%", end="")
            print()


if __name__ == "__main__":
    main()
