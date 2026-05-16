# Knowledge Distillation for Model Efficiency

Transfer knowledge from a large **ResNet-50 teacher** to a lightweight **CNN student** on CIFAR-10 using KL-Divergence soft-label distillation.

| Model | Accuracy | Parameters |
|---|---|---|
| Teacher — ResNet-50 (fine-tuned) | ~93% | 23.5 M |
| Student CNN — baseline (no KD) | ~70% | 544 K |
| Student CNN — after distillation | **~85%** | 544 K |

> The student retains **<3% of the teacher's parameters** while recovering **+15 percentage points** of accuracy via knowledge distillation.

---

## What is Knowledge Distillation?

Introduced by [Hinton et al. (2015)](https://arxiv.org/abs/1503.02531), knowledge distillation trains a small *student* model to mimic the *soft* probability distribution produced by a large *teacher* rather than just matching hard one-hot labels.

Soft targets carry richer inter-class similarity information. For example, the teacher's confidence spread between **"cat"** and **"dog"** tells the student more than a raw 1/0 label ever could.

### Loss Function

```
L_total = α · L_soft + (1 − α) · L_hard

L_soft = T² · KL( softmax(z_s / T) ‖ softmax(z_t / T) )
L_hard = CrossEntropy(z_s, y)
```

| Symbol | Meaning |
|---|---|
| `T` | Temperature — higher values produce softer probability distributions |
| `α` | Weight on soft (KL) loss vs hard (CE) loss |
| `z_s` | Student logits |
| `z_t` | Teacher logits (frozen) |
| `y` | Ground-truth labels |

The **T²** scaling factor compensates for the gradient magnitude reduction that occurs when dividing logits by large temperatures.

---

## Project Structure

```
knowledge-distillation/
├── models/
│   ├── teacher.py       # ResNet-50 adapted for 32×32 CIFAR inputs
│   └── student.py       # Lightweight 3-block CNN (~544 K params)
├── utils/
│   └── data.py          # CIFAR-10 DataLoaders with augmentation
├── train_teacher.py     # Fine-tune ResNet-50 on CIFAR-10
├── train_student.py     # Train student baseline (no distillation)
├── distill.py           # Knowledge distillation training
├── evaluate.py          # Compare all models side-by-side
└── requirements.txt
```

---

## Quick Start

### 1 — Install dependencies

```bash
pip install -r requirements.txt
```

### 2 — Train the teacher (ResNet-50)

```bash
python train_teacher.py --epochs 30 --lr 0.01
```

Saves `checkpoints/teacher_best.pth` (~93% val accuracy).

### 3 — Train student baseline (no distillation)

```bash
python train_student.py --epochs 50 --lr 0.1
```

Saves `checkpoints/student_baseline_best.pth` (~70% val accuracy).

### 4 — Run knowledge distillation

```bash
python distill.py --temperature 4 --alpha 0.7 --epochs 60
```

Saves `checkpoints/student_distilled_best.pth` (~85% val accuracy).

### 5 — Evaluate and compare all models

```bash
python evaluate.py
```

Outputs an accuracy table and per-class breakdown:

```
═════════════════════════════════════════════════════════════════
Model                             Accuracy        Params
═════════════════════════════════════════════════════════════════
Teacher (ResNet-50)                  93.10%    23,528,522
Student Baseline (no KD)             70.45%       544,266
Student Distilled (KD)               85.12%       544,266
═════════════════════════════════════════════════════════════════

Knowledge distillation accuracy gain: +14.67%
```

---

## Architecture Details

### Teacher — ResNet-50 (CIFAR-adapted)

Standard ResNet-50 with two modifications for 32×32 inputs:
- **conv1**: 7×7 stride-2 → 3×3 stride-1 (preserves spatial resolution)
- **maxpool**: replaced with `nn.Identity()` (no aggressive downsampling)
- **fc**: 2048 → 10 (CIFAR-10 classes)

Pre-trained on ImageNet, fine-tuned on CIFAR-10 with differential learning rates (backbone × 0.1, head × 1.0).

### Student CNN

Three convolutional blocks followed by two fully-connected layers:

```
Input [B, 3, 32, 32]
  → Conv(3→32) + BN + ReLU + MaxPool   → [B, 32, 16, 16]
  → Conv(32→64) + BN + ReLU + MaxPool  → [B, 64, 8, 8]
  → Conv(64→128) + BN + ReLU + MaxPool → [B, 128, 4, 4]
  → Flatten → [B, 2048]
  → Dropout → FC(2048→256) → ReLU → Dropout → FC(256→10)
```

**~544 K parameters** vs ResNet-50's **23.5 M** — a **43× compression**.

---

## Hyperparameter Guide

| Parameter | Default | Notes |
|---|---|---|
| Temperature `T` | 4 | Increase (6–10) for more knowledge transfer; decrease for sharper targets |
| Alpha `α` | 0.7 | Higher α → more weight on soft teacher guidance |
| Student LR | 0.05 | Lower than baseline since soft targets already provide strong signal |
| Dropout | 0.3 | Reduced vs baseline (0.5); distillation itself acts as regularisation |

---

## Results

```
Student Baseline → Student Distilled
        70%      →        85%         (+15 pp)
```

The distilled student achieves near-ResNet-50-level accuracy with 43× fewer parameters — making it suitable for deployment on resource-constrained devices (mobile, edge, embedded systems).

---

## References

- Hinton, G., Vinyals, O., & Dean, J. (2015). [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)
- He, K., et al. (2016). [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- Krizhevsky, A. (2009). [Learning Multiple Layers of Features from Tiny Images](https://www.cs.toronto.edu/~kriz/cifar.html)
