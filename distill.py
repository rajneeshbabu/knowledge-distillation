"""Knowledge distillation: ResNet-50 teacher -> small CNN student on CIFAR-10.

Three models are trained and compared:
  1. teacher   - ImageNet-pretrained ResNet-50, adapted to 32x32 and fine-tuned
  2. baseline  - a 3-block CNN trained on hard labels only
  3. distilled - the same CNN trained on the teacher's soft labels as well

Loss for the distilled student (Hinton et al. 2015):

    L = alpha * T^2 * KL( softmax(z_s/T) || softmax(z_t/T) ) + (1-alpha) * CE(z_s, y)

The T^2 factor restores the gradient magnitude that dividing the logits by T
removes, so alpha keeps meaning the same thing as T changes.

Teacher logits are computed once and cached, so the distillation run costs no
more than the baseline run.
"""
import json, time, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision
from torchvision import transforms
from torchvision.models import resnet50

torch.manual_seed(0)
np.random.seed(0)
torch.set_num_threads(2)

OUT = Path("runs")
OUT.mkdir(exist_ok=True)
DEVICE = "cpu"

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


N_TRAIN = 20000   # CPU-only budget: a fixed 20k subset of the 50k training images


def subset_indices(n_total=50000, n=N_TRAIN, seed=0):
    g = np.random.default_rng(seed)
    return np.sort(g.choice(n_total, size=n, replace=False))


TEACHER_RES = 64   # the pretrained stem expects to downsample; 32x32 is too small for it


def _tf(train, resize=None):
    ops = []
    if train:
        ops += [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
    if resize:
        ops += [transforms.Resize(resize, antialias=True)]
    return transforms.Compose(ops + [transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


def loaders(batch_size=48, root="data"):
    train_tf = _tf(train=True)
    test_tf = _tf(train=False)
    train_tf_t = _tf(train=True, resize=TEACHER_RES)    # teacher: upsampled
    test_tf_t = _tf(train=False, resize=TEACHER_RES)
    plain_tf_t = _tf(train=False, resize=TEACHER_RES)
    idx = subset_indices()
    C = lambda tf, train=True: torchvision.datasets.CIFAR10(root, train=train, download=True, transform=tf)

    train = Subset(C(train_tf), idx)                  # student training images
    train_t = Subset(C(train_tf_t), idx)              # same images, teacher resolution
    plain_t = Subset(C(plain_tf_t), idx)              # un-augmented, for the logit cache
    test = C(test_tf, train=False)
    test_t = C(test_tf_t, train=False)
    # a fixed 2k slice of the test set for the per-epoch curve; the headline
    # numbers at the end are always measured on the full 10k test set
    L = lambda ds, sh=False: DataLoader(ds, batch_size, shuffle=sh, num_workers=0)
    return {
        "train": L(train, True), "train_t": L(train_t, True),
        "plain_t": L(plain_t), "test": L(test), "test_t": L(test_t),
        "quick": L(Subset(test, list(range(2000)))),
        "quick_t": L(Subset(test_t, list(range(2000)))),
    }


# ------------------------------------------------------------------ models --
def build_teacher():
    """ResNet-50 pretrained on ImageNet, with its stem left alone.

    The obvious CIFAR adaptation is to replace the 7x7 stride-2 stem and drop
    the max-pool, so a 32x32 image is not immediately reduced to 8x8. Tried
    first, it is a mistake: the replacement stem is randomly initialised, so
    the pretrained blocks downstream receive features they were never trained
    on, and a few epochs are nowhere near enough to recover. That teacher
    reached 82.1% -- below the 83.9% of the student it was meant to teach, and
    distilling from it COST 3.0 points (results_weak_teacher.json).

    Keeping the stem and feeding it CIFAR upsampled to 64x64 instead lets every
    pretrained weight do the job it was trained for. It is also ~2.8x cheaper
    per step than the stride-1 version, because layer1 then runs at 16x16
    rather than 32x32.
    """
    model = resnet50(weights="IMAGENET1K_V1")
    model.fc = nn.Linear(2048, 10)
    # freeze the first two stages: their low-level filters transfer as they are,
    # and not backpropagating through them roughly halves the step cost
    for name, param in model.named_parameters():
        if name.startswith(("layer1", "layer2")):
            param.requires_grad = False
    return model


class StudentCNN(nn.Module):
    """Three conv blocks, ~0.5M parameters -- about 2% of the teacher."""

    def __init__(self, n_classes=10):
        super().__init__()
        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )
        self.features = nn.Sequential(block(3, 32), block(32, 64), block(64, 128))
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(0.1), nn.Linear(128, n_classes))

    def forward(self, x):
        return self.classifier(self.features(x))


# --------------------------------------------------------------- utilities --
@torch.no_grad()
def accuracy(model, loader):
    model.eval()
    correct = total = 0
    for x, y in loader:
        correct += (model(x).argmax(1) == y).sum().item()
        total += y.size(0)
    return correct / total


def log(msg):
    print(msg, flush=True)
    with open(OUT / "progress.log", "a") as f:
        f.write(msg + "\n")


# ----------------------------------------------------------------- training --
def train_teacher(train_loader, test_loader, epochs=3, lr=1e-3):
    model = build_teacher()
    ckpt = OUT / "teacher.pt"
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt))
        acc = accuracy(model, test_loader)
        log(f"teacher restored from checkpoint: test acc {acc:.4f}")
        return model, acc
    trainable = [p for p in model.parameters() if p.requires_grad]
    log(f"teacher: {sum(p.numel() for p in model.parameters()):,} params "
        f"({sum(p.numel() for p in trainable):,} trainable)")
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train(); started = time.time(); total = seen = 0
        for i, (x, y) in enumerate(train_loader):
            loss = criterion(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item(); seen += 1
            if i % 50 == 0:
                log(f"  teacher epoch {epoch} batch {i}/{len(train_loader)} "
                    f"loss {total/seen:.4f} [{time.time()-started:.0f}s]")
        sched.step()
        acc = accuracy(model, test_loader)
        log(f"teacher epoch {epoch}: loss {total/seen:.4f}  test acc {acc:.4f}  "
            f"[{(time.time()-started)/60:.1f} min]")
        torch.save(model.state_dict(), OUT / f"teacher_ep{epoch}.pt")
    torch.save(model.state_dict(), OUT / "teacher.pt")
    return model, acc


@torch.no_grad()
def cache_teacher_logits(model, plain_loader):
    """One forward pass over the training set; the distillation run then costs
    the same as the baseline run."""
    cache = OUT / "teacher_logits.pt"
    if cache.exists():
        logits = torch.load(cache)
        log(f"teacher logits restored from cache {tuple(logits.shape)}")
        return logits
    model.eval()
    chunks = []
    started = time.time()
    for i, (x, _) in enumerate(plain_loader):
        chunks.append(model(x))
        if i % 100 == 0:
            log(f"  caching logits {i}/{len(plain_loader)} [{time.time()-started:.0f}s]")
    logits = torch.cat(chunks)
    torch.save(logits, OUT / "teacher_logits.pt")
    log(f"cached teacher logits {tuple(logits.shape)} [{(time.time()-started)/60:.1f} min]")
    return logits


def train_student(train_loader, test_loader, epochs=15, lr=2e-3,
                  teacher_logits=None, temperature=4.0, alpha=0.7, label=""):
    model = StudentCNN()
    log(f"{label}: {sum(p.numel() for p in model.parameters()):,} params")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    history = []

    for epoch in range(1, epochs + 1):
        model.train(); started = time.time(); total = seen = 0
        for x, y, *rest in train_loader:
            logits = model(x)
            hard = F.cross_entropy(logits, y)
            if teacher_logits is not None:
                t_logits = rest[0]
                soft = F.kl_div(F.log_softmax(logits / temperature, dim=1),
                                F.softmax(t_logits / temperature, dim=1),
                                reduction="batchmean") * (temperature ** 2)
                loss = alpha * soft + (1 - alpha) * hard
            else:
                loss = hard
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item(); seen += 1
        sched.step()
        acc = accuracy(model, test_loader)
        history.append({"epoch": epoch, "loss": total / seen, "test_acc": acc})
        log(f"{label} epoch {epoch:2d}: loss {total/seen:.4f}  test acc {acc:.4f}  "
            f"[{time.time()-started:.0f}s]")
    torch.save(model.state_dict(), OUT / f"{label}.pt")
    return model, history, acc


def indexed_loader(logits, batch_size=48):  # must match the baseline loader exactly
    """Training loader that also yields the cached teacher logits."""
    full = torchvision.datasets.CIFAR10("data", train=True,
                                        download=False, transform=_tf(train=True))
    base = Subset(full, subset_indices())

    class WithLogits(torch.utils.data.Dataset):
        def __len__(self):
            return len(base)

        def __getitem__(self, i):
            x, y = base[i]          # i indexes the subset, exactly as the
            return x, y, logits[i]  # cached logits do -- same order, same images

    return DataLoader(WithLogits(), batch_size, shuffle=True, num_workers=0)


def main():
    L = loaders()
    log(f"CIFAR-10: {len(L['train'].dataset):,} train / {len(L['test'].dataset):,} test")
    log(f"teacher sees {TEACHER_RES}x{TEACHER_RES}; students see native 32x32")

    teacher, _ = train_teacher(L["train_t"], L["quick_t"], epochs=8)
    logits = cache_teacher_logits(teacher, L["plain_t"])

    base_model, base_hist, _ = train_student(L["train"], L["quick"],
                                             epochs=20, label="student_baseline")
    kd_model, kd_hist, _ = train_student(indexed_loader(logits), L["quick"], epochs=20,
                                         teacher_logits=logits, temperature=4.0,
                                         alpha=0.7, label="student_distilled")

    # headline numbers: full 10,000-image test set, final weights
    log("scoring all three models on the full test set...")
    teacher_acc = accuracy(teacher, L["test_t"])
    base_acc = accuracy(base_model, L["test"])
    kd_acc = accuracy(kd_model, L["test"])

    student_params = sum(p.numel() for p in StudentCNN().parameters())
    teacher_params = sum(p.numel() for p in build_teacher().parameters())
    results = {
        "teacher_acc": teacher_acc, "baseline_acc": base_acc, "distilled_acc": kd_acc,
        "gain": kd_acc - base_acc,
        "teacher_params": teacher_params, "student_params": student_params,
        "compression": teacher_params / student_params,
        "baseline_history": base_hist, "distilled_history": kd_hist,
        "n_train": N_TRAIN, "n_test": len(L["test"].dataset),
        "teacher_res": TEACHER_RES, "teacher_epochs": 8, "student_epochs": 20,
        "temperature": 4.0, "alpha": 0.7,
    }
    json.dump(results, open(OUT / "results.json", "w"), indent=1)
    log("\n=== RESULTS ===")
    log(f"teacher  ResNet-50      {teacher_acc:.4f}   {teacher_params:,} params")
    log(f"student  baseline       {base_acc:.4f}   {student_params:,} params")
    log(f"student  distilled      {kd_acc:.4f}   {student_params:,} params")
    log(f"distillation gain       {kd_acc-base_acc:+.4f}")
    log(f"compression             {teacher_params/student_params:.0f}x fewer parameters")


if __name__ == "__main__":
    main()
