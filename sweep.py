"""Alpha sweep for the distilled student, with an honest model-selection protocol.

The first run at alpha=0.7 gave +0.04 points over the baseline -- nothing. Either
distillation does not help this student, or 0.7 is the wrong balance. This
answers that by sweeping alpha with everything else held fixed.

Protocol, so the answer means something:

  * 2,000 of the 20,000 training images are held out as a validation set. No
    model trains on them.
  * every run -- baseline and each alpha -- trains on the same 18,000 images,
    same seed, same schedule, same augmentation.
  * alpha is chosen by VALIDATION accuracy, and only the chosen model is then
    scored on the test set. Choosing by test accuracy and reporting that number
    is how a sweep turns into a lie.

The teacher and its cached logits are reused from the main run.
"""
import json, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision

from distill import (MEAN, STD, N_TRAIN, OUT, StudentCNN, _tf, accuracy,
                    build_teacher, log, subset_indices)

ALPHAS = [0.3, 0.5, 0.7, 0.9]
TEMPERATURE = 4.0
EPOCHS = 20
N_VAL = 2000
ROOT = "data"


def build_splits():
    """18k train / 2k val, carved from the same 20k subset the teacher scored."""
    idx = subset_indices()                       # positions into the full 50k
    rng = np.random.default_rng(1)
    order = rng.permutation(len(idx))            # positions into the 20k subset
    val_pos, train_pos = order[:N_VAL], order[N_VAL:]
    return np.sort(train_pos), np.sort(val_pos), idx


def make_loaders(train_pos, val_pos, idx, logits, batch_size=48):
    aug = torchvision.datasets.CIFAR10(ROOT, train=True, transform=_tf(train=True))
    plain = torchvision.datasets.CIFAR10(ROOT, train=True, transform=_tf(train=False))
    test = torchvision.datasets.CIFAR10(ROOT, train=False, transform=_tf(train=False))

    class TrainSet(torch.utils.data.Dataset):
        """Yields (image, label, teacher_logits) for the 18k training positions."""

        def __len__(self):
            return len(train_pos)

        def __getitem__(self, i):
            pos = train_pos[i]                   # position within the 20k subset
            x, y = aug[idx[pos]]
            return x, y, logits[pos]             # cache is indexed by subset position

    val = Subset(plain, [idx[p] for p in val_pos])
    return (DataLoader(TrainSet(), batch_size, shuffle=True, num_workers=0),
            DataLoader(val, batch_size, num_workers=0),
            DataLoader(test, batch_size, num_workers=0))


def train(train_loader, val_loader, alpha=None, lr=2e-3, label=""):
    """alpha=None trains on hard labels only; otherwise the KD objective."""
    torch.manual_seed(0)                          # identical initialisation every run
    model = StudentCNN()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train(); started = time.time(); total = seen = 0
        for x, y, t_logits in train_loader:
            logits = model(x)
            hard = F.cross_entropy(logits, y)
            if alpha is None:
                loss = hard
            else:
                soft = F.kl_div(F.log_softmax(logits / TEMPERATURE, dim=1),
                                F.softmax(t_logits / TEMPERATURE, dim=1),
                                reduction="batchmean") * (TEMPERATURE ** 2)
                loss = alpha * soft + (1 - alpha) * hard
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item(); seen += 1
        sched.step()
        acc = accuracy(model, val_loader)
        history.append({"epoch": epoch, "loss": total / seen, "val_acc": acc})
        log(f"{label} epoch {epoch:2d}: loss {total/seen:.4f}  val acc {acc:.4f}  "
            f"[{time.time()-started:.0f}s]")
    torch.save(model.state_dict(), OUT / f"sweep_{label}.pt")
    return model, history, acc


def main():
    logits = torch.load(OUT / "teacher_logits.pt")
    train_pos, val_pos, idx = build_splits()
    train_loader, val_loader, test_loader = make_loaders(train_pos, val_pos, idx, logits)
    log(f"sweep: {len(train_pos):,} train / {len(val_pos):,} val / "
        f"{len(test_loader.dataset):,} test")

    runs = {}
    model, hist, val_acc = train(train_loader, val_loader, alpha=None, label="baseline")
    runs["baseline"] = {"alpha": None, "val_acc": val_acc, "history": hist,
                        "test_acc": accuracy(model, test_loader)}
    log(f"baseline: val {val_acc:.4f}  test {runs['baseline']['test_acc']:.4f}")

    for a in ALPHAS:
        model, hist, val_acc = train(train_loader, val_loader, alpha=a, label=f"alpha{a}")
        runs[f"alpha{a}"] = {"alpha": a, "val_acc": val_acc, "history": hist,
                             "test_acc": accuracy(model, test_loader)}
        log(f"alpha={a}: val {val_acc:.4f}  test {runs[f'alpha{a}']['test_acc']:.4f}")

    # selection happens on validation, full stop
    kd_runs = {k: v for k, v in runs.items() if v["alpha"] is not None}
    best = max(kd_runs, key=lambda k: kd_runs[k]["val_acc"])
    out = {
        "runs": runs, "selected": best,
        "selected_alpha": runs[best]["alpha"],
        "selected_test_acc": runs[best]["test_acc"],
        "baseline_test_acc": runs["baseline"]["test_acc"],
        "gain": runs[best]["test_acc"] - runs["baseline"]["test_acc"],
        "temperature": TEMPERATURE, "epochs": EPOCHS,
        "n_train": len(train_pos), "n_val": len(val_pos),
        "n_test": len(test_loader.dataset),
    }
    json.dump(out, open(OUT / "sweep_results.json", "w"), indent=1)

    log("\n=== SWEEP ===")
    log(f"{'run':<12}{'alpha':>7}{'val':>9}{'test':>9}")
    for k, v in runs.items():
        a = "--" if v["alpha"] is None else f"{v['alpha']:.1f}"
        log(f"{k:<12}{a:>7}{v['val_acc']*100:>8.2f}%{v['test_acc']*100:>8.2f}%")
    log(f"\nselected on validation: {best} (alpha={runs[best]['alpha']})")
    log(f"test accuracy          : {out['selected_test_acc']*100:.2f}% "
        f"vs baseline {out['baseline_test_acc']*100:.2f}%  ({out['gain']*100:+.2f} pts)")


if __name__ == "__main__":
    main()
