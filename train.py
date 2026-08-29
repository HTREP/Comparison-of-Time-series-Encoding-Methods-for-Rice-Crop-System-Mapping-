import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, classification_report,
                             cohen_kappa_score, confusion_matrix, f1_score)
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset

from image_encoding import TRANSFORMS, encode_batch, scale_1d
from models import build_model, count_parameters

CLASSES = ["SRC", "DRC", "HRC", "TRC"]
N_FOLDS = 5
RANDOM_STATE = 42
LR = 1e-4
WEIGHT_DECAY = 1e-2
BATCH_SIZE = 128
MAX_EPOCHS = 500
ES_PATIENCE = 100


def load_split(path):
    df = pd.read_csv(path)
    meta = [c for c in df.columns if not c.startswith("t")]
    tcols = sorted([c for c in df.columns if c.startswith("t")],
                   key=lambda c: int(c[1:]))
    X = df[tcols].to_numpy(dtype=np.float32)
    y = np.array([CLASSES.index(v) for v in df["class"]], dtype=np.int64)
    return X, y, meta


def make_inputs(arm, series):
    return scale_1d(series) if arm == "1d" else encode_batch(arm, series)


def loaders(Xtr, ytr, Xva, yva, device):
    tr = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr))
    va = TensorDataset(torch.from_numpy(Xva), torch.from_numpy(yva))
    return (DataLoader(tr, batch_size=BATCH_SIZE, shuffle=True),
            DataLoader(va, batch_size=BATCH_SIZE, shuffle=False))


def evaluate(model, loader, device):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in loader:
            out = model(xb.to(device))
            preds.append(out.argmax(1).cpu().numpy())
            trues.append(yb.numpy())
    return np.concatenate(preds), np.concatenate(trues)


def train_model(model, tr_loader, va_loader, device, max_epochs, use_es):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    crit = nn.CrossEntropyLoss()
    best_f1, best_state, best_epoch, stale = -1.0, None, 0, 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        for xb, yb in tr_loader:
            opt.zero_grad()
            loss = crit(model(xb.to(device)), yb.to(device))
            loss.backward()
            opt.step()

        if va_loader is None:
            continue

        p, t = evaluate(model, va_loader, device)
        f1 = f1_score(t, p, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1, best_epoch, stale = f1, epoch, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            stale += 1
            if use_es and stale >= ES_PATIENCE:
                break

    return best_state, best_f1, best_epoch


def run_arm(arm, Xtr, ytr, Xev, yev, device, outdir, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    Atr = make_inputs(arm, Xtr)
    Aev = make_inputs(arm, Xev)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                          random_state=RANDOM_STATE)
    fold_f1, fold_epochs = [], []
    oof = np.zeros_like(ytr)

    for k, (i_tr, i_va) in enumerate(skf.split(Atr, ytr), 1):
        tr_loader, va_loader = loaders(Atr[i_tr], ytr[i_tr],
                                       Atr[i_va], ytr[i_va], device)
        model = build_model(arm, len(CLASSES))
        state, f1, epoch = train_model(model, tr_loader, va_loader, device,
                                       MAX_EPOCHS, use_es=True)
        model.load_state_dict(state)
        p, _ = evaluate(model, va_loader, device)
        oof[i_va] = p
        fold_f1.append(f1)
        fold_epochs.append(epoch)
        print(f"  fold {k}/{N_FOLDS}  macro-F1 {f1:.3f}  best epoch {epoch}")

    final_epochs = max(50, int(round(np.mean(fold_epochs))))
    full_loader, _ = loaders(Atr, ytr, Atr, ytr, device)
    model = build_model(arm, len(CLASSES))
    train_model(model, full_loader, None, device, final_epochs, use_es=False)

    ev_loader = DataLoader(TensorDataset(torch.from_numpy(Aev),
                                         torch.from_numpy(yev)),
                           batch_size=BATCH_SIZE, shuffle=False)
    pred, true = evaluate(model, ev_loader, device)

    oa = accuracy_score(true, pred)
    mf1 = f1_score(true, pred, average="macro", zero_division=0)
    kappa = cohen_kappa_score(true, pred)
    cm = confusion_matrix(true, pred, labels=range(len(CLASSES)))
    rep = classification_report(true, pred, target_names=CLASSES,
                                output_dict=True, zero_division=0)

    os.makedirs(outdir, exist_ok=True)
    pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_csv(
        os.path.join(outdir, f"confusion_{arm}.csv"))
    pd.DataFrame(rep).T.to_csv(os.path.join(outdir, f"report_{arm}.csv"))
    pd.DataFrame({"fold": range(1, N_FOLDS + 1),
                  "macro_f1": fold_f1,
                  "best_epoch": fold_epochs}).to_csv(
        os.path.join(outdir, f"cv_{arm}.csv"), index=False)
    torch.save(model.state_dict(), os.path.join(outdir, f"model_{arm}.pth"))
    np.save(os.path.join(outdir, f"pred_{arm}.npy"), pred)

    return {"arm": arm,
            "params": count_parameters(model),
            "cv_macro_f1_mean": float(np.mean(fold_f1)),
            "cv_macro_f1_std": float(np.std(fold_f1)),
            "oof_macro_f1": float(f1_score(ytr, oof, average="macro",
                                           zero_division=0)),
            "final_epochs": final_epochs,
            "eval_oa": float(oa),
            "eval_macro_f1": float(mf1),
            "eval_kappa": float(kappa)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/train.csv")
    ap.add_argument("--eval", default="data/eval.csv")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--arms", nargs="+", default=TRANSFORMS + ["1d"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    Xtr, ytr, _ = load_split(args.train)
    Xev, yev, _ = load_split(args.eval)
    print(f"train {Xtr.shape} | eval {Xev.shape} | device {device}")

    rows = []
    for arm in args.arms:
        print(f"\n[{arm}]")
        rows.append(run_arm(arm, Xtr, ytr, Xev, yev, device, args.outdir, args.seed))
        r = rows[-1]
        print(f"  OA {r['eval_oa']:.3f} | macro-F1 {r['eval_macro_f1']:.3f} "
              f"| kappa {r['eval_kappa']:.3f}")

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.DataFrame(rows).sort_values("eval_oa", ascending=False)
    df.to_csv(os.path.join(args.outdir, "summary.csv"), index=False)
    with open(os.path.join(args.outdir, "config.json"), "w") as f:
        json.dump({"n_folds": N_FOLDS, "random_state": RANDOM_STATE, "lr": LR,
                   "weight_decay": WEIGHT_DECAY, "batch_size": BATCH_SIZE,
                   "max_epochs": MAX_EPOCHS, "es_patience": ES_PATIENCE,
                   "seed": args.seed}, f, indent=2)
    print("\n" + df.to_string(index=False))


if __name__ == "__main__":
    main()
