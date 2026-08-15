"""
=============================================================================
Graphite LightGBM + SHAP Malware Analyser
=============================================================================
Optimal pipeline (RQ1 winner: LightGBM | RQ2 winner: SHAP)

FIRST RUN  — trains the model, saves it to disk, then analyses
LATER RUNS — loads the saved model instantly, skips training entirely

Usage
-----
  # First time (trains and saves):
  python lightgbm_shap_analyzer.py --dataset-path ../dataset

  # Every run after (loads saved model, no training):
  python lightgbm_shap_analyzer.py --dataset-path ../dataset

  # Force retrain even if a saved model exists:
  python lightgbm_shap_analyzer.py --dataset-path ../dataset --retrain

  # Analyse a single pickle file:
  python lightgbm_shap_analyzer.py --dataset-path ../dataset \
      --single-sample path/to/sample.pickle

  # Control how many test samples to explain:
  python lightgbm_shap_analyzer.py --dataset-path ../dataset --num-samples 50
=============================================================================
"""
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

import os
import sys
import json
import time
import argparse
import pathlib
import pickle

import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

import shap
import torch
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_curve, auc
)
from lightgbm import LGBMClassifier
from torch_geometric.data import Data

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from dataprocessor_graphs import load_dataset

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
MODEL_DIR   = pathlib.Path(__file__).parent / "saved_model"
MODEL_PATH  = MODEL_DIR / "lightgbm_model.joblib"
EMBEDDER_PATH = MODEL_DIR / "graphite_embedder.joblib"
META_PATH   = MODEL_DIR / "model_meta.json"

TEAL   = "#0F9E85"
CORAL  = "#E8593C"
PURPLE = "#7F77DD"
GRAY   = "#888780"
BG     = "#F8F7F3"


# ─────────────────────────────────────────────────────────────────────────────
# Graphite N-gram Embedder  (standalone, no class-dependency issues)
# ─────────────────────────────────────────────────────────────────────────────
class GraphiteEmbedder:
    """Converts a provenance graph (Data object) into a flat feature vector."""

    def __init__(self, N=4, pool="sum"):
        self.N = N
        pool_map = {"sum": torch.sum, "mean": torch.mean, "max": torch.max}
        self.pool = pool_map[pool]
        self.count_vectorizer = CountVectorizer(ngram_range=(N, N))
        self.nodetype_nodefeats  = None
        self.eventname_edgefeats = None

    # ── internal helpers ──────────────────────────────────────────────────────

    def _sorted_events(self, data: Data, thread_idx: int):
        src, tar = data.edge_index
        out = torch.nonzero(src == thread_idx).flatten()
        inc = torch.nonzero(tar == thread_idx).flatten()
        if out.numel() == 0 and inc.numel() == 0:
            return []
        feats = torch.cat([data.edge_attr[inc], data.edge_attr[out]], dim=0)
        order = torch.argsort(feats[:, -1])
        sorted_f = feats[order]
        idxs = torch.nonzero(sorted_f[:, :-1], as_tuple=False)[:, -1]
        return [self.eventname_edgefeats[i] for i in idxs]

    def _neighbor_nodetypes(self, data: Data, thread_idx: int):
        out = torch.nonzero(data.edge_index[0] == thread_idx).flatten()
        inc = torch.nonzero(data.edge_index[1] == thread_idx).flatten()
        adj = []
        if out.numel() > 0: adj.append(data.edge_index[1, out])
        if inc.numel() > 0: adj.append(data.edge_index[0, inc])
        if not adj:
            return torch.zeros((1, len(self.nodetype_nodefeats)))
        return torch.sum(data.x[torch.unique(torch.cat(adj))], dim=0).view(1, -1)

    def _thread_type_vec(self):
        return torch.tensor(
            [1 if t.lower() == "thread" else 0 for t in self.nodetype_nodefeats]
        )

    # ── public API ────────────────────────────────────────────────────────────

    def fit_vectorizer(self, train_dataset):
        tv = self._thread_type_vec()
        seqs = []
        for data in train_dataset:
            for idx in torch.nonzero(torch.all(torch.eq(data.x, tv), dim=1)).flatten().tolist():
                s = self._sorted_events(data, idx)
                if s: seqs.append(s)
        self.count_vectorizer.fit([" ".join(s) for s in seqs if len(s) >= self.N])

    def embed(self, data: Data) -> torch.Tensor:
        tv = self._thread_type_vec()
        thread_idxs = torch.nonzero(torch.all(torch.eq(data.x, tv), dim=1)).flatten().tolist()
        embs = []
        for idx in thread_idxs:
            seq     = self._sorted_events(data, idx)
            ngram   = self.count_vectorizer.transform([" ".join(seq)]).toarray()
            neigh   = self._neighbor_nodetypes(data, idx)
            embs.append(torch.cat([neigh, torch.Tensor(ngram).view(1, -1)], dim=1))
        if not embs:
            dim = len(self.nodetype_nodefeats) + len(self.count_vectorizer.get_feature_names_out())
            return torch.zeros((1, dim))
        return self.pool(torch.cat(embs, dim=0), dim=0)

    def embed_dataset(self, dataset):
        X, y, names = [], [], []
        for data in dataset:
            X.append(self.embed(data).tolist())
            y.append(1 if "malware" in data.name else 0)
            names.append(data.name)
        return np.array(X), np.array(y), names

    @property
    def feature_names(self):
        nodetype_names = list(self.nodetype_nodefeats)
        ngram_names    = list(self.count_vectorizer.get_feature_names_out())
        return nodetype_names + ngram_names


# ─────────────────────────────────────────────────────────────────────────────
# Model persistence
# ─────────────────────────────────────────────────────────────────────────────

def save_model(model, embedder, meta: dict):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model,    MODEL_PATH)
    joblib.dump(embedder, EMBEDDER_PATH)
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n  [SAVED] Model  → {MODEL_PATH}")
    print(f"  [SAVED] Embedder → {EMBEDDER_PATH}")
    print(f"  [SAVED] Meta   → {META_PATH}")


def load_model():
    model    = joblib.load(MODEL_PATH)
    embedder = joblib.load(EMBEDDER_PATH)
    with open(META_PATH) as f:
        meta = json.load(f)
    print(f"\n  [LOADED] LightGBM model from {MODEL_PATH}")
    print(f"  [LOADED] Trained on: {meta.get('trained_on', '?')} samples")
    print(f"  [LOADED] Test F1={meta.get('test_f1','?'):.4f}  AUC={meta.get('test_auc','?'):.4f}")
    return model, embedder, meta


def model_exists() -> bool:
    return MODEL_PATH.exists() and EMBEDDER_PATH.exists() and META_PATH.exists()


# ─────────────────────────────────────────────────────────────────────────────
# SHAP helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_shap_values(model, X: np.ndarray) -> np.ndarray:
    """Return SHAP values (class-1/malware) for every row in X."""
    explainer  = shap.TreeExplainer(model)
    shap_vals  = explainer.shap_values(X)
    if isinstance(shap_vals, list):
        return shap_vals[1]
    if shap_vals.ndim == 3:
        return shap_vals[:, :, 1]
    return shap_vals


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def _savefig(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [PLOT] {path}")


# 1. Global SHAP Beeswarm ─────────────────────────────────────────────────────
def plot_global_shap_beeswarm(shap_vals, X, feature_names, output_dir, top_k=20):
    """
    Shows WHICH n-gram sequences drive the model globally.
    • Y-axis  : top-20 most influential features (ranked by mean |SHAP|)
    • X-axis  : SHAP value — right=malware signal, left=benign signal
    • Colour  : red=high feature count in that sample, blue=low/absent
    """
    mean_abs  = np.abs(shap_vals).mean(axis=0)
    top_idx   = np.argsort(mean_abs)[::-1][:top_k]
    sv_top    = shap_vals[:, top_idx]
    fv_top    = X[:, top_idx]
    feat_top  = np.array(feature_names)[top_idx]

    fig, ax = plt.subplots(figsize=(11, 0.45 * top_k + 2.5))
    cmap = plt.get_cmap("coolwarm")
    rng  = np.random.default_rng(42)

    for rank in range(top_k - 1, -1, -1):
        y     = top_k - 1 - rank
        sv_c  = sv_top[:, rank]
        fv_c  = fv_top[:, rank]
        fv_min, fv_max = fv_c.min(), fv_c.max()
        fv_norm = (fv_c - fv_min) / (fv_max - fv_min + 1e-9)
        colors = cmap(fv_norm)
        jitter = rng.uniform(-0.28, 0.28, size=len(sv_c))
        ax.scatter(sv_c, y + jitter, c=colors, s=9, alpha=0.7, linewidths=0)

    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--", zorder=1)
    ax.set_yticks(range(top_k))
    ax.set_yticklabels(feat_top[::-1], fontsize=8.5)
    ax.set_xlabel("SHAP value  (impact on malware classification)", fontsize=10)
    ax.set_title(
        f"Global SHAP Beeswarm — LightGBM (Top {top_k} n-gram features)\n"
        "Colour = feature frequency in sample  |  blue = low/absent  |  red = high",
        fontsize=11, fontweight="bold"
    )
    ax.grid(axis="x", alpha=0.3)

    sm   = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.01)
    cbar.set_label("Feature value", rotation=270, labelpad=14, fontsize=9)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["Low", "High"])

    plt.tight_layout()
    _savefig(fig, os.path.join(output_dir, "shap_global_beeswarm.png"))


# 2. SHAP Bar Summary ─────────────────────────────────────────────────────────
def plot_shap_bar_summary(shap_vals, feature_names, output_dir, top_k=20):
    """
    Mean |SHAP| per feature — clean horizontal bar chart.
    Easier to read than beeswarm when you need exact rankings.
    """
    mean_abs = np.abs(shap_vals).mean(axis=0)
    top_idx  = np.argsort(mean_abs)[::-1][:top_k]
    vals     = mean_abs[top_idx]
    names    = [feature_names[i] for i in top_idx]

    fig, ax = plt.subplots(figsize=(10, 0.40 * top_k + 2))
    colors  = [TEAL if "nodetype" not in n else PURPLE for n in names]
    bars    = ax.barh(range(top_k - 1, -1, -1), vals, color=colors[::-1],
                      edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(top_k))
    ax.set_yticklabels(names[::-1], fontsize=8.5)
    ax.set_xlabel("Mean |SHAP| value", fontsize=10)
    ax.set_title(
        f"Top {top_k} Most Influential Features — LightGBM + SHAP\n"
        "(mean absolute SHAP across all test samples)",
        fontsize=11, fontweight="bold"
    )
    ax.grid(axis="x", alpha=0.3)
    for bar, v in zip(reversed(list(bars)), vals):
        ax.text(v + 0.0005, bar.get_y() + bar.get_height() / 2,
                f"{v:.4f}", va="center", fontsize=7.5)

    p1 = mpatches.Patch(color=TEAL,   label="N-gram feature")
    p2 = mpatches.Patch(color=PURPLE, label="Node-type feature")
    ax.legend(handles=[p1, p2], fontsize=9)
    plt.tight_layout()
    _savefig(fig, os.path.join(output_dir, "shap_bar_summary.png"))


# 3. Per-sample SHAP waterfall strip ─────────────────────────────────────────
def plot_sample_shap_waterfall(shap_vals_single, feature_names, sample_name,
                                base_value, pred_proba, output_dir,
                                sample_idx=0, top_k=12):
    """
    Waterfall chart for ONE malware sample.
    Shows exactly HOW the model built up its confidence from the base rate.
    """
    abs_order = np.argsort(np.abs(shap_vals_single))[::-1][:top_k]
    vals      = shap_vals_single[abs_order]
    names     = [feature_names[i] for i in abs_order]

    # Build cumulative bars
    cumulative = base_value
    lefts, widths, colors_bar = [], [], []
    for v in vals:
        lefts.append(min(cumulative, cumulative + v))
        widths.append(abs(v))
        colors_bar.append(CORAL if v > 0 else TEAL)
        cumulative += v

    fig, ax = plt.subplots(figsize=(10, 0.50 * top_k + 2.5))
    y_pos = range(top_k - 1, -1, -1)

    for y, l, w, c, v in zip(y_pos, lefts, widths, colors_bar, vals):
        ax.barh(y, w, left=l, color=c, alpha=0.85, edgecolor="white", linewidth=0.5)
        sign = "+" if v > 0 else ""
        ax.text(l + w / 2, y, f"{sign}{v:.4f}", ha="center", va="center",
                fontsize=7.5, color="white", fontweight="bold")

    ax.axvline(base_value, color="gray", linewidth=1.2, linestyle="--",
               label=f"Base value = {base_value:.3f}")
    ax.axvline(pred_proba, color="black", linewidth=1.5,
               label=f"Prediction = {pred_proba:.3f}")

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(names[::-1], fontsize=8.5)
    ax.set_xlabel("Malware probability contribution", fontsize=10)
    short_name = os.path.basename(sample_name)[:60]
    ax.set_title(
        f"SHAP Waterfall — Sample #{sample_idx + 1}\n{short_name}",
        fontsize=10, fontweight="bold"
    )
    ax.legend(fontsize=9)
    ax.grid(axis="x", alpha=0.3)

    p1 = mpatches.Patch(color=CORAL, label="Pushes toward malware ↑")
    p2 = mpatches.Patch(color=TEAL,  label="Pushes toward benign ↓")
    ax.legend(handles=[p1, p2], fontsize=9, loc="lower right")

    plt.tight_layout()
    fname = f"shap_waterfall_sample_{sample_idx:03d}.png"
    _savefig(fig, os.path.join(output_dir, fname))


# 4. Confidence distribution ──────────────────────────────────────────────────
def plot_confidence_distribution(probs, y_true, output_dir):
    """
    Histogram of model confidence scores split by true class.
    Shows how well-separated the model is — good models have two tight peaks.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    benign_probs  = probs[y_true == 0]
    malware_probs = probs[y_true == 1]

    ax.hist(benign_probs,  bins=25, alpha=0.7, color=TEAL,  label="Benign (true)",  density=True)
    ax.hist(malware_probs, bins=25, alpha=0.7, color=CORAL, label="Malware (true)", density=True)
    ax.axvline(0.5, color="black", linewidth=1.5, linestyle="--", label="Decision threshold (0.5)")
    ax.set_xlabel("Model malware confidence score", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title(
        "LightGBM Confidence Distribution — Test Set\n"
        "(Well-separated peaks = reliable model)",
        fontsize=11, fontweight="bold"
    )
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    _savefig(fig, os.path.join(output_dir, "confidence_distribution.png"))


# 5. ROC curve ────────────────────────────────────────────────────────────────
def plot_roc(y_true, probs, output_dir):
    fpr, tpr, _ = roc_curve(y_true, probs)
    roc_auc     = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color=TEAL, linewidth=2.5,
            label=f"LightGBM (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random")
    ax.fill_between(fpr, tpr, alpha=0.08, color=TEAL)
    ax.set_xlabel("False Positive Rate", fontsize=10)
    ax.set_ylabel("True Positive Rate", fontsize=10)
    ax.set_title("ROC Curve — LightGBM", fontsize=11, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    _savefig(fig, os.path.join(output_dir, "roc_curve.png"))
    return roc_auc


# 6. Confusion matrix ─────────────────────────────────────────────────────────
def plot_confusion(y_true, preds, output_dir):
    cm = confusion_matrix(y_true, preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Benign", "Malware"],
                yticklabels=["Benign", "Malware"],
                linewidths=0.5, linecolor="white")
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("Actual",    fontsize=10)
    ax.set_title(
        f"Confusion Matrix — LightGBM\n"
        f"Acc={accuracy_score(y_true, preds):.4f}  "
        f"F1={f1_score(y_true, preds):.4f}  "
        f"FPR={cm[0,1]/(cm[0,0]+cm[0,1]):.4f}",
        fontsize=10, fontweight="bold"
    )
    plt.tight_layout()
    _savefig(fig, os.path.join(output_dir, "confusion_matrix.png"))


# 7. SHAP dependence plot ─────────────────────────────────────────────────────
def plot_shap_dependence(shap_vals, X, feature_names, output_dir, top_n=6):
    """
    For each of the top-N features: scatter of (feature count) vs (SHAP value).
    Reveals non-linear relationships — e.g. does more of this n-gram always
    mean more malware, or only after a threshold?
    """
    mean_abs = np.abs(shap_vals).mean(axis=0)
    top_idx  = np.argsort(mean_abs)[::-1][:top_n]

    cols = 3
    rows = (top_n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = axes.flatten()

    for i, fi in enumerate(top_idx):
        ax  = axes[i]
        sv  = shap_vals[:, fi]
        fv  = X[:, fi]
        sc  = ax.scatter(fv, sv, c=sv, cmap="coolwarm",
                         alpha=0.6, s=15, linewidths=0)
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_xlabel(f"Feature count", fontsize=8)
        ax.set_ylabel("SHAP value",     fontsize=8)
        name = feature_names[fi]
        ax.set_title(name if len(name) <= 38 else name[:35] + "…",
                     fontsize=8, fontweight="bold")
        fig.colorbar(sc, ax=ax, pad=0.02).set_label("SHAP", fontsize=7)
        ax.grid(alpha=0.25)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        f"SHAP Dependence Plots — Top {top_n} Features (LightGBM)\n"
        "Shows how feature frequency relates to its malware contribution",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout()
    _savefig(fig, os.path.join(output_dir, "shap_dependence_plots.png"))


# 8. SHAP heatmap (top features × samples) ────────────────────────────────────
def plot_shap_heatmap(shap_vals, feature_names, sample_names, output_dir,
                      top_k=15, max_samples=40):
    """
    Heatmap where rows = top features, columns = individual test samples.
    Red cell = feature pushed this sample toward malware.
    Blue cell = pushed toward benign.
    Lets you spot patterns across malware families.
    """
    mean_abs  = np.abs(shap_vals).mean(axis=0)
    top_idx   = np.argsort(mean_abs)[::-1][:top_k]
    sv_plot   = shap_vals[:max_samples, :][:, top_idx].T
    feat_lbls = [feature_names[i] for i in top_idx]
    smp_lbls  = [os.path.basename(n)[:25] for n in sample_names[:max_samples]]

    fig, ax = plt.subplots(figsize=(max(12, len(smp_lbls) * 0.35), top_k * 0.55 + 2))
    vmax = np.percentile(np.abs(sv_plot), 95)
    im   = ax.imshow(sv_plot, cmap="RdBu_r", aspect="auto",
                     vmin=-vmax, vmax=vmax)
    ax.set_yticks(range(top_k))
    ax.set_yticklabels(feat_lbls, fontsize=8)
    ax.set_xticks(range(len(smp_lbls)))
    ax.set_xticklabels(smp_lbls, rotation=90, fontsize=6.5)
    ax.set_title(
        f"SHAP Heatmap — Top {top_k} Features × {len(smp_lbls)} Samples (LightGBM)\n"
        "Red = pushes toward malware  |  Blue = pushes toward benign",
        fontsize=10, fontweight="bold"
    )
    fig.colorbar(im, ax=ax, label="SHAP value", shrink=0.6)
    plt.tight_layout()
    _savefig(fig, os.path.join(output_dir, "shap_heatmap.png"))


# 9. Per-sample confidence bar ────────────────────────────────────────────────
def plot_per_sample_confidence(probs, y_true, names, output_dir, max_show=60):
    """
    Horizontal bar per test sample coloured by true class.
    Instantly shows which samples were borderline vs confidently classified.
    """
    order  = np.argsort(probs)[::-1][:max_show]
    probs_ = probs[order]
    true_  = y_true[order]
    names_ = [os.path.basename(names[i])[:40] for i in order]
    colors = [CORAL if t == 1 else TEAL for t in true_]

    fig, ax = plt.subplots(figsize=(10, max(6, len(order) * 0.26)))
    ax.barh(range(len(order)), probs_, color=colors, alpha=0.8,
            edgecolor="white", linewidth=0.4)
    ax.axvline(0.5, color="black", linewidth=1.2, linestyle="--",
               label="Threshold 0.5")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(names_, fontsize=7)
    ax.set_xlabel("Malware confidence", fontsize=10)
    ax.set_title(
        f"Per-Sample Malware Confidence — LightGBM (top {len(order)} by confidence)\n"
        "Colour = true label",
        fontsize=10, fontweight="bold"
    )
    p1 = mpatches.Patch(color=CORAL, label="True malware")
    p2 = mpatches.Patch(color=TEAL,  label="True benign")
    ax.legend(handles=[p1, p2, ax.lines[0]], fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    _savefig(fig, os.path.join(output_dir, "per_sample_confidence.png"))


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def build_lightgbm():
    return LGBMClassifier(
        n_estimators=500, max_depth=-1, num_leaves=63,
        learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
        min_child_samples=10, random_state=42, n_jobs=-1, verbose=-1
    )


def train(args, nodetype_nodefeats, eventname_edgefeats):
    print("\n[Step 1] Loading datasets...")
    train_ds = load_dataset(
        os.path.join(args.dataset_path, "train/benign"),
        os.path.join(args.dataset_path, "train/malware"),
        len(nodetype_nodefeats), len(eventname_edgefeats) + 1
    )
    test_ds = load_dataset(
        os.path.join(args.dataset_path, "test/benign"),
        os.path.join(args.dataset_path, "test/malware"),
        len(nodetype_nodefeats), len(eventname_edgefeats) + 1
    )

    print("[Step 2] Fitting N-gram vectorizer and extracting embeddings...")
    embedder = GraphiteEmbedder(N=args.N, pool=args.pool)
    embedder.nodetype_nodefeats  = nodetype_nodefeats
    embedder.eventname_edgefeats = eventname_edgefeats
    embedder.fit_vectorizer(train_ds)

    X_train, y_train, _           = embedder.embed_dataset(train_ds)
    X_test,  y_test,  test_names  = embedder.embed_dataset(test_ds)
    print(f"  Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"  Features: {len(embedder.feature_names)}")

    print("[Step 3] Training LightGBM (this happens only once)...")
    t0    = time.time()
    model = build_lightgbm()
    model.fit(X_train, y_train)
    train_time = time.time() - t0
    print(f"  Done in {train_time:.1f}s")

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    fpr_r, tpr_r, _ = roc_curve(y_test, probs)

    meta = {
        "trained_on":  int(len(X_train)),
        "test_samples": int(len(X_test)),
        "test_f1":     float(f1_score(y_test, preds)),
        "test_acc":    float(accuracy_score(y_test, preds)),
        "test_auc":    float(auc(fpr_r, tpr_r)),
        "test_fpr":    float(confusion_matrix(y_test, preds)[0, 1] /
                             confusion_matrix(y_test, preds)[0].sum()),
        "train_time_s": float(train_time),
        "N":            args.N,
        "pool":         args.pool,
    }
    save_model(model, embedder, meta)
    return model, embedder, X_train, X_test, y_test, test_names, probs, preds, meta


# ─────────────────────────────────────────────────────────────────────────────
# Analyse a single pickle sample
# ─────────────────────────────────────────────────────────────────────────────

def analyse_single(model, embedder, sample_path: str, output_dir: str):
    """Load one .pickle provenance graph, classify it, explain with SHAP."""
    with open(sample_path, "rb") as f:
        data = pickle.load(f)

    emb   = embedder.embed(data).numpy()
    proba = model.predict_proba(emb.reshape(1, -1))[0][1]
    pred  = int(proba >= 0.5)

    print(f"\n  Sample : {os.path.basename(sample_path)}")
    print(f"  Class  : {'MALWARE' if pred else 'BENIGN'}  (confidence {proba:.4f})")

    # SHAP for this one sample
    explainer = shap.TreeExplainer(model)
    sv        = explainer.shap_values(emb.reshape(1, -1))
    if isinstance(sv, list) and len(sv) > 1:
        sv = sv[1].flatten()
    elif isinstance(sv, list):
        sv = sv[0].flatten()
    else:
        sv = sv.flatten()

    ev = explainer.expected_value
    if hasattr(ev, '__len__') and len(ev) > 1:
        base_val = float(ev[1])
    elif hasattr(ev, '__len__'):
        base_val = float(ev[0])
    else:
        base_val = float(ev)

    feat_names = embedder.feature_names
    top_idx    = np.argsort(np.abs(sv))[::-1][:10]

    print(f"\n  Top-10 SHAP features:")
    print(f"  {'Rank':<5} {'Feature':<50} {'SHAP':>10}  Direction")
    print(f"  {'─'*80}")
    for rank, i in enumerate(top_idx, 1):
        direction = "→ MALWARE" if sv[i] > 0 else "→ BENIGN"
        print(f"  {rank:<5} {feat_names[i]:<50} {sv[i]:>10.5f}  {direction}")

    plot_sample_shap_waterfall(sv, feat_names, sample_path,
                                base_val, proba, output_dir,
                                sample_idx=0, top_k=12)
    return pred, proba

def _get_expected_value(explainer):
    """
    Safely extract the malware-class expected value from a SHAP TreeExplainer.
    LightGBM binary classification returns a scalar; multi-output returns a list.
    """
    ev = explainer.expected_value
    if hasattr(ev, '__len__') and len(ev) > 1:
        return float(ev[1])   # index 1 = malware class
    # Scalar or single-element array — binary LightGBM returns log-odds directly
    return float(ev) if not hasattr(ev, '__len__') else float(ev[0])
# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="LightGBM + SHAP Malware Analyser")
    p.add_argument("--dataset-path", type=str,
                   default=str(pathlib.Path(__file__).parent.parent / "dataset"))
    p.add_argument("--eventname-edgefeats-path", type=str,
                   default=str(pathlib.Path(__file__).parent.parent /
                               "dataset/EventName_EdgeFeatures.json"))
    p.add_argument("--nodetype-nodefeats-path", type=str,
                   default=str(pathlib.Path(__file__).parent.parent /
                               "dataset/NodeType_NodeFeatures.json"))
    p.add_argument("--output-dir", type=str,
                   default=str(pathlib.Path(__file__).parent / "results_lightgbm_shap"))
    p.add_argument("--retrain",     action="store_true",
                   help="Force retrain even if saved model exists")
    p.add_argument("--num-samples", type=int, default=30,
                   help="How many test samples to generate SHAP waterfalls for (default 30)")
    p.add_argument("--single-sample", type=str, default=None,
                   help="Path to a single .pickle provenance graph to classify + explain")
    p.add_argument("--N",    type=int, default=4)
    p.add_argument("--pool", type=str, default="sum", choices=["sum", "mean", "max"])
    p.add_argument("--shap-background", type=int, default=100,
                   help="Background samples for SHAP (default 100)")
    p.add_argument("--top-k-waterfall", type=int, default=5,
                   help="Max waterfall charts to generate (default 5; set 0 to skip)")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("  GRAPHITE — LightGBM + SHAP Malware Analyser")
    print("  Optimal combo from RQ1 (classifier) + RQ2 (XAI method)")
    print("=" * 70)

    nodetype_nodefeats  = json.load(open(args.nodetype_nodefeats_path))
    eventname_edgefeats = json.load(open(args.eventname_edgefeats_path))

    # ── Load or train ─────────────────────────────────────────────────────────
    if model_exists() and not args.retrain:
        print("\n[INFO] Saved model found — loading instantly (no training needed).")
        print(f"       To retrain from scratch, add --retrain flag.")
        model, embedder, meta = load_model()

        # We still need embeddings for analysis — re-embed the test set
        print("\n[Step 1] Loading test dataset for analysis...")
        test_ds = load_dataset(
            os.path.join(args.dataset_path, "test/benign"),
            os.path.join(args.dataset_path, "test/malware"),
            len(nodetype_nodefeats), len(eventname_edgefeats) + 1
        )
        train_ds = load_dataset(
            os.path.join(args.dataset_path, "train/benign"),
            os.path.join(args.dataset_path, "train/malware"),
            len(nodetype_nodefeats), len(eventname_edgefeats) + 1
        )
        print("[Step 2] Extracting embeddings...")
        X_train, _, _              = embedder.embed_dataset(train_ds)
        X_test,  y_test, test_names = embedder.embed_dataset(test_ds)
        probs = model.predict_proba(X_test)[:, 1]
        preds = (probs >= 0.5).astype(int)

    else:
        if args.retrain:
            print("\n[INFO] --retrain flag set — retraining model.")
        else:
            print("\n[INFO] No saved model found — training for the first time.")
        model, embedder, X_train, X_test, y_test, test_names, probs, preds, meta = \
            train(args, nodetype_nodefeats, eventname_edgefeats)

    # ── Single sample mode ────────────────────────────────────────────────────
    if args.single_sample:
        print(f"\n[Single-Sample Mode] Analysing: {args.single_sample}")
        analyse_single(model, embedder, args.single_sample, args.output_dir)
        print(f"\n  Waterfall chart saved to {args.output_dir}/")
        return

    # ── Full test-set analysis ────────────────────────────────────────────────
    feat_names = embedder.feature_names

    print(f"\n[Step 3] Computing SHAP values for all {len(X_test)} test samples...")
    t0        = time.time()
    shap_vals = get_shap_values(model, X_test)
    print(f"  Done in {time.time()-t0:.1f}s")

    base_value = float(shap.TreeExplainer(model).expected_value
                       if not isinstance(shap.TreeExplainer(model).expected_value,
                                         (list, np.ndarray))
                       else _get_expected_value(shap.TreeExplainer(model)))

    # ── Performance summary ───────────────────────────────────────────────────
    acc  = accuracy_score(y_test, preds)
    f1   = f1_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec  = recall_score(y_test, preds)
    cm   = confusion_matrix(y_test, preds)
    fpr_ = cm[0, 1] / cm[0].sum()

    print(f"\n{'='*70}")
    print(f"  PERFORMANCE (LightGBM on test set)")
    print(f"{'='*70}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1-Score : {f1:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  FPR      : {fpr_:.4f}")
    print(f"  AUC      : {meta.get('test_auc', '?'):.4f}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print(f"\n[Step 4] Generating plots...")
    plot_global_shap_beeswarm(shap_vals, X_test, feat_names, args.output_dir)
    plot_shap_bar_summary(shap_vals, feat_names, args.output_dir)
    plot_confidence_distribution(probs, y_test, args.output_dir)
    plot_roc(y_test, probs, args.output_dir)
    plot_confusion(y_test, preds, args.output_dir)
    plot_shap_dependence(shap_vals, X_test, feat_names, args.output_dir)
    plot_shap_heatmap(shap_vals, feat_names, test_names, args.output_dir)
    plot_per_sample_confidence(probs, y_test, test_names, args.output_dir)

    # ── Waterfall charts for top-N malware samples ────────────────────────────
    if args.top_k_waterfall > 0:
        print(f"\n[Step 5] Generating SHAP waterfall charts for top {args.top_k_waterfall} malware samples...")
        malware_idx  = np.where(y_test == 1)[0]
        # Sort by model confidence descending — most certain malware first
        sorted_idx   = malware_idx[np.argsort(probs[malware_idx])[::-1]]
        for si, idx in enumerate(sorted_idx[:args.top_k_waterfall]):
            plot_sample_shap_waterfall(
                shap_vals[idx], feat_names, test_names[idx],
                base_value, probs[idx], args.output_dir,
                sample_idx=si
            )

    # ── Save analysis JSON ────────────────────────────────────────────────────
    top20_idx   = np.argsort(np.abs(shap_vals).mean(axis=0))[::-1][:20]
    analysis    = {
        "model":    "LightGBM",
        "xai":      "SHAP (TreeExplainer)",
        "performance": {
            "accuracy": acc, "f1": f1, "precision": prec,
            "recall": rec, "fpr": fpr_,
            "auc": float(meta.get("test_auc", 0))
        },
        "top_20_features": [
            {"rank": i+1, "feature": feat_names[fi],
             "mean_abs_shap": float(np.abs(shap_vals).mean(axis=0)[fi])}
            for i, fi in enumerate(top20_idx)
        ],
        "per_sample": [
            {"name": test_names[i], "true": int(y_test[i]),
             "pred": int(preds[i]), "confidence": float(probs[i])}
            for i in range(len(X_test))
        ]
    }
    out_json = os.path.join(args.output_dir, "lightgbm_shap_analysis.json")
    with open(out_json, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"  [SAVED] Analysis JSON → {out_json}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  All outputs saved to: {args.output_dir}/")
    print(f"  Plots:")
    print(f"    shap_global_beeswarm.png    ← which n-grams matter globally")
    print(f"    shap_bar_summary.png        ← clean ranked bar chart")
    print(f"    shap_dependence_plots.png   ← feature count vs SHAP value")
    print(f"    shap_heatmap.png            ← features × samples heatmap")
    print(f"    shap_waterfall_sample_*.png ← per-sample explanation")
    print(f"    confidence_distribution.png ← how confident the model is")
    print(f"    roc_curve.png               ← ROC / AUC")
    print(f"    confusion_matrix.png        ← TP/FP/TN/FN breakdown")
    print(f"    per_sample_confidence.png   ← every test sample ranked")
    print(f"  Data:")
    print(f"    lightgbm_shap_analysis.json")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
