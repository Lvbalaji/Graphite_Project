"""
=============================================================================
RQ2: XAI Faithfulness Evaluation — SHAP vs LIME Feature Masking Analysis
=============================================================================
Applies SHAP and LIME to ALL four classifiers (RF, XGBoost, LightGBM, MLP),
performs top-K feature masking, and measures BOTH the absolute and relative
confidence drop to assess which XAI method more faithfully captures each
model's decisions.

This version reports:
  * Absolute drop : (orig_conf - masked_conf)          — robust to tiny baselines
  * Relative drop : (orig_conf - masked_conf)/orig_conf — original % metric

The absolute drop is the primary metric when evaluating misclassified samples
(where original confidence can be near 0 or 1 and the % metric explodes).

Usage:
  python xai_faithfulness_analysis.py --dataset-path ../dataset
  python xai_faithfulness_analysis.py --dataset-path ../dataset --num-samples 50
  python xai_faithfulness_analysis.py --dataset-path ../dataset --k-values 3 5 10
  python xai_faithfulness_analysis.py --dataset-path ../dataset --sample-mode wrong --wrong-consensus all
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
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import shap
from lime.lime_tabular import LimeTabularExplainer

# ── Import Graphite pipeline ──────────────────────────────────────────────
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from dataprocessor_graphs import load_dataset

import torch
from torch_geometric.data import Data
from sklearn.feature_extraction.text import CountVectorizer

# ── Constants ─────────────────────────────────────────────────────────────
CLASSIFIERS = ["rf", "xgboost", "lightgbm", "mlp"]
LABELS = {
    "rf": "Random Forest",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "mlp": "MLP"
}
COLORS = {"rf": "#2196F3", "xgboost": "#FF9800", "lightgbm": "#4CAF50", "mlp": "#9C27B0"}
XAI_COLORS = {"SHAP": "#E53935", "LIME": "#1E88E5"}


# ============================================================================
# Graphite N-gram embedding (standalone, no class dependency issues)
# ============================================================================

class GraphiteEmbedder:
    """Lightweight re-implementation of Graphite N-gram for embedding extraction."""

    def __init__(self, N=4, pool="sum"):
        self.N = N
        pool_choices = {"sum": torch.sum, "mean": torch.mean, "max": torch.max}
        self.pool = pool_choices[pool]
        self.count_vectorizer = CountVectorizer(ngram_range=(N, N))
        self.nodetype_nodefeats = None
        self.eventname_edgefeats = None

    def _get_thread_sorted_event_sequence(self, data, thread_node_idx):
        edge_src = data.edge_index[0]
        edge_tar = data.edge_index[1]
        outgoing = torch.nonzero(edge_src == thread_node_idx).flatten()
        incoming = torch.nonzero(edge_tar == thread_node_idx).flatten()
        if outgoing.numel() == 0 and incoming.numel() == 0:
            return []
        edge_feats = torch.cat([data.edge_attr[incoming], data.edge_attr[outgoing]], dim=0)
        sort_by_timestamp = torch.argsort(edge_feats[:, -1], descending=False)
        sorted_feats = edge_feats[sort_by_timestamp]
        eventname_indices = torch.nonzero(sorted_feats[:, :-1], as_tuple=False)[:, -1]
        return [self.eventname_edgefeats[i] for i in eventname_indices]

    def _get_thread_neighboring_nodetypes(self, data, thread_node_idx):
        outgoing = torch.nonzero(data.edge_index[0] == thread_node_idx).flatten()
        incoming = torch.nonzero(data.edge_index[1] == thread_node_idx).flatten()
        adj_nodes = []
        if outgoing.numel() > 0:
            adj_nodes.append(data.edge_index[1, outgoing])
        if incoming.numel() > 0:
            adj_nodes.append(data.edge_index[0, incoming])
        if not adj_nodes:
            return torch.zeros((1, len(self.nodetype_nodefeats)))
        neighbors = torch.unique(torch.cat(adj_nodes))
        return torch.sum(data.x[neighbors], dim=0).view(1, -1)

    def fit_vectorizer(self, train_dataset):
        thread_nodetype = torch.tensor(
            [1 if _type.lower() == "thread" else 0 for _type in self.nodetype_nodefeats]
        )
        all_sequences = []
        for data in train_dataset:
            thread_indices = torch.nonzero(
                torch.all(torch.eq(data.x, thread_nodetype), dim=1)
            ).flatten()
            for idx in thread_indices.tolist():
                seq = self._get_thread_sorted_event_sequence(data=data, thread_node_idx=idx)
                if seq:
                    all_sequences.append(seq)
        formatted = [' '.join(s) for s in all_sequences if len(s) >= self.N]
        self.count_vectorizer.fit(formatted)

    def generate_embedding(self, data):
        thread_nodetype = torch.tensor(
            [1 if _type.lower() == "thread" else 0 for _type in self.nodetype_nodefeats]
        )
        thread_indices = torch.nonzero(
            torch.all(torch.eq(data.x, thread_nodetype), dim=1)
        ).flatten()
        all_embs = []
        for idx in thread_indices.tolist():
            seq = self._get_thread_sorted_event_sequence(data=data, thread_node_idx=idx)
            ngram_vec = self.count_vectorizer.transform([" ".join(seq)]).toarray()
            node_emb = torch.cat([
                self._get_thread_neighboring_nodetypes(data, idx),
                torch.Tensor(ngram_vec).view(1, -1)
            ], dim=1)
            all_embs.append(node_emb)
        if not all_embs:
            total_dim = len(self.nodetype_nodefeats) + len(self.count_vectorizer.get_feature_names_out())
            return torch.zeros((1, total_dim))
        return self.pool(torch.cat(all_embs, dim=0), dim=0)

    def extract_all(self, dataset):
        X, y, names = [], [], []
        for data in dataset:
            X.append(self.generate_embedding(data).tolist())
            y.append(1 if "malware" in data.name else 0)
            names.append(data.name)
        return np.array(X), np.array(y), names


# ============================================================================
# Classifier builders
# ============================================================================

def build_classifier(name):
    classifiers = {
        "rf": RandomForestClassifier(
            n_estimators=500, criterion='gini', max_depth=20,
            min_samples_split=2, min_samples_leaf=1, max_features='sqrt',
            bootstrap=False, random_state=42
        ),
        "xgboost": XGBClassifier(
            n_estimators=500, learning_rate=0.05, max_depth=10,
            subsample=0.8, colsample_bytree=0.8, eval_metric='logloss',
            random_state=42
        ),
        "lightgbm": LGBMClassifier(
            n_estimators=500, max_depth=-1, num_leaves=63,
            learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
            min_child_samples=10, random_state=42, n_jobs=-1, verbose=-1
        ),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(256, 128, 64), activation='relu',
            solver='adam', alpha=0.001, learning_rate='adaptive',
            learning_rate_init=0.001, max_iter=500, early_stopping=True,
            validation_fraction=0.15, n_iter_no_change=20,
            random_state=42, verbose=False
        ),
    }
    return classifiers[name]


# ============================================================================
# XAI: SHAP explanations
# ============================================================================

def get_shap_explanation(model, clf_name, X_sample, X_background):
    """Return SHAP values for a single sample."""
    if clf_name in ["rf", "xgboost", "lightgbm"]:
        explainer = shap.TreeExplainer(model, data=X_background, model_output='probability')
        shap_values = explainer.shap_values(X_sample.reshape(1, -1))
        if isinstance(shap_values, list):
            return shap_values[1].flatten()  # class 1 = malware
        if shap_values.ndim == 3:
            return shap_values[0, :, 1]
        return shap_values.flatten()
    else:  # MLP
        explainer = shap.KernelExplainer(model.predict_proba, X_background)
        shap_values = explainer.shap_values(X_sample.reshape(1, -1), nsamples=200)
        if isinstance(shap_values, list):
            return shap_values[1].flatten()
        if shap_values.ndim == 3:
            return shap_values[0, :, 1]
        return shap_values.flatten()


def get_shap_top_k_features(shap_vals, k):
    """Return indices of top-K most influential features by absolute SHAP value."""
    return np.argsort(np.abs(shap_vals))[-k:][::-1]


# ============================================================================
# XAI: LIME explanations
# ============================================================================

def get_lime_explanation(model, X_sample, X_train, feature_names, num_features=20):
    """Return LIME explanation for a single sample."""
    explainer = LimeTabularExplainer(
        X_train, feature_names=feature_names, class_names=['benign', 'malware'],
        discretize_continuous=False, random_state=42
    )
    exp = explainer.explain_instance(
        X_sample, model.predict_proba, num_features=num_features, num_samples=500
    )
    return exp


def get_lime_top_k_features(lime_exp, feature_names, k):
    """Return indices of top-K features from LIME explanation."""
    feature_weights = lime_exp.as_list()
    indices = []
    for feat_name, weight in feature_weights:
        for i, fn in enumerate(feature_names):
            if fn in feat_name or feat_name in fn:
                indices.append((i, abs(weight)))
                break
    indices.sort(key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in indices[:k]]


# ============================================================================
# Feature masking evaluation
# ============================================================================

def compute_confidence_drop(model, X_sample, top_k_indices, original_confidence):
    """
    Mask top-K features (set to 0) and measure confidence drop.

    Returns:
        masked_confidence : float  — P(malware) after masking
        drop              : float  — absolute drop (orig - masked), signed
        drop_pct          : float  — relative drop as percentage of original
    """
    X_masked = X_sample.copy()
    X_masked[top_k_indices] = 0.0
    masked_confidence = model.predict_proba(X_masked.reshape(1, -1))[0][1]
    drop = original_confidence - masked_confidence
    drop_pct = (drop / original_confidence) * 100 if original_confidence > 0 else 0.0
    return masked_confidence, drop, drop_pct


# ============================================================================
# Visualization functions
# ============================================================================

def plot_confidence_drop_comparison(results, k_values, output_dir):
    """Bar chart: avg confidence drop (%) for SHAP vs LIME across classifiers and K values."""
    n_k = len(k_values)
    n_clf = len(CLASSIFIERS)

    fig, axes = plt.subplots(1, n_k, figsize=(7 * n_k, 6))
    if n_k == 1:
        axes = [axes]

    fig.suptitle('RQ2: Average Confidence Drop After Masking Top-K Features\n(Higher = More Faithful)',
                 fontsize=14, fontweight='bold')

    for ki, k in enumerate(k_values):
        ax = axes[ki]
        x = np.arange(n_clf)
        width = 0.35

        shap_drops = [results[clf][k]['shap']['avg_drop_pct'] for clf in CLASSIFIERS]
        lime_drops = [results[clf][k]['lime']['avg_drop_pct'] for clf in CLASSIFIERS]

        bars1 = ax.bar(x - width/2, shap_drops, width, label='SHAP',
                       color=XAI_COLORS['SHAP'], edgecolor='black', linewidth=0.5)
        bars2 = ax.bar(x + width/2, lime_drops, width, label='LIME',
                       color=XAI_COLORS['LIME'], edgecolor='black', linewidth=0.5)

        for bar in bars1:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                    f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
        for bar in bars2:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                    f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

        ax.set_xlabel('Classifier')
        ax.set_ylabel('Avg Confidence Drop (%)')
        ax.set_title(f'K = {k}')
        ax.set_xticks(x)
        ax.set_xticklabels([LABELS[c] for c in CLASSIFIERS], fontsize=9)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rq2_confidence_drop_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_absolute_drop_comparison(results, k_values, output_dir):
    """Bar chart: avg ABSOLUTE confidence drop for SHAP vs LIME — robust when % misleads."""
    n_k = len(k_values)
    n_clf = len(CLASSIFIERS)

    fig, axes = plt.subplots(1, n_k, figsize=(7 * n_k, 6))
    if n_k == 1:
        axes = [axes]

    fig.suptitle('RQ2: Average ABSOLUTE Confidence Drop After Masking Top-K Features\n'
                 '(Robust to near-zero baselines — Higher = More Faithful)',
                 fontsize=14, fontweight='bold')

    for ki, k in enumerate(k_values):
        ax = axes[ki]
        x = np.arange(n_clf)
        width = 0.35

        shap_drops = [results[clf][k]['shap']['avg_abs_drop'] for clf in CLASSIFIERS]
        lime_drops = [results[clf][k]['lime']['avg_abs_drop'] for clf in CLASSIFIERS]

        bars1 = ax.bar(x - width/2, shap_drops, width, label='SHAP',
                       color=XAI_COLORS['SHAP'], edgecolor='black', linewidth=0.5)
        bars2 = ax.bar(x + width/2, lime_drops, width, label='LIME',
                       color=XAI_COLORS['LIME'], edgecolor='black', linewidth=0.5)

        for bar in bars1:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{bar.get_height():+.4f}', ha='center',
                    va='bottom' if bar.get_height() >= 0 else 'top',
                    fontsize=8, fontweight='bold')
        for bar in bars2:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{bar.get_height():+.4f}', ha='center',
                    va='bottom' if bar.get_height() >= 0 else 'top',
                    fontsize=8, fontweight='bold')

        ax.axhline(0, color='black', linewidth=0.8)
        ax.set_xlabel('Classifier')
        ax.set_ylabel('Avg Absolute Confidence Drop (prob. units)')
        ax.set_title(f'K = {k}')
        ax.set_xticks(x)
        ax.set_xticklabels([LABELS[c] for c in CLASSIFIERS], fontsize=9)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rq2_absolute_drop_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_cumulative_masking(results, k_values, output_dir):
    """Line plot showing cumulative confidence drop as K increases, per classifier."""
    if len(k_values) < 2:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('RQ2: Cumulative Confidence Drop as More Features are Masked',
                 fontsize=14, fontweight='bold')

    for idx, clf in enumerate(CLASSIFIERS):
        ax = axes[idx // 2][idx % 2]
        shap_drops = [results[clf][k]['shap']['avg_drop_pct'] for k in k_values]
        lime_drops = [results[clf][k]['lime']['avg_drop_pct'] for k in k_values]

        ax.plot(k_values, shap_drops, 'o-', color=XAI_COLORS['SHAP'], linewidth=2,
                markersize=8, label='SHAP')
        ax.plot(k_values, lime_drops, 's--', color=XAI_COLORS['LIME'], linewidth=2,
                markersize=8, label='LIME')

        ax.set_xlabel('K (Number of Features Masked)')
        ax.set_ylabel('Avg Confidence Drop (%)')
        ax.set_title(f'{LABELS[clf]}', fontweight='bold', color=COLORS[clf])
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rq2_cumulative_masking.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_per_sample_drops(sample_results, k, clf, output_dir):
    """Scatter plot of per-sample confidence drops for SHAP vs LIME."""
    fig, ax = plt.subplots(figsize=(8, 6))

    shap_drops = [s['shap_drop_pct'] for s in sample_results]
    lime_drops = [s['lime_drop_pct'] for s in sample_results]

    ax.scatter(shap_drops, lime_drops, alpha=0.6, c=COLORS[clf], edgecolors='black', linewidth=0.5, s=60)
    max_val = max(max(shap_drops + [1]), max(lime_drops + [1])) * 1.1
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.4, label='Equal faithfulness')

    ax.set_xlabel('SHAP Confidence Drop (%)')
    ax.set_ylabel('LIME Confidence Drop (%)')
    ax.set_title(f'{LABELS[clf]}: Per-Sample Confidence Drop (K={k})\nPoints above line → LIME more faithful | Below → SHAP more faithful')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'rq2_scatter_{clf}_k{k}.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_xai_timing(timing_results, output_dir):
    """Bar chart of SHAP vs LIME explanation times per classifier."""
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(CLASSIFIERS))
    width = 0.35

    shap_times = [timing_results[clf]['shap_avg_time'] for clf in CLASSIFIERS]
    lime_times = [timing_results[clf]['lime_avg_time'] for clf in CLASSIFIERS]

    bars1 = ax.bar(x - width/2, shap_times, width, label='SHAP',
                   color=XAI_COLORS['SHAP'], edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, lime_times, width, label='LIME',
                   color=XAI_COLORS['LIME'], edgecolor='black', linewidth=0.5)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{bar.get_height():.2f}s', ha='center', va='bottom', fontsize=9, fontweight='bold')
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{bar.get_height():.2f}s', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xlabel('Classifier')
    ax.set_ylabel('Avg Explanation Time (seconds)')
    ax.set_title('RQ2: SHAP vs LIME — Average Explanation Time per Sample')
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[c] for c in CLASSIFIERS])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rq2_explanation_timing.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_summary_heatmap(results, k_values, output_dir):
    """Heatmap summarising SHAP vs LIME faithfulness across all classifiers and K values."""
    data_matrix = []
    row_labels = []
    for k in k_values:
        row = []
        for clf in CLASSIFIERS:
            diff = results[clf][k]['shap']['avg_drop_pct'] - results[clf][k]['lime']['avg_drop_pct']
            row.append(diff)
        data_matrix.append(row)
        row_labels.append(f'K={k}')

    fig, ax = plt.subplots(figsize=(8, max(4, len(k_values) * 1.5)))
    im = ax.imshow(data_matrix, cmap='RdBu_r', aspect='auto')

    ax.set_xticks(np.arange(len(CLASSIFIERS)))
    ax.set_yticks(np.arange(len(k_values)))
    ax.set_xticklabels([LABELS[c] for c in CLASSIFIERS])
    ax.set_yticklabels(row_labels)

    for i in range(len(k_values)):
        for j in range(len(CLASSIFIERS)):
            val = data_matrix[i][j]
            color = 'white' if abs(val) > 5 else 'black'
            ax.text(j, i, f'{val:+.1f}%', ha='center', va='center', color=color, fontweight='bold')

    ax.set_title('SHAP − LIME Confidence Drop Difference (%)\nRed = SHAP more faithful | Blue = LIME more faithful',
                 fontweight='bold')
    fig.colorbar(im, ax=ax, label='Difference (pp)')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rq2_faithfulness_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_global_shap_beeswarm(global_shap_matrix, global_X_matrix, feature_names, output_dir,
                               top_n=20, clf_name="lightgbm"):
    """Global SHAP beeswarm plot."""
    shap_matrix = np.array(global_shap_matrix[clf_name])
    X_matrix    = np.array(global_X_matrix[clf_name])

    if shap_matrix.shape[0] == 0:
        return

    mean_abs_shap = np.mean(np.abs(shap_matrix), axis=0)
    top_idx = np.argsort(mean_abs_shap)[::-1][:top_n]
    top_idx = top_idx[::-1]

    top_feature_names = [feature_names[i] for i in top_idx]
    top_shap           = shap_matrix[:, top_idx]
    top_X              = X_matrix[:, top_idx]

    vmax = np.percentile(top_X, 95) if top_X.max() > 0 else 1.0
    top_X_norm = np.clip(top_X / (vmax + 1e-9), 0, 1)

    fig_height = max(8, top_n * 0.45 + 2)
    fig, ax = plt.subplots(figsize=(13, fig_height))

    cmap = plt.cm.RdBu_r

    n_samples = top_shap.shape[0]
    jitter_scale = 0.25

    for fi, feat_pos in enumerate(range(top_n)):
        shap_vals_feat = top_shap[:, fi]
        x_norm_feat    = top_X_norm[:, fi]

        rng = np.random.RandomState(42 + fi)
        y_jitter = feat_pos + rng.uniform(-jitter_scale, jitter_scale, n_samples)

        ax.scatter(
            shap_vals_feat, y_jitter,
            c=x_norm_feat, cmap='RdBu_r', vmin=0, vmax=1,
            s=55, alpha=0.80, linewidths=0.4, edgecolors='grey',
            zorder=3
        )

    ax.axvline(0, color='black', linewidth=1.0, linestyle='--', alpha=0.6, zorder=2)

    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_feature_names, fontsize=9)
    ax.set_xlabel('SHAP Value  (← pushes toward Benign   |   pushes toward Malware →)',
                  fontsize=11)
    ax.set_title(
        f'Global SHAP Feature Impact — {LABELS.get(clf_name, clf_name)}\n'
        f'Top-{top_n} N-gram Features Ranked by Mean |SHAP| Across {n_samples} Samples',
        fontsize=12, fontweight='bold', pad=14
    )
    ax.grid(axis='x', alpha=0.25, zorder=1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    sm = plt.cm.ScalarMappable(cmap='RdBu_r', norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.03, shrink=0.6)
    cbar.set_label('Feature value (normalised n-gram count)', fontsize=9)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(['Low / Absent', 'Medium', 'High Count'])

    plt.tight_layout()
    out_path = os.path.join(output_dir, f'rq2_global_shap_beeswarm_{clf_name}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {os.path.basename(out_path)}")


# ============================================================================
# Sample selection
# ============================================================================

def select_samples(X_test, y_test, test_names, models, mode, n_requested,
                   wrong_consensus='any', seed=42):
    """Pick the test-set indices to run XAI on, based on --sample-mode."""
    rng = np.random.RandomState(seed)
    clf_names = list(models.keys())

    preds_per_clf = {c: models[c].predict(X_test) for c in clf_names}

    malware_idx = np.where(y_test == 1)[0]
    benign_idx  = np.where(y_test == 0)[0]

    correct_malware_all, correct_benign_all = [], []
    for i in malware_idx:
        if all(preds_per_clf[c][i] == 1 for c in clf_names):
            correct_malware_all.append(i)
    for i in benign_idx:
        if all(preds_per_clf[c][i] == 0 for c in clf_names):
            correct_benign_all.append(i)

    wrong_fn, wrong_fp = [], []
    for i in malware_idx:
        misclf = [c for c in clf_names if preds_per_clf[c][i] == 0]
        if (wrong_consensus == 'any' and len(misclf) >= 1) or \
           (wrong_consensus == 'all' and len(misclf) == len(clf_names)):
            wrong_fn.append((i, misclf))
    for i in benign_idx:
        misclf = [c for c in clf_names if preds_per_clf[c][i] == 1]
        if (wrong_consensus == 'any' and len(misclf) >= 1) or \
           (wrong_consensus == 'all' and len(misclf) == len(clf_names)):
            wrong_fp.append((i, misclf))

    print(f"  Pool sizes — correct malware (all models): {len(correct_malware_all)} | "
          f"correct benign (all models): {len(correct_benign_all)}")
    print(f"  Pool sizes — wrong FN ({wrong_consensus}): {len(wrong_fn)} | "
          f"wrong FP ({wrong_consensus}): {len(wrong_fp)}")

    def _pick(pool, n):
        if len(pool) == 0:
            return []
        n = min(n, len(pool))
        chosen = rng.choice(len(pool), size=n, replace=False)
        return [pool[j] for j in chosen]

    selected = []

    if mode == 'correct-malware':
        for i in _pick(correct_malware_all, n_requested):
            selected.append((i, 'correct_malware', []))

    elif mode == 'correct-benign':
        for i in _pick(correct_benign_all, n_requested):
            selected.append((i, 'correct_benign', []))

    elif mode == 'correct-both':
        half = n_requested // 2
        other = n_requested - half
        for i in _pick(correct_malware_all, half):
            selected.append((i, 'correct_malware', []))
        for i in _pick(correct_benign_all, other):
            selected.append((i, 'correct_benign', []))

    elif mode == 'wrong':
        half = n_requested // 2
        other = n_requested - half
        fn_taken = _pick(wrong_fn, half)
        fp_taken = _pick(wrong_fp, other)
        deficit = n_requested - (len(fn_taken) + len(fp_taken))
        if deficit > 0:
            remaining_fn = [p for p in wrong_fn if p not in fn_taken]
            remaining_fp = [p for p in wrong_fp if p not in fp_taken]
            extras = _pick(remaining_fn + remaining_fp, deficit)
            fn_ids = {p[0] for p in wrong_fn}
            for p in extras:
                if p[0] in fn_ids:
                    fn_taken.append(p)
                else:
                    fp_taken.append(p)
        for idx, misclf in fn_taken:
            selected.append((idx, 'wrong_fn', misclf))
        for idx, misclf in fp_taken:
            selected.append((idx, 'wrong_fp', misclf))

    elif mode == 'mixed':
        half_correct = n_requested // 2
        half_wrong   = n_requested - half_correct
        cm = half_correct // 2
        cb = half_correct - cm
        wn = half_wrong // 2
        wp = half_wrong - wn
        for i in _pick(correct_malware_all, cm):
            selected.append((i, 'correct_malware', []))
        for i in _pick(correct_benign_all, cb):
            selected.append((i, 'correct_benign', []))
        for idx, misclf in _pick(wrong_fn, wn):
            selected.append((idx, 'wrong_fn', misclf))
        for idx, misclf in _pick(wrong_fp, wp):
            selected.append((idx, 'wrong_fp', misclf))

    seen = set()
    unique = []
    for tup in selected:
        if tup[0] not in seen:
            unique.append(tup)
            seen.add(tup[0])

    sample_indices = np.array([t[0] for t in unique], dtype=int)
    sample_meta = [
        {'idx': int(t[0]),
         'true_label': int(y_test[t[0]]),
         'category': t[1],
         'misclassified_by': t[2]}
        for t in unique
    ]

    from collections import Counter
    cat_counts = Counter(m['category'] for m in sample_meta)
    print(f"  Selected {len(sample_meta)} samples for mode='{mode}':")
    for cat, cnt in cat_counts.items():
        print(f"    {cat:<18}: {cnt}")
    if len(sample_meta) < n_requested:
        print(f"  NOTE: Requested {n_requested} but only {len(sample_meta)} available in this mode's pool.")

    return sample_indices, sample_meta


# ============================================================================
# Main
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='RQ2: XAI Faithfulness Analysis')
    parser.add_argument('--dataset-path', type=str,
                        default=str(pathlib.Path(__file__).parent.parent.joinpath("dataset")),
                        help='Path to dataset directory')
    parser.add_argument('--eventname-edgefeats-path', type=str,
                        default=str(pathlib.Path(__file__).parent.parent.joinpath("dataset/EventName_EdgeFeatures.json")))
    parser.add_argument('--nodetype-nodefeats-path', type=str,
                        default=str(pathlib.Path(__file__).parent.parent.joinpath("dataset/NodeType_NodeFeatures.json")))
    parser.add_argument('--output-dir', type=str,
                        default=str(pathlib.Path(__file__).parent.joinpath("results_rq2")),
                        help='Output directory')
    parser.add_argument('--num-samples', type=int, default=30,
                        help='Number of test samples to evaluate (default: 30)')
    parser.add_argument('--sample-mode', type=str, default='correct-malware',
                        choices=['correct-malware', 'correct-benign', 'correct-both',
                                 'wrong', 'mixed'],
                        help=("Which test samples to run XAI on:\n"
                              "  correct-malware : malware correctly classified by ALL models (default)\n"
                              "  correct-benign  : benign correctly classified by ALL models\n"
                              "  correct-both    : ~50/50 split of correct malware + correct benign\n"
                              "  wrong           : misclassified samples (FN + FP) — for error analysis\n"
                              "  mixed           : ~50/50 split of correctly classified + misclassified"))
    parser.add_argument('--wrong-consensus', type=str, default='any',
                        choices=['any', 'all'],
                        help=("For --sample-mode wrong/mixed: how many models must misclassify a sample. "
                              "'any' = at least 1 model, 'all' = every model. Default: any"))
    parser.add_argument('--k-values', nargs='+', type=int, default=[5],
                        help="Top-K feature(s) to mask. Examples: --k-values 5  |  --k-values 3 5 10")
    parser.add_argument('--shap-background', type=int, default=50,
                        help='Number of background samples for SHAP (default: 50)')
    parser.add_argument('--N', type=int, default=4, help='N-gram size')
    parser.add_argument('--pool', type=str, default='sum', choices=['sum', 'mean', 'max'])
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    k_values = sorted(args.k_values)

    print("=" * 70)
    print("  RQ2: XAI FAITHFULNESS EVALUATION — SHAP vs LIME")
    print("  Feature Masking Analysis Across All Classifiers")
    print(f"  K values: {k_values}")
    print(f"  Samples per classifier: {args.num_samples}")
    print(f"  Sample mode: {args.sample_mode}"
          + (f" (wrong-consensus={args.wrong_consensus})"
             if args.sample_mode in ('wrong', 'mixed') else ""))
    print("=" * 70)

    # ── Load data ─────────────────────────────────────────────────────────
    nodetype_nodefeats = json.load(open(args.nodetype_nodefeats_path, "r"))
    eventname_edgefeats = json.load(open(args.eventname_edgefeats_path, "r"))

    print("\n[Step 1] Loading datasets...")
    train_dataset = load_dataset(
        os.path.join(args.dataset_path, "train/benign"),
        os.path.join(args.dataset_path, "train/malware"),
        len(nodetype_nodefeats), len(eventname_edgefeats) + 1
    )
    test_dataset = load_dataset(
        os.path.join(args.dataset_path, "test/benign"),
        os.path.join(args.dataset_path, "test/malware"),
        len(nodetype_nodefeats), len(eventname_edgefeats) + 1
    )

    # ── Extract embeddings ────────────────────────────────────────────────
    print("\n[Step 2] Extracting N-gram embeddings...")
    embedder = GraphiteEmbedder(N=args.N, pool=args.pool)
    embedder.nodetype_nodefeats = nodetype_nodefeats
    embedder.eventname_edgefeats = eventname_edgefeats
    embedder.fit_vectorizer(train_dataset)

    X_train, y_train, train_names = embedder.extract_all(train_dataset)
    X_test, y_test, test_names = embedder.extract_all(test_dataset)
    print(f"  Train: {X_train.shape} | Test: {X_test.shape}")

    ngram_names = list(embedder.count_vectorizer.get_feature_names_out())
    feature_names = list(nodetype_nodefeats) + ngram_names
    print(f"  Total features: {len(feature_names)} ({len(nodetype_nodefeats)} node-type + {len(ngram_names)} n-gram)")

    # ── SHAP background set ───────────────────────────────────────────────
    bg_size = min(args.shap_background, len(X_train))
    np.random.seed(42)
    bg_indices = np.random.choice(len(X_train), bg_size, replace=False)
    X_background = X_train[bg_indices]
    print(f"  SHAP background samples: {bg_size}")

    # ── Train all classifiers ─────────────────────────────────────────────
    print("\n[Step 3] Training all classifiers...")
    models = {}
    for clf_name in CLASSIFIERS:
        print(f"  Training {LABELS[clf_name]}...", end=" ", flush=True)
        t0 = time.time()
        model = build_classifier(clf_name)
        model.fit(X_train, y_train)
        models[clf_name] = model
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        print(f"Done ({time.time()-t0:.1f}s) | Acc={acc:.4f} F1={f1:.4f}")

    # ── Select samples ─────────────────────────────────────────────────────
    print(f"\n[Step 4] Selecting test samples (mode='{args.sample_mode}')...")
    sample_indices, sample_meta = select_samples(
        X_test=X_test, y_test=y_test, test_names=test_names, models=models,
        mode=args.sample_mode, n_requested=args.num_samples,
        wrong_consensus=args.wrong_consensus, seed=42,
    )
    n_samples = len(sample_indices)
    if n_samples == 0:
        print("  ERROR: No samples matched the requested mode. Exiting.")
        return
    meta_by_idx = {m['idx']: m for m in sample_meta}

    # ── Run XAI evaluation ────────────────────────────────────────────────
    print(f"\n[Step 5] Running XAI faithfulness evaluation...")
    all_results = {}
    timing_results = {}
    detailed_sample_results = {}
    global_shap_matrix = {}
    global_X_matrix = {}

    for clf_name in CLASSIFIERS:
        model = models[clf_name]
        print(f"\n{'='*60}")
        print(f"  Evaluating: {LABELS[clf_name]}")
        print(f"{'='*60}")

        all_results[clf_name] = {}
        timing_results[clf_name] = {'shap_times': [], 'lime_times': []}
        detailed_sample_results[clf_name] = {}
        global_shap_matrix[clf_name] = []
        global_X_matrix[clf_name] = []

        for k in k_values:
            all_results[clf_name][k] = {
                'shap': {'drops': [], 'drop_pcts': [], 'abs_drops': []},
                'lime': {'drops': [], 'drop_pcts': [], 'abs_drops': []}
            }
            detailed_sample_results[clf_name][k] = []

        for si, idx in enumerate(sample_indices):
            X_sample = X_test[idx]
            original_proba = model.predict_proba(X_sample.reshape(1, -1))[0][1]

            # ── SHAP ──
            t0 = time.time()
            try:
                shap_vals = get_shap_explanation(model, clf_name, X_sample, X_background)
            except Exception as e:
                print(f"    [SHAP ERROR] Sample {si}: {e}")
                shap_vals = np.zeros(len(feature_names))
            shap_time = time.time() - t0
            timing_results[clf_name]['shap_times'].append(shap_time)
            global_shap_matrix[clf_name].append(shap_vals.copy())
            global_X_matrix[clf_name].append(X_sample.copy())

            # ── LIME ──
            t0 = time.time()
            try:
                lime_exp = get_lime_explanation(model, X_sample, X_train, feature_names,
                                                num_features=max(k_values) + 5)
            except Exception as e:
                print(f"    [LIME ERROR] Sample {si}: {e}")
                lime_exp = None
            lime_time = time.time() - t0
            timing_results[clf_name]['lime_times'].append(lime_time)

            for k in k_values:
                # SHAP top-K
                shap_top_k = get_shap_top_k_features(shap_vals, k)
                shap_masked_conf, shap_drop, shap_drop_pct = compute_confidence_drop(
                    model, X_sample, shap_top_k, original_proba
                )

                # LIME top-K
                if lime_exp is not None:
                    lime_top_k = get_lime_top_k_features(lime_exp, feature_names, k)
                    if len(lime_top_k) < k:
                        remaining = [i for i in range(len(feature_names)) if i not in lime_top_k]
                        lime_top_k.extend(remaining[:k - len(lime_top_k)])
                else:
                    lime_top_k = list(range(k))

                lime_masked_conf, lime_drop, lime_drop_pct = compute_confidence_drop(
                    model, X_sample, lime_top_k, original_proba
                )

                # ════════════════════════════════════════════════════════════
                # VERBOSE per-sample print-out
                # ════════════════════════════════════════════════════════════
                meta = meta_by_idx[int(idx)]
                true_lbl_str = "MALWARE" if meta['true_label'] == 1 else "BENIGN"
                pred_lbl = int(model.predict(X_sample.reshape(1, -1))[0])
                pred_lbl_str = "MALWARE" if pred_lbl == 1 else "BENIGN"
                correct_flag = "✓ correct" if pred_lbl == meta['true_label'] else "✗ MISCLASSIFIED"
                cat_pretty = {
                    'correct_malware': 'Correctly classified MALWARE',
                    'correct_benign':  'Correctly classified BENIGN',
                    'wrong_fn':        'Missed malware (FALSE NEGATIVE)',
                    'wrong_fp':        'Benign flagged as malware (FALSE POSITIVE)',
                }.get(meta['category'], meta['category'])

                print(f"\n  {'━'*110}")
                print(f"  SAMPLE [{si+1}/{n_samples}] | K={k} | Classifier: {LABELS[clf_name]}")
                print(f"  Name: {test_names[idx][:90]}")
                print(f"  True label: {true_lbl_str} | This model predicts: {pred_lbl_str} ({correct_flag})")
                print(f"  Sample category: {cat_pretty}")
                if meta['misclassified_by']:
                    print(f"  Misclassified by: {', '.join(LABELS[c] for c in meta['misclassified_by'])}")
                print(f"  Original malware-class probability: {original_proba:.6f}")
                print(f"    → P(malware) = {original_proba*100:.2f}% before any masking.")
                print(f"  {'━'*110}")

                # ── SHAP top-K details ──
                print(f"\n    ▶ SHAP Top-{k} Features (ranked by |SHAP value|):")
                print(f"    {'Rank':<6} {'Feature Name':<55} {'Idx':<6} {'Orig Val':<12} {'SHAP Val':<14} {'|SHAP|':<12}")
                print(f"    {'─'*105}")
                for rank, feat_idx in enumerate(shap_top_k, 1):
                    fname = feature_names[feat_idx] if feat_idx < len(feature_names) else f"feat_{feat_idx}"
                    orig_val = X_sample[feat_idx]
                    shap_val = shap_vals[feat_idx]
                    if feat_idx < len(nodetype_nodefeats):
                        feat_type = "node-type"
                        val_meaning = f"(sum of '{nodetype_nodefeats[feat_idx]}' neighbors across threads)"
                    else:
                        feat_type = "n-gram"
                        val_meaning = f"(this 4-event sequence occurred {int(orig_val)} time{'s' if orig_val != 1 else ''} in the graph)"
                    direction = "pushes TOWARD malware" if shap_val > 0 else "pushes AWAY from malware"
                    print(f"    {rank:<6} {fname:<55} {feat_idx:<6} {orig_val:<12.4f} {shap_val:<14.6f} {abs(shap_val):<12.6f}")
                    print(f"           [{feat_type}] {val_meaning}")
                    print(f"           SHAP interpretation: this feature {direction} by {abs(shap_val):.6f} probability units")

                print(f"\n    MASKING STEP — Setting the above {k} features to 0.0:")
                print(f"    ┌──────────────────────────────────────────────────────────────────────────────┐")
                print(f"    │  Before masking:  model confidence = {original_proba:.6f} ({original_proba*100:.2f}% malware)")
                print(f"    │  After masking:   model confidence = {shap_masked_conf:.6f} ({shap_masked_conf*100:.2f}% malware)")
                print(f"    │  Absolute drop:   {shap_drop:+.6f} probability units")
                print(f"    │  Relative drop:   {shap_drop_pct:+.2f}%  (of original confidence)")
                if abs(original_proba) < 0.01 or abs(original_proba) > 0.99:
                    print(f"    │  ⚠ NOTE: original_proba={original_proba:.6f} is near 0 or 1 — % unreliable, trust absolute")
                if shap_drop_pct > 10:
                    print(f"    │  → Large drop: SHAP correctly found features the model heavily relies on")
                elif shap_drop_pct > 0:
                    print(f"    │  → Moderate drop: SHAP found features with some influence on the decision")
                else:
                    print(f"    │  → No/negative drop: masking these features did not reduce confidence")
                print(f"    └──────────────────────────────────────────────────────────────────────────────┘")

                # ── LIME top-K details ──
                print(f"\n    ▶ LIME Top-{k} Features (ranked by |LIME weight|):")
                print(f"    {'Rank':<6} {'Feature Name':<55} {'Idx':<6} {'Orig Val':<12} {'LIME Wt':<14}")
                print(f"    {'─'*93}")

                lime_weight_map = {}
                if lime_exp is not None:
                    for feat_name_lime, weight in lime_exp.as_list():
                        for fi, fn in enumerate(feature_names):
                            if fn in feat_name_lime or feat_name_lime in fn:
                                lime_weight_map[fi] = weight
                                break

                for rank, feat_idx in enumerate(lime_top_k, 1):
                    fname = feature_names[feat_idx] if feat_idx < len(feature_names) else f"feat_{feat_idx}"
                    orig_val = X_sample[feat_idx]
                    lime_wt = lime_weight_map.get(feat_idx, 0.0)
                    if feat_idx < len(nodetype_nodefeats):
                        feat_type = "node-type"
                        val_meaning = f"(sum of '{nodetype_nodefeats[feat_idx]}' neighbors across threads)"
                    else:
                        feat_type = "n-gram"
                        val_meaning = f"(this 4-event sequence occurred {int(orig_val)} time{'s' if orig_val != 1 else ''} in the graph)"
                    direction = "pushes TOWARD malware locally" if lime_wt > 0 else "pushes AWAY from malware locally"
                    print(f"    {rank:<6} {fname:<55} {feat_idx:<6} {orig_val:<12.4f} {lime_wt:<14.6f}")
                    print(f"           [{feat_type}] {val_meaning}")
                    if orig_val == 0.0:
                        print(f"           ⚠ WARNING: Orig Val=0 — masking this feature has NO effect (0→0 = no change)")
                    else:
                        print(f"           LIME interpretation: this feature {direction} (weight={lime_wt:.6f})")

                print(f"\n    MASKING STEP — Setting the above {k} features to 0.0:")
                print(f"    ┌──────────────────────────────────────────────────────────────────────────────┐")
                print(f"    │  Before masking:  model confidence = {original_proba:.6f} ({original_proba*100:.2f}% malware)")
                print(f"    │  After masking:   model confidence = {lime_masked_conf:.6f} ({lime_masked_conf*100:.2f}% malware)")
                print(f"    │  Absolute drop:   {lime_drop:+.6f} probability units")
                print(f"    │  Relative drop:   {lime_drop_pct:+.2f}%  (of original confidence)")
                if abs(original_proba) < 0.01 or abs(original_proba) > 0.99:
                    print(f"    │  ⚠ NOTE: original_proba={original_proba:.6f} is near 0 or 1 — % unreliable, trust absolute")
                n_lime_zero = sum(1 for fi in lime_top_k if X_sample[fi] == 0.0)
                if n_lime_zero > 0:
                    print(f"    │  → {n_lime_zero}/{k} LIME-selected features had Orig Val=0 (masking had no effect on them)")
                if lime_drop_pct > 10:
                    print(f"    │  → Large drop: LIME found features the model relies on")
                elif lime_drop_pct > 0:
                    print(f"    │  → Small drop: LIME's selected features had limited real influence")
                else:
                    print(f"    │  → No/negative drop: LIME's features are NOT what the model actually uses")
                print(f"    └──────────────────────────────────────────────────────────────────────────────┘")

                # ── Side-by-side comparison (BOTH metrics) ──
                winner_pct = "SHAP" if shap_drop_pct > lime_drop_pct else "LIME"
                winner_abs = "SHAP" if shap_drop > lime_drop else "LIME"
                overlap = set(shap_top_k) & set(lime_top_k)
                n_shap_nonzero = sum(1 for fi in shap_top_k if X_sample[fi] != 0.0)
                n_lime_nonzero = sum(1 for fi in lime_top_k if X_sample[fi] != 0.0)
                print(f"\n    ✦ COMPARISON:")
                print(f"      By absolute drop : SHAP = {shap_drop:+.6f}  |  LIME = {lime_drop:+.6f}  →  Winner: {winner_abs}")
                print(f"      By relative drop : SHAP = {shap_drop_pct:+8.2f}%   |  LIME = {lime_drop_pct:+8.2f}%   →  Winner: {winner_pct}")
                if winner_abs != winner_pct:
                    print(f"      ⚠ Metrics disagree — likely because original_proba={original_proba:.6f} is near 0 or 1.")
                    print(f"        Trust absolute-drop winner ({winner_abs}) in this case.")
                winner = winner_pct  # preserved for downstream compatibility
                print(f"      SHAP selected {n_shap_nonzero}/{k} non-zero features  |  LIME selected {n_lime_nonzero}/{k} non-zero features")
                print(f"      Feature overlap (SHAP ∩ LIME): {len(overlap)}/{k} features in common", end="")
                if overlap:
                    overlap_names = [feature_names[i] for i in overlap]
                    print(f" → {overlap_names[:5]}")
                else:
                    print(f"  → (completely different features selected)")
                if n_lime_nonzero < n_shap_nonzero:
                    print(f"      → KEY INSIGHT: LIME chose {k - n_lime_nonzero} features with value=0 (masking them does nothing),")
                    print(f"        while SHAP chose features with actual behavioral signal.")

                # ════════════════════════════════════════════════════════════

                all_results[clf_name][k]['shap']['drops'].append(shap_drop)
                all_results[clf_name][k]['shap']['drop_pcts'].append(shap_drop_pct)
                all_results[clf_name][k]['shap']['abs_drops'].append(shap_drop)
                all_results[clf_name][k]['lime']['drops'].append(lime_drop)
                all_results[clf_name][k]['lime']['drop_pcts'].append(lime_drop_pct)
                all_results[clf_name][k]['lime']['abs_drops'].append(lime_drop)

                detailed_sample_results[clf_name][k].append({
                    'sample_idx': int(idx),
                    'sample_name': test_names[idx],
                    'true_label': int(meta['true_label']),
                    'sample_category': meta['category'],
                    'misclassified_by': meta['misclassified_by'],
                    'original_confidence': float(original_proba),
                    'shap_drop_pct': float(shap_drop_pct),
                    'lime_drop_pct': float(lime_drop_pct),
                    'shap_abs_drop': float(shap_drop),
                    'lime_abs_drop': float(lime_drop),
                    'shap_top_k_features': [feature_names[i] for i in shap_top_k[:min(5, k)]],
                    'lime_top_k_features': [feature_names[i] for i in lime_top_k[:min(5, k)]],
                    'shap_top_k_values': [float(X_sample[i]) for i in shap_top_k],
                    'lime_top_k_values': [float(X_sample[i]) for i in lime_top_k],
                    'shap_importance_scores': [float(shap_vals[i]) for i in shap_top_k],
                    'lime_weights': [float(lime_weight_map.get(i, 0.0)) for i in lime_top_k],
                    'shap_masked_confidence': float(shap_masked_conf),
                    'lime_masked_confidence': float(lime_masked_conf),
                    'feature_overlap_count': len(overlap),
                    'winner_by_pct': winner_pct,
                    'winner_by_abs': winner_abs,
                })

        # Compute averages
        timing_results[clf_name]['shap_avg_time'] = float(np.mean(timing_results[clf_name]['shap_times']))
        timing_results[clf_name]['lime_avg_time'] = float(np.mean(timing_results[clf_name]['lime_times']))

        for k in k_values:
            s = all_results[clf_name][k]['shap']
            l = all_results[clf_name][k]['lime']
            s['avg_drop_pct'] = float(np.mean(s['drop_pcts']))
            s['std_drop_pct'] = float(np.std(s['drop_pcts']))
            s['avg_abs_drop'] = float(np.mean(s['abs_drops']))
            s['std_abs_drop'] = float(np.std(s['abs_drops']))
            l['avg_drop_pct'] = float(np.mean(l['drop_pcts']))
            l['std_drop_pct'] = float(np.std(l['drop_pcts']))
            l['avg_abs_drop'] = float(np.mean(l['abs_drops']))
            l['std_abs_drop'] = float(np.std(l['abs_drops']))

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*130}")
    print(f"  RQ2 RESULTS SUMMARY: Average Confidence Drop After Masking Top-K Features")
    print(f"  (Reporting BOTH relative % and absolute probability-unit drops)")
    print(f"{'='*130}")

    for k in k_values:
        print(f"\n  K = {k}:")
        print(f"  {'─'*120}")
        print(f"  {'Classifier':<20} | {'SHAP Δ%':>14} | {'LIME Δ%':>14} | {'SHAP Δabs':>18} | {'LIME Δabs':>18} | {'Winner(abs)':>14}")
        print(f"  {'─'*120}")
        for clf in CLASSIFIERS:
            s = all_results[clf][k]['shap']
            l = all_results[clf][k]['lime']
            winner_pct = "SHAP" if s['avg_drop_pct'] > l['avg_drop_pct'] else "LIME"
            winner_abs = "SHAP" if s['avg_abs_drop'] > l['avg_abs_drop'] else "LIME"
            flag = " *" if winner_pct != winner_abs else ""
            print(f"  {LABELS[clf]:<20} | {s['avg_drop_pct']:>7.2f}±{s['std_drop_pct']:<5.2f} | "
                  f"{l['avg_drop_pct']:>7.2f}±{l['std_drop_pct']:<5.2f} | "
                  f"{s['avg_abs_drop']:>+9.5f}±{s['std_abs_drop']:<7.5f} | "
                  f"{l['avg_abs_drop']:>+9.5f}±{l['std_abs_drop']:<7.5f} | "
                  f"{winner_abs+flag:>14}")
        print(f"  (* = winner by % disagrees with winner by absolute drop — trust absolute when original confidence is near 0 or 1)")

    # Overall winners
    primary_k = k_values[0]
    shap_wins_pct = sum(1 for clf in CLASSIFIERS
                        if all_results[clf][primary_k]['shap']['avg_drop_pct'] >
                           all_results[clf][primary_k]['lime']['avg_drop_pct'])
    shap_wins_abs = sum(1 for clf in CLASSIFIERS
                        if all_results[clf][primary_k]['shap']['avg_abs_drop'] >
                           all_results[clf][primary_k]['lime']['avg_abs_drop'])
    lime_wins_pct = len(CLASSIFIERS) - shap_wins_pct
    lime_wins_abs = len(CLASSIFIERS) - shap_wins_abs

    print(f"\n  OVERALL (K={primary_k}):")
    print(f"    By relative %    : {'SHAP' if shap_wins_pct > lime_wins_pct else 'LIME'} more faithful "
          f"({shap_wins_pct} SHAP vs {lime_wins_pct} LIME across classifiers)")
    print(f"    By absolute drop : {'SHAP' if shap_wins_abs > lime_wins_abs else 'LIME'} more faithful "
          f"({shap_wins_abs} SHAP vs {lime_wins_abs} LIME across classifiers)")

    # Timing summary
    print(f"\n  EXPLANATION TIMING (avg per sample):")
    for clf in CLASSIFIERS:
        print(f"  {LABELS[clf]:<20} | SHAP: {timing_results[clf]['shap_avg_time']:.2f}s | "
              f"LIME: {timing_results[clf]['lime_avg_time']:.2f}s")

    # ── Global Feature Importance Summary ─────────────────────────────
    from collections import Counter
    print(f"\n{'='*90}")
    print(f"  GLOBAL FEATURE IMPORTANCE: Most Frequently Selected Top-K Features Across All Samples")
    print(f"{'='*90}")

    for clf in CLASSIFIERS:
        print(f"\n  ── {LABELS[clf]} (K={primary_k}) ──")

        shap_feat_counter = Counter()
        lime_feat_counter = Counter()
        shap_importance_accum = {}
        lime_importance_accum = {}

        for sample_detail in detailed_sample_results[clf][primary_k]:
            for fname in sample_detail['shap_top_k_features']:
                shap_feat_counter[fname] += 1
            for fname in sample_detail['lime_top_k_features']:
                lime_feat_counter[fname] += 1
            if 'shap_importance_scores' in sample_detail:
                for fname, score in zip(sample_detail['shap_top_k_features'],
                                         sample_detail['shap_importance_scores'][:len(sample_detail['shap_top_k_features'])]):
                    shap_importance_accum.setdefault(fname, []).append(abs(score))
            if 'lime_weights' in sample_detail:
                for fname, score in zip(sample_detail['lime_top_k_features'],
                                         sample_detail['lime_weights'][:len(sample_detail['lime_top_k_features'])]):
                    lime_importance_accum.setdefault(fname, []).append(abs(score))

        print(f"\n    SHAP — Top 10 most frequently selected features:")
        print(f"    {'Rank':<6} {'Feature':<55} {'Count':<8} {'Avg |SHAP|':<14}")
        print(f"    {'─'*83}")
        for rank, (fname, count) in enumerate(shap_feat_counter.most_common(10), 1):
            avg_imp = np.mean(shap_importance_accum.get(fname, [0.0]))
            print(f"    {rank:<6} {fname:<55} {count:<8} {avg_imp:<14.6f}")

        print(f"\n    LIME — Top 10 most frequently selected features:")
        print(f"    {'Rank':<6} {'Feature':<55} {'Count':<8} {'Avg |LIME wt|':<14}")
        print(f"    {'─'*83}")
        for rank, (fname, count) in enumerate(lime_feat_counter.most_common(10), 1):
            avg_imp = np.mean(lime_importance_accum.get(fname, [0.0]))
            print(f"    {rank:<6} {fname:<55} {count:<8} {avg_imp:<14.6f}")

        shap_top10 = set(f for f, _ in shap_feat_counter.most_common(10))
        lime_top10 = set(f for f, _ in lime_feat_counter.most_common(10))
        common = shap_top10 & lime_top10
        print(f"\n    Feature overlap (SHAP top-10 ∩ LIME top-10): {len(common)}/10")
        if common:
            print(f"    Common features: {list(common)[:5]}")
        else:
            print(f"    → SHAP and LIME select entirely different features for {LABELS[clf]}")

    print(f"\n{'='*90}")

    # ── Generate visualizations ───────────────────────────────────────────
    print(f"\n[Step 6] Generating visualizations...")
    plot_confidence_drop_comparison(all_results, k_values, output_dir)
    plot_absolute_drop_comparison(all_results, k_values, output_dir)
    plot_cumulative_masking(all_results, k_values, output_dir)
    plot_summary_heatmap(all_results, k_values, output_dir)
    plot_xai_timing(timing_results, output_dir)

    for clf in CLASSIFIERS:
        plot_per_sample_drops(detailed_sample_results[clf][primary_k], primary_k, clf, output_dir)

    print(f"\n  Generating global SHAP beeswarm plots...")
    for clf in CLASSIFIERS:
        plot_global_shap_beeswarm(
            global_shap_matrix, global_X_matrix, feature_names,
            output_dir, top_n=20, clf_name=clf
        )

    # ── Save JSON results ─────────────────────────────────────────────────
    save_data = {
        'config': {
            'k_values': k_values,
            'num_samples': n_samples,
            'sample_mode': args.sample_mode,
            'wrong_consensus': args.wrong_consensus,
            'shap_background_size': bg_size,
            'N': args.N,
            'pool': args.pool,
        },
        'summary': {},
        'timing': {},
        'sample_meta': sample_meta,
    }

    for clf in CLASSIFIERS:
        save_data['summary'][clf] = {}
        for k in k_values:
            s = all_results[clf][k]['shap']
            l = all_results[clf][k]['lime']
            save_data['summary'][clf][str(k)] = {
                'shap_avg_drop_pct': s['avg_drop_pct'],
                'shap_std_drop_pct': s['std_drop_pct'],
                'lime_avg_drop_pct': l['avg_drop_pct'],
                'lime_std_drop_pct': l['std_drop_pct'],
                'shap_avg_abs_drop': s['avg_abs_drop'],
                'shap_std_abs_drop': s['std_abs_drop'],
                'lime_avg_abs_drop': l['avg_abs_drop'],
                'lime_std_abs_drop': l['std_abs_drop'],
                'winner_by_pct': 'SHAP' if s['avg_drop_pct'] > l['avg_drop_pct'] else 'LIME',
                'winner_by_abs': 'SHAP' if s['avg_abs_drop'] > l['avg_abs_drop'] else 'LIME',
            }
        save_data['timing'][clf] = {
            'shap_avg_time_s': timing_results[clf]['shap_avg_time'],
            'lime_avg_time_s': timing_results[clf]['lime_avg_time'],
        }

    save_data['detailed_samples'] = {}
    for clf in CLASSIFIERS:
        save_data['detailed_samples'][clf] = {}
        for k in k_values:
            save_data['detailed_samples'][clf][str(k)] = detailed_sample_results[clf][k]

    with open(os.path.join(output_dir, 'rq2_results.json'), 'w') as f:
        json.dump(save_data, f, indent=2)

    print(f"\n  All outputs saved to: {output_dir}/")
    print(f"  Files: rq2_confidence_drop_comparison.png, rq2_absolute_drop_comparison.png,")
    print(f"         rq2_cumulative_masking.png, rq2_faithfulness_heatmap.png,")
    print(f"         rq2_explanation_timing.png, rq2_scatter_*_k*.png,")
    print(f"         rq2_global_shap_beeswarm_*.png, rq2_results.json")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
