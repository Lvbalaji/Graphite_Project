"""
=============================================================================
RQ1: Comparative Classifier Analysis for Graphite N-gram Framework
=============================================================================
Classifiers: Random Forest (baseline) vs XGBoost vs LightGBM vs MLP

Usage:
  python comparative_analysis.py --dataset-path ../dataset
  python comparative_analysis.py --dataset-path ../dataset --classifier xgboost
  python comparative_analysis.py --dataset-path ../dataset --classifier all --output-dir ./results
=============================================================================
"""
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from parameter_parser import param_parser
from dataprocessor_graphs import load_dataset
from graphite_n_gram import Graphite_Ngram

import os
import json
import time
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, roc_curve, auc
)
from sklearn.model_selection import cross_val_score

CLASSIFIERS = ["rf", "xgboost", "lightgbm", "mlp"]
LABELS = {
    "rf": "Random Forest (Baseline)",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "mlp": "MLP"
}
COLORS = {"rf": "#2196F3", "xgboost": "#FF9800", "lightgbm": "#4CAF50", "mlp": "#9C27B0"}


def extract_embeddings(graphite_ngram, dataset):
    """Extract graph embeddings once — shared across all classifiers."""
    X, y, names = [], [], []
    for data in dataset:
        X.append(graphite_ngram.generate_graph_embedding(data).tolist())
        y.append(1 if "malware" in data.name else 0)
        names.append(data.name)
    return np.array(X), np.array(y), names


def train_and_evaluate(clf_name, X_train, y_train, X_test, y_test, test_names):
    """Train one classifier, run 5-fold CV, and return all metrics."""
    print(f"\n{'='*50}")
    print(f"  Training: {LABELS[clf_name]}")
    print(f"{'='*50}")

    model = Graphite_Ngram(classifier=clf_name).base_model

    # 5-Fold Cross-Validation on training data
    print(f"  Running 5-fold cross-validation...")
    cv_model = Graphite_Ngram(classifier=clf_name).base_model
    cv_scores = cross_val_score(cv_model, X_train, y_train, cv=5, scoring='f1')
    print(f"  CV F1 scores: {[f'{s:.4f}' for s in cv_scores]}")
    print(f"  CV F1 mean: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

    # Train on full training set
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    # Predict on test set
    t0 = time.time()
    probs = model.predict_proba(X_test)[:, 1]
    pred_time = time.time() - t0
    preds = (probs > 0.5).astype(int)

    # Metrics
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)
    tn, fp, fn, tp = cm.ravel()
    fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fpr_roc, tpr_roc, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr_roc, tpr_roc)

    # --- NEW: Track misclassified sample names ---
    # False Negative: actual=1 (malware), predicted=0 (benign)  -> missed malware
    # False Positive: actual=0 (benign),  predicted=1 (malware) -> benign flagged as malware
    fn_samples = [test_names[i] for i in range(len(y_test))
                  if y_test[i] == 1 and preds[i] == 0]
    fp_samples = [test_names[i] for i in range(len(y_test))
                  if y_test[i] == 0 and preds[i] == 1]

    print(f"  Accuracy: {acc:.4f} | F1: {f1:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f}")
    print(f"  FPR: {fpr_val:.4f} | AUC: {roc_auc:.4f} | Train: {train_time:.1f}s | Predict: {pred_time:.1f}s")
    print(f"  FN count: {fn} | FP count: {fp}")

    return {
        "classifier": clf_name, "display_name": LABELS[clf_name],
        "accuracy": acc, "f1_score": f1, "precision": prec, "recall": rec,
        "fpr": fpr_val, "roc_auc": roc_auc,
        "train_time_s": train_time, "predict_time_s": pred_time,
        "cv_f1_mean": float(np.mean(cv_scores)),
        "cv_f1_std": float(np.std(cv_scores)),
        "cv_f1_scores": [float(s) for s in cv_scores],
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "roc_fpr": fpr_roc.tolist(), "roc_tpr": tpr_roc.tolist(),
        "classification_report": classification_report(y_test, preds, output_dict=True),
        "fn_samples": fn_samples,
        "fp_samples": fp_samples,
    }


# ============================================================================
# NEW: Misclassification Overlap Analysis
# ============================================================================

def _invert_to_sample_map(results, clf_list, key):
    """Return dict: sample_name -> list of classifiers that misclassified it."""
    sample_to_clfs = defaultdict(list)
    for clf in clf_list:
        for name in results[clf][key]:
            sample_to_clfs[name].append(clf)
    return sample_to_clfs


def analyze_misclassification_overlap(results, clf_list, output_dir):
    """
    Print per-model FN/FP samples, overlap across models (>= 2 models),
    and any samples that appear in BOTH FN and FP sets across the model pool.
    Also persists the analysis to a JSON file.
    """
    print(f"\n{'='*110}")
    print(f"  MISCLASSIFICATION OVERLAP ANALYSIS")
    print(f"{'='*110}")

    # ---- Per-model listings ----
    for clf in clf_list:
        r = results[clf]
        print(f"\n  [{LABELS[clf]}]")
        print(f"    False Negatives ({len(r['fn_samples'])}): "
              f"{r['fn_samples'] if r['fn_samples'] else '—'}")
        print(f"    False Positives ({len(r['fp_samples'])}): "
              f"{r['fp_samples'] if r['fp_samples'] else '—'}")

    # Only meaningful with >= 2 classifiers
    if len(clf_list) < 2:
        print("\n  (Overlap analysis skipped — need at least 2 classifiers.)")
        return {}

    fn_map = _invert_to_sample_map(results, clf_list, "fn_samples")
    fp_map = _invert_to_sample_map(results, clf_list, "fp_samples")

    # ---- FN overlap: samples missed by >= 2 models ----
    fn_shared = {s: clfs for s, clfs in fn_map.items() if len(clfs) >= 2}
    print(f"\n  {'-'*100}")
    print(f"  SHARED FALSE NEGATIVES (malware missed by >= 2 models): {len(fn_shared)}")
    print(f"  {'-'*100}")
    if fn_shared:
        # Sort: first by how many models missed it (desc), then by name
        for sample in sorted(fn_shared, key=lambda s: (-len(fn_shared[s]), s)):
            clfs = fn_shared[sample]
            tag = "ALL MODELS" if len(clfs) == len(clf_list) else f"{len(clfs)}/{len(clf_list)} models"
            print(f"    [{tag}] {sample}")
            print(f"        missed by: {', '.join(LABELS[c] for c in clfs)}")
    else:
        print("    None — no malware sample was missed by two or more models simultaneously.")

    # ---- FP overlap: benign samples flagged by >= 2 models ----
    fp_shared = {s: clfs for s, clfs in fp_map.items() if len(clfs) >= 2}
    print(f"\n  {'-'*100}")
    print(f"  SHARED FALSE POSITIVES (benign flagged by >= 2 models): {len(fp_shared)}")
    print(f"  {'-'*100}")
    if fp_shared:
        for sample in sorted(fp_shared, key=lambda s: (-len(fp_shared[s]), s)):
            clfs = fp_shared[sample]
            tag = "ALL MODELS" if len(clfs) == len(clf_list) else f"{len(clfs)}/{len(clf_list)} models"
            print(f"    [{tag}] {sample}")
            print(f"        flagged by: {', '.join(LABELS[c] for c in clfs)}")
    else:
        print("    None — no benign sample was flagged by two or more models simultaneously.")

    # ---- Samples that appear in BOTH FN and FP pools across all models ----
    # (i.e. same sample name misclassified as FN by some model AND FP by another)
    fn_names_union = set(fn_map.keys())
    fp_names_union = set(fp_map.keys())
    both = fn_names_union & fp_names_union
    print(f"\n  {'-'*100}")
    print(f"  SAMPLES APPEARING IN BOTH FN AND FP SETS (across models): {len(both)}")
    print(f"  {'-'*100}")
    if both:
        for sample in sorted(both):
            fn_clfs = fn_map[sample]
            fp_clfs = fp_map[sample]
            print(f"    {sample}")
            print(f"        FN by: {', '.join(LABELS[c] for c in fn_clfs)}")
            print(f"        FP by: {', '.join(LABELS[c] for c in fp_clfs)}")
        print("\n    NOTE: A sample appearing in both lists is unusual — it typically indicates "
              "either a label-ambiguous sample or (more commonly) a name collision between "
              "benign and malware files. Worth investigating.")
    else:
        print("    None — no sample name appears in both the FN and FP pools. (Expected.)")

    # ---- Persist to JSON ----
    overlap_summary = {
        "per_model": {
            clf: {
                "fn_samples": results[clf]["fn_samples"],
                "fp_samples": results[clf]["fp_samples"],
                "fn_count": len(results[clf]["fn_samples"]),
                "fp_count": len(results[clf]["fp_samples"]),
            } for clf in clf_list
        },
        "shared_false_negatives": {
            s: [LABELS[c] for c in clfs] for s, clfs in fn_shared.items()
        },
        "shared_false_positives": {
            s: [LABELS[c] for c in clfs] for s, clfs in fp_shared.items()
        },
        "samples_in_both_fn_and_fp": {
            s: {
                "fn_by": [LABELS[c] for c in fn_map[s]],
                "fp_by": [LABELS[c] for c in fp_map[s]],
            } for s in sorted(both)
        },
    }
    with open(os.path.join(output_dir, "misclassification_overlap.json"), "w") as f:
        json.dump(overlap_summary, f, indent=2)
    print(f"\n  Overlap analysis saved to: {os.path.join(output_dir, 'misclassification_overlap.json')}")
    return overlap_summary


# ============================================================================
# Visualization
# ============================================================================

def plot_metrics(results, clf_list, output_dir):
    """Bar chart comparing Accuracy, F1, Precision, Recall."""
    metrics = ['accuracy', 'f1_score', 'precision', 'recall']
    labels = ['Accuracy', 'F1-Score', 'Precision', 'Recall']

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle('RQ1: Comparative Classifier Performance on Graphite N-gram Features', fontsize=14, fontweight='bold')

    for idx, (m, l) in enumerate(zip(metrics, labels)):
        ax = axes[idx]
        vals = [results[c][m] for c in clf_list]
        bars = ax.bar([LABELS[c].replace(' (Baseline)', '\n(Baseline)') for c in clf_list],
                      vals, color=[COLORS[c] for c in clf_list], edgecolor='black', linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.set_ylabel(l)
        ax.set_ylim(0, 1.1)
        ax.tick_params(axis='x', labelsize=8)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metric_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_confusion_matrices(results, clf_list, output_dir):
    """Confusion matrices for classifiers."""
    n = len(clf_list)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5))
    if n == 1: axes = [axes]
    fig.suptitle('Confusion Matrices', fontsize=14, fontweight='bold')

    for idx, clf in enumerate(clf_list):
        ax = axes[idx]
        cm = results[clf]['confusion_matrix']
        matrix = np.array([[cm['tn'], cm['fp']], [cm['fn'], cm['tp']]])
        sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Benign', 'Malware'], yticklabels=['Benign', 'Malware'])
        ax.set_title(f"{LABELS[clf]}\nAcc={results[clf]['accuracy']:.3f} | F1={results[clf]['f1_score']:.3f}")
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrices.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_roc_curves(results, clf_list, output_dir):
    """ROC curves on one figure."""
    fig, ax = plt.subplots(figsize=(8, 7))

    for clf in clf_list:
        r = results[clf]
        ax.plot(r['roc_fpr'], r['roc_tpr'], color=COLORS[clf], linewidth=2,
                label=f"{LABELS[clf]} (AUC={r['roc_auc']:.3f})")

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves — Classifier Comparison')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_fpr(results, clf_list, output_dir):
    """FPR comparison — critical for SOC."""
    fig, ax = plt.subplots(figsize=(8, 5))
    vals = [results[c]['fpr'] for c in clf_list]
    bars = ax.barh([LABELS[c] for c in clf_list], vals,
                   color=[COLORS[c] for c in clf_list], edgecolor='black', linewidth=0.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2.,
                f'{v:.4f}', ha='left', va='center', fontsize=10, fontweight='bold')
    ax.set_xlabel('False Positive Rate')
    ax.set_title('False Positive Rate Comparison (Lower = Better)')
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fpr_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_timing(results, clf_list, output_dir):
    """Training time comparison."""
    fig, ax = plt.subplots(figsize=(8, 5))
    vals = [results[c]['train_time_s'] for c in clf_list]
    bars = ax.bar([LABELS[c].replace(' (Baseline)', '\n(Baseline)') for c in clf_list],
                  vals, color=[COLORS[c] for c in clf_list], edgecolor='black', linewidth=0.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.2,
                f'{v:.1f}s', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_ylabel('Training Time (seconds)')
    ax.set_title('Training Time Comparison')
    ax.tick_params(axis='x', labelsize=8)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'timing_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_cv_f1(results, clf_list, output_dir):
    """5-Fold Cross-Validation F1-Score with error bars."""
    fig, ax = plt.subplots(figsize=(8, 5))

    means = [results[c]['cv_f1_mean'] for c in clf_list]
    stds = [results[c]['cv_f1_std'] for c in clf_list]

    bars = ax.bar(
        [LABELS[c].replace(' (Baseline)', '\n(Baseline)') for c in clf_list],
        means, yerr=stds,
        color=[COLORS[c] for c in clf_list],
        edgecolor='black', linewidth=0.5,
        capsize=5, error_kw={'linewidth': 2}
    )

    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + std + 0.01,
                f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_ylabel('F1-Score', fontsize=12)
    ax.set_title('5-Fold Cross-Validation F1-Score\n(Error bars = ±1 std. dev.)', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.15)
    ax.tick_params(axis='x', labelsize=8)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cv_f1_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# Main
# ============================================================================

def main(args):
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Determine which classifiers to run
    if args.classifier == "all":
        clf_list = CLASSIFIERS
    else:
        clf_list = [args.classifier]

    print("=" * 60)
    print("  RQ1: COMPARATIVE CLASSIFIER ANALYSIS")
    if len(clf_list) == 1:
        print(f"  Classifier: {LABELS[clf_list[0]]}")
    else:
        print("  RF (baseline) vs XGBoost vs LightGBM vs MLP")
    print("=" * 60)

    # Load data
    nodetype_nodefeats = json.load(open(args.nodetype_nodefeats_path, "r"))
    eventname_edgefeats = json.load(open(args.eventname_edgefeats_path, "r"))

    train_dataset = load_dataset(
        os.path.join(args.dataset_path, "train/benign"),
        os.path.join(args.dataset_path, "train/malware"),
        len(nodetype_nodefeats), len(eventname_edgefeats) + 1)

    test_dataset = load_dataset(
        os.path.join(args.dataset_path, "test/benign"),
        os.path.join(args.dataset_path, "test/malware"),
        len(nodetype_nodefeats), len(eventname_edgefeats) + 1)

    # Fit vectorizer and extract embeddings ONCE
    print("\n[Step 1] Fitting N-gram vectorizer...")
    base = Graphite_Ngram(N=args.N, pool=args.pool, classifier="rf")
    base.nodetype_nodefeats = nodetype_nodefeats
    base.eventname_edgefeats = eventname_edgefeats
    base.fit_count_vectorizer(train_dataset)

    print("[Step 2] Extracting training embeddings...")
    X_train, y_train, _ = extract_embeddings(base, train_dataset)
    print(f"  Train: {len(X_train)} samples | Features: {X_train.shape[1]}")

    print("[Step 3] Extracting test embeddings...")
    X_test, y_test, test_names = extract_embeddings(base, test_dataset)
    print(f"  Test: {len(X_test)} samples")

    # Run classifiers
    print(f"\n[Step 4] Training and evaluating classifiers...")
    results = {}
    for clf in clf_list:
        results[clf] = train_and_evaluate(clf, X_train, y_train, X_test, y_test, test_names)

    # ---- Summary table (now includes FN column) ----
    print(f"\n{'='*120}")
    print(f"  RQ1 RESULTS SUMMARY")
    print(f"{'='*120}")
    header = (f"{'Classifier':<30} | {'Acc':>7} | {'F1':>7} | {'Prec':>7} | {'Rec':>7} | "
              f"{'FPR':>7} | {'AUC':>7} | {'FN':>4} | {'FP':>4} | {'CV-F1':>12} | {'Train':>8}")
    print(header)
    print("-" * len(header))
    for clf in clf_list:
        r = results[clf]
        cv_str = f"{r['cv_f1_mean']:.3f}±{r['cv_f1_std']:.3f}"
        cm = r['confusion_matrix']
        print(f"{r['display_name']:<30} | {r['accuracy']:>7.4f} | {r['f1_score']:>7.4f} | "
              f"{r['precision']:>7.4f} | {r['recall']:>7.4f} | {r['fpr']:>7.4f} | "
              f"{r['roc_auc']:>7.4f} | {cm['fn']:>4d} | {cm['fp']:>4d} | "
              f"{cv_str:>12} | {r['train_time_s']:>7.1f}s")

    # Best classifier (only meaningful when running all)
    if len(clf_list) > 1:
        best = max(results, key=lambda k: results[k]['f1_score'])
        baseline_f1 = results['rf']['f1_score'] if 'rf' in results else 0
        best_f1 = results[best]['f1_score']
        print(f"\n  OPTIMAL MODEL: {LABELS[best]} (F1={best_f1:.4f}, Acc={results[best]['accuracy']:.4f})")
        if best != 'rf' and baseline_f1 > 0:
            print(f"  Improvement over RF baseline: {((best_f1 - baseline_f1) / baseline_f1) * 100:+.2f}%")

    # ---- NEW: Misclassification overlap analysis ----
    analyze_misclassification_overlap(results, clf_list, output_dir)

    # Generate plots
    print(f"\n[Step 5] Generating visualizations...")
    if len(clf_list) > 1:
        plot_metrics(results, clf_list, output_dir)
        plot_fpr(results, clf_list, output_dir)
        plot_timing(results, clf_list, output_dir)
        plot_cv_f1(results, clf_list, output_dir)
    plot_confusion_matrices(results, clf_list, output_dir)
    plot_roc_curves(results, clf_list, output_dir)

    # Save JSON (keep fn_samples / fp_samples; drop bulky ROC arrays)
    save_results = {}
    for clf in results:
        save_results[clf] = {k: v for k, v in results[clf].items() if k not in ['roc_fpr', 'roc_tpr']}
    with open(os.path.join(output_dir, 'rq1_results.json'), 'w') as f:
        json.dump(save_results, f, indent=2)

    print(f"\n  All outputs saved to: {output_dir}/")
    if len(clf_list) > 1:
        print(f"  Files: metric_comparison.png, confusion_matrices.png, roc_curves.png,")
        print(f"         fpr_comparison.png, timing_comparison.png, cv_f1_comparison.png,")
        print(f"         rq1_results.json, misclassification_overlap.json")
    else:
        print(f"  Files: confusion_matrices.png, roc_curves.png, rq1_results.json,")
        print(f"         misclassification_overlap.json")
    print(f"{'='*60}")


if __name__ == "__main__":
    args = param_parser()
    main(args)
