# RQ1: Classifier Comparison

**How do modern classifiers compare on the Graphite N-gram embedding?**

This stage trains four classifiers on one shared embedding and compares them, so any
difference in performance is down to the model rather than the features. The baseline is
the original Graphite Random Forest; the challengers are XGBoost, LightGBM, and MLP.

It also records which samples each model gets wrong and where those errors overlap. The
set of samples that every model misclassifies is the evidence for the
feature-representation ceiling, the central argument of the thesis.

## Files

- `comparative_analysis.py`: trains and evaluates the four classifiers on the shared
  embedding, computes all metrics, and writes the misclassification overlap.
- `results/`: the outputs produced by the script.

## Requirements

`comparative_analysis.py` imports the shared modules in `../core`. Add the two lines below
at the top of the script so those imports resolve, or run with `PYTHONPATH=core`:

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "core"))
```

## How to run

From the repository root:

```bash
python RQ1/comparative_analysis.py --dataset-path dataset --classifier all --output-dir RQ1/results
```

Run a single classifier instead of all four:

```bash
python RQ1/comparative_analysis.py --dataset-path dataset --classifier lightgbm --output-dir RQ1/results
```

## Outputs

Written to `results/`:

- `rq1_results.json`: the per-classifier metrics (accuracy, F1, AUC, false positive rate,
  cross-validated F1).
- `misclassification_overlap.json`: the false negatives and false positives each model
  produces, and the samples missed by more than one model.
- The comparison figures (ROC curves, cross-validated F1, false positive rate, confusion
  matrices, timing).

## Result

LightGBM is the strongest classifier, reaching 91.81% accuracy and 91.30% F1, a 3.99 point
F1 gain over the Random Forest baseline, with the highest ROC-AUC and a low false positive
rate. The overlap analysis shows nine samples that every one of the four classifiers gets
wrong, which points to a limit in the N-gram embedding rather than in any single model.
