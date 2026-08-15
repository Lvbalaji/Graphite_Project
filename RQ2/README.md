# RQ2: XAI Faithfulness (SHAP vs LIME)

**Which explanation method more faithfully reflects the classifier's decisions?**

This stage applies SHAP and LIME to the trained classifiers, masks the top-K features each
method ranks as most important, and measures how far the model's confidence drops. A
faithful explainer points at features the model actually relies on, so masking them should
cause a large drop. The method with the larger drop is the more faithful one.

## Files

- `xai_faithfulness_analysis.py`: the head-to-head SHAP vs LIME comparison across all four
  classifiers, using the top-K feature-masking protocol.
- `lightgbm_shap_analyzer.py`: the LightGBM plus SHAP deep dive. It trains and saves the
  final model on first run, then produces the SHAP analysis and figures. This is the script
  that creates the saved model used by the live pipeline (see note below).
- `results_rq2/`: the outputs produced by the scripts.

## Requirements

Both scripts import the shared modules in `../core`. Add the two lines below at the top of
each script so those imports resolve, or run with `PYTHONPATH=core`:

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "core"))
```

## How to run

From the repository root.

SHAP vs LIME faithfulness across several masking depths:

```bash
python RQ2/xai_faithfulness_analysis.py --dataset-path dataset --k-values 3 5 10
```

Analyse the consensus-failure samples, the ones every model gets wrong:

```bash
python RQ2/xai_faithfulness_analysis.py --dataset-path dataset --sample-mode wrong --wrong-consensus all
```

LightGBM plus SHAP deep dive. The first run trains and saves the model, later runs load it
and skip training. Use `--retrain` to force retraining:

```bash
python RQ2/lightgbm_shap_analyzer.py --dataset-path dataset
python RQ2/lightgbm_shap_analyzer.py --dataset-path dataset --retrain
```

## Outputs

`xai_faithfulness_analysis.py` writes to `results_rq2/`: the faithfulness results JSON, plus
the SHAP vs LIME figures (the faithfulness heatmap, confidence-drop and cumulative-masking
charts, per-model beeswarms, and scatter plots).

`lightgbm_shap_analyzer.py` produces its own SHAP figures and, importantly, a `saved_model/`
folder containing `lightgbm_model.joblib`, `graphite_embedder.joblib`, and `model_meta.json`.

## Note on the saved model

The model written by `lightgbm_shap_analyzer.py` is the exact model the live detector in
`RQ3&4/` loads. If you retrain here, copy the refreshed `saved_model/` into `RQ3&4/` so the
live pipeline uses the same model the offline results describe.

## Result

SHAP is the more faithful explainer. Masking SHAP-ranked features causes a much larger
confidence drop than masking LIME-ranked features across every classifier and every value
of K, so SHAP won all twelve comparisons. LIME stays close to flat because it is
structurally unsuited to the correlated N-gram features.
