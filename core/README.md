# core: Shared Graphite Modules

These are the shared building blocks used by the RQ1 and RQ2 experiments. They are kept in
one place rather than duplicated per research question. The RQ scripts import from here.

## Files

- `graphite_n_gram.py`: the Graphite N-gram embedder (`Graphite_Ngram`). It turns a
  process graph into a fixed-size feature vector by taking each thread's ordered event
  sequence, building N-gram counts over it, and pooling across threads. It also defines the
  four classifiers used in RQ1 (Random Forest, XGBoost, LightGBM, MLP) behind one interface.
- `dataprocessor_graphs.py`: the dataset loader (`load_dataset`). It reads the graph
  pickle files and returns them as objects with 5-dimensional node type features and the
  event-typed edges the embedder expects.
- `parameter_parser.py`: command-line argument parsing and the default paths to the two
  lookup tables, `EventName_EdgeFeatures.json` and `NodeType_NodeFeatures.json`, which it
  expects to find under `../dataset`.
- `main.py`: the original Graphite entry point. It loads the dataset, builds embeddings,
  trains a classifier, and reports test performance. Useful as a minimal end-to-end check
  of the modelling path on its own.

## Running main.py

From the repository root:

```bash
python core/main.py
```

Change the N-gram size or the pooling method:

```bash
python core/main.py --N 2
python core/main.py --pool mean
```

## How the pieces fit together

`dataprocessor_graphs.py` loads the graphs, `graphite_n_gram.py` embeds them and classifies
them, and `parameter_parser.py` supplies the paths and settings. The RQ1 and RQ2 scripts in
`../RQ1` and `../RQ2` import all three of these directly, which is why those scripts add
`../core` to the path (or are run with `PYTHONPATH=core`). The lookup tables and the
dataset both live under `../dataset`.
