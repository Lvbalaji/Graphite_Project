import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from typing import List
import torch
import numpy as np
from torch_geometric.data import Data
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


class Graphite_Ngram:
   def __init__(self, N: int = 4, pool: str = "sum", classifier: str = "xgboost"):
      self.N = N
      self.classifier_name = classifier
      pool_choices = {"sum": torch.sum, "mean": torch.mean, "max": torch.max}
      self.pool = pool_choices[pool]
      self.count_vectorizer = CountVectorizer(ngram_range=(N, N))
      self.base_model = self._build_classifier(classifier)
      return

   def _build_classifier(self, classifier: str):
      classifiers = {
         # Baseline: Original Graphite Random Forest
         "rf": RandomForestClassifier(
            n_estimators=500,
            criterion='gini',
            max_depth=20,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            bootstrap=False,
            random_state=42
         ),
         # XGBoost (90.1% accuracy config)
         "xgboost": XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=10,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='logloss',
            random_state=42
         ),
         # LightGBM
         "lightgbm": LGBMClassifier(
            n_estimators=500,
            max_depth=-1,
            num_leaves=63,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=10,
            random_state=42,
            n_jobs=-1,
            verbose=-1
         ),
         # MLP
         "mlp": MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            activation='relu',
            solver='adam',
            alpha=0.001,
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            random_state=42,
            verbose=False
         ),
      }
      if classifier not in classifiers:
         raise ValueError(f"Unknown classifier '{classifier}'. Choose from: {list(classifiers.keys())}")
      return classifiers[classifier]

   def _get_thread_sorted_event_sequence(self, data: Data, thread_node_idx: int) -> List[str]:
      edge_src = data.edge_index[0]
      edge_tar = data.edge_index[1]
      outgoing = torch.nonzero(edge_src == thread_node_idx).flatten()
      incoming = torch.nonzero(edge_tar == thread_node_idx).flatten()

      if outgoing.numel() == 0 and incoming.numel() == 0:
         return []

      edge_feats = torch.cat([data.edge_attr[incoming], data.edge_attr[outgoing]], dim=0)
      sort_by_timestamp = torch.argsort(edge_feats[:, -1], descending=False)
      sorted_feats = edge_feats[sort_by_timestamp]

      eventname_indices = torch.nonzero(sorted_feats[:,:-1], as_tuple=False)[:, -1]
      return [self.eventname_edgefeats[i] for i in eventname_indices]

   def _get_thread_neighboring_nodetypes(self, data: Data, thread_node_idx: int) -> torch.tensor:
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
      return torch.sum(data.x[neighbors], dim=0).view(1,-1)

   def fit_count_vectorizer(self, train_dataset: List[Data]) -> None:
      thread_nodetype = torch.tensor([1 if _type.lower() == "thread" else 0 for _type in self.nodetype_nodefeats])
      all_sequences = []
      for train_data in train_dataset:
         thread_indices = torch.nonzero(torch.all(torch.eq(train_data.x, thread_nodetype), dim=1)).flatten()
         for idx in thread_indices.tolist():
            seq = self._get_thread_sorted_event_sequence(data=train_data, thread_node_idx=idx)
            if seq: all_sequences.append(seq)

      formatted = [' '.join(s) for s in all_sequences if len(s) >= self.N]
      self.count_vectorizer.fit(formatted)

   def generate_graph_embedding(self, data: Data) -> torch.tensor:
      thread_nodetype = torch.tensor([1 if _type.lower() == "thread" else 0 for _type in self.nodetype_nodefeats])
      thread_indices = torch.nonzero(torch.all(torch.eq(data.x, thread_nodetype), dim=1)).flatten()

      all_embs = []
      for idx in thread_indices.tolist():
         seq = self._get_thread_sorted_event_sequence(data=data, thread_node_idx=idx)
         ngram_vec = self.count_vectorizer.transform([" ".join(seq)]).toarray()
         node_emb = torch.cat([self._get_thread_neighboring_nodetypes(data, idx), torch.Tensor(ngram_vec).view(1,-1)], dim=1)
         all_embs.append(node_emb)

      if not all_embs:
          total_dim = len(self.nodetype_nodefeats) + len(self.count_vectorizer.get_feature_names_out())
          return torch.zeros((1, total_dim))

      return self.pool(torch.cat(all_embs, dim=0), dim=0)

   def fit(self, train_dataset: List[Data], nodetype_nodefeats: List[str], eventname_edgefeats: List[str]) -> None:
      self.nodetype_nodefeats, self.eventname_edgefeats = nodetype_nodefeats, eventname_edgefeats
      self.fit_count_vectorizer(train_dataset)

      X, names = [], []
      for train_data in train_dataset:
         X.append(self.generate_graph_embedding(train_data).tolist())
         names.append(train_data.name)

      y = [1 if "malware" in n else 0 for n in names]
      self.base_model.fit(np.array(X), y)
      print(f"[{self.classifier_name.upper()}] Model trained.", flush=True)

   def predict(self, test_data: Data):
      emb = self.generate_graph_embedding(test_data)
      return self.base_model.predict([emb.tolist()])[0]

   def predict_proba(self, test_data: Data):
      emb = self.generate_graph_embedding(test_data)
      return self.base_model.predict_proba([emb.tolist()])[0][1]
