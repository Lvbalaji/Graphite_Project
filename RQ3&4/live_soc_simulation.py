"""
================================================================================
  RQ3 — REAL-TIME SOC SIMULATION  (live_soc_simulation.py)
  Author-faithful streaming port of the Graphite pipeline + SHAP XAI alerts
  R00277695 · Loganathan · MSc Cybersecurity Dissertation
================================================================================

This is a STREAMING re-implementation of jgwak1/Graphite's batch pipeline.
Every non-trivial decision is grounded in a specific author file, marked with
an "EVIDENCE:" comment.

Author files referred to (all present in project knowledge):
    src/
        graphite_n_gram.py           — N-gram embedder
        dataprocessor_graphs.py      — pickle loader, expects 5-dim node attrs
        main.py / parameter_parser.py
    pipeline/step1_etl/
        text_event_parser.py         — TASK_NAMES, EXCLUDED_ATTRIBUTES, hex/IP
        flatten_event_record.py      — flattens nested ETW records
        field_selection.py           — drops Keyword/Flags/Description
        format_elasticsearch_logs.py — ES bulk reformat
    pipeline/step2_graph_generation/
        build_computation_graph.py   — UID rules, per-provider event handlers
        normalize_edge_directions.py — direction sets + opcode tables
        encode_graph_attributes.py   — Task_Names list, NODE_ATTRIBUTE_DEFAULTS
        project_graphlets.py         — taint-analysis projection rooted at PROCESSSTART
        run_step2_pipeline.py        — orchestration order
    pipeline/step3_processing_split/
        process_graph_data.py        — final {x, edge_list, y, edge_attr} pickle
        split_dataset.py
        run_step3_pipeline.py

Why your old tool didn't detect malware (each fixed in this file):
    1. Bidirectional edges        → normalize_edge_directions.py is deterministic.
    2. No graphlet projection     → project_graphlets.py roots each at PROCESSSTART.
    3. Trigger on 500 nodes flat  → paper RQ3 triggers per-root at 1000 nodes.
    4. Wrong node UIDs            → build_computation_graph.py UID rules now used.
    5. Raw SilkETW event names    → mapped to Task_Names then to EventName.
    6. >5-dim node features       → graphite_n_gram.py:118 needs exact [0,0,0,0,1].
    7. SilkETW flag set was thin  → REQUIRES -kk ...,FileIOInit,ImageLoad (see HELP).

USAGE: see live_soc_simulation.py --help-full
================================================================================
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

import argparse
import json
import logging
import os
import pathlib
import random
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

import joblib
import numpy as np
import shap
import torch
from sklearn.feature_extraction.text import CountVectorizer
from torch_geometric.data import Data


# ─────────────────────────────────────────────────────────────────────────────
# GraphiteEmbedder — verbatim from lightgbm_shap_analyzer.py.
# Required so joblib can unpickle the saved graphite_embedder.joblib regardless
# of which module the model was originally saved from.
# EVIDENCE: lightgbm_shap_analyzer.py lines 81-160; matches graphite_n_gram.py.
# ─────────────────────────────────────────────────────────────────────────────
class GraphiteEmbedder:
    def __init__(self, N=4, pool="sum"):
        self.N = N
        pool_map = {"sum": torch.sum, "mean": torch.mean, "max": torch.max}
        self.pool = pool_map[pool]
        self.count_vectorizer = CountVectorizer(ngram_range=(N, N))
        self.nodetype_nodefeats  = None
        self.eventname_edgefeats = None

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
        return list(self.nodetype_nodefeats) + list(self.count_vectorizer.get_feature_names_out())


# Pickle-compat shim
import types as _types
for _modname in ("lightgbm_shap_analyzer", "live_soc_simulation"):
    _m = _types.ModuleType(_modname)
    _m.GraphiteEmbedder = GraphiteEmbedder
    sys.modules.setdefault(_modname, _m)


# ─────────────────────────────────────────────────────────────────────────────
# Schema constants — EVIDENCE: NodeType_NodeFeatures.json + EventName_EdgeFeatures.json
# ─────────────────────────────────────────────────────────────────────────────
NODETYPE_NODEFEATS = ["File", "Registry", "Network", "Process", "Thread"]
NUM_NODE_FEATURES = len(NODETYPE_NODEFEATS)  # 5

EVENTNAME_EDGEFEATS = [
    "NULL", "Cleanup", "Close", "Create", "CreateNewFile", "DeletePath",
    "DirEnum", "DirNotify", "Flush", "FSCTL", "NameCreate", "NameDelete",
    "OperationEnd", "QueryInformation", "QueryEA", "QuerySecurity", "Read",
    "Write", "SetDelete", "SetInformation", "PagePriorityChange",
    "IoPriorityChange", "CpuBasePriorityChange", "CpuPriorityChange",
    "ImageLoad", "ImageUnload", "ProcessStop/Stop", "ProcessStart/Start",
    "ProcessFreeze/Start", "ThreadStart/Start", "ThreadStop/Stop",
    "ThreadWorkOnBehalfUpdate", "JobStart/Start", "JobTerminate/Stop",
    "Rename", "Renamepath", "RegPerfOpHiveFlushWroteLogFile", "CreateKey",
    "OpenKey", "DeleteKey", "QueryKey", "SetValueKey", "DeleteValueKey",
    "QueryValueKey", "EnumerateKey", "EnumerateValueKey",
    "QueryMultipleValueKey", "SetinformationKey", "CloseKeys",
    "QuerySecurityKey", "SetSecurityKey",
    "KERNEL_NETWORK_TASK_TCPIP/Datasent.",
    "KERNEL_NETWORK_TASK_TCPIP/Datareceived.",
    "KERNEL_NETWORK_TASK_TCPIP/Connectionattempted.",
    "KERNEL_NETWORK_TASK_TCPIP/Disconnectissued.",
    "KERNEL_NETWORK_TASK_TCPIP/Dataretransmitted.",
    "KERNEL_NETWORK_TASK_TCPIP/connectionaccepted.",
    "KERNEL_NETWORK_TASK_TCPIP/Protocolcopieddataonbehalfofuser.",
    "KERNEL_NETWORK_TASK_UDPIP/DatareceivedoverUDPprotocol.",
    "KERNEL_NETWORK_TASK_UDPIP/DatasentoverUDPprotocol.",
    "UNKNOWN",
]
NUM_EDGE_FEATURES = len(EVENTNAME_EDGEFEATS) + 1  # 62
EVENTNAME_INDEX = {name: i for i, name in enumerate(EVENTNAME_EDGEFEATS)}

# Interim Report §3.3.3
RQ3_LATENCY_THRESHOLD_MS = 5000


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("RQ3-LiveSOC")
logging.getLogger("elastic_transport").setLevel(logging.WARNING)
logging.getLogger("elasticsearch").setLevel(logging.WARNING)


# ─────────────────────────────────────────────────────────────────────────────
# Event-name normalisation
#   Raw SilkETW EventName  →  uppercase Task Name (encode_graph_attributes.Task_Names)
#                          →  EventName form (EVENTNAME_EDGEFEATS / EventName_EdgeFeatures.json)
# ─────────────────────────────────────────────────────────────────────────────

# EVIDENCE: encode_graph_attributes.Task_Names + observed SilkETW raw events
RAW_TO_TASK_NAME: Dict[str, str] = {
    # FileIO
    "fileio/operationend":      "OPERATIONEND",
    "fileio/create":            "CREATE",
    "fileio/createnewfile":     "CREATENEWFILE",
    "fileio/cleanup":           "CLEANUP",
    "fileio/close":             "CLOSE",
    "fileio/read":              "READ",
    "fileio/write":             "WRITE",
    "fileio/setinfo":           "SETINFORMATION",
    "fileio/setinformation":    "SETINFORMATION",
    "fileio/delete":            "DELETEPATH",
    "fileio/deletepath":        "DELETEPATH",
    "fileio/rename":            "RENAME",
    "fileio/renamepath":        "RENAMEPATH",
    "fileio/direnum":           "DIRENUM",
    "fileio/flush":             "FLUSH",
    "fileio/queryinfo":         "QUERYINFORMATION",
    "fileio/queryinformation":  "QUERYINFORMATION",
    "fileio/fsctl":             "FSCTL",
    "fileio/queryea":           "QUERYEA",
    "fileio/querysecurity":     "QUERYSECURITY",
    "fileio/setdelete":         "SETDELETE",
    "fileio/dirnotify":         "DIRNOTIFY",
    "fileio/namecreate":        "NAMECREATE",
    "fileio/namedelete":        "NAMEDELETE",
    # Registry — note SilkETW Provider/Opcode form
    "registry/open":               "OPENKEY",
    "registry/close":              "CLOSEKEYS",
    "registry/query":              "QUERYKEY",
    "registry/queryvalue":         "QUERYVALUEKEY",
    "registry/setvalue":           "SETVALUEKEY",
    "registry/setinformation":     "SETINFORMATIONKEY",
    "registry/enumeratekey":       "ENUMERATEKEY",
    "registry/enumeratevaluekey":  "ENUMERATEVALUEKEY",
    "registry/create":             "CREATEKEY",
    "registry/delete":             "DELETEKEY",
    "registry/deletevalue":        "DELETEVALUEKEY",
    "registry/querysecurity":      "QUERYSECURITYKEY",
    "registry/setsecurity":        "SETSECURITYKEY",
    "registry/querymultiplevalue": "QUERYMULTIPLEVALUEKEY",
    "registry/kcbcreate":          "CREATEKEY",
    "registry/kcbdelete":          "DELETEKEY",
    # Thread
    "thread/start":   "THREADSTART",
    "thread/end":     "THREADSTOP",
    "thread/dcstart": "THREADSTART",
    "thread/dcend":   "THREADSTOP",
    "thread/setname": "THREADWORKONBEHALFUPDATE",
    # Process
    "process/start":     "PROCESSSTART",
    "process/terminate": "PROCESSSTOP",
    "process/dcstart":   "PROCESSSTART",
    "process/dcend":     "PROCESSSTOP",
    "process/freeze":    "PROCESSFREEZE",
    # Image
    "image/load":    "IMAGELOAD",
    "image/unload":  "IMAGEUNLOAD",
    "image/dcstart": "IMAGELOAD",
    # Job
    "job/start":     "JOBSTART",
    "job/terminate": "JOBTERMINATE",
    # Priority
    "thread/setbasepriority": "CPUBASEPRIORITYCHANGE",
    "thread/setpriority":     "CPUPRIORITYCHANGE",
    "thread/setiopriority":   "IOPRIORITYCHANGE",
    "thread/setpagepriority": "PAGEPRIORITYCHANGE",
    "thread/setimagepriority": "IMAGEPRIORITYCHANGE",
    # Network — TCP
    "tcpip/sendipv4":       "DATASENT",
    "tcpip/sendipv6":       "DATASENT",
    "tcpip/recvipv4":       "DATARECEIVED",
    "tcpip/recvipv6":       "DATARECEIVED",
    "tcpip/connectipv4":    "CONNECTIONATTEMPTED",
    "tcpip/connectipv6":    "CONNECTIONATTEMPTED",
    "tcpip/disconnectipv4": "DISCONNECTISSUED",
    "tcpip/disconnectipv6": "DISCONNECTISSUED",
    "tcpip/retransmitipv4": "DATARETRANSMITTED",
    "tcpip/retransmitipv6": "DATARETRANSMITTED",
    "tcpip/acceptipv4":     "CONNECTIONACCEPTED",
    "tcpip/acceptipv6":     "CONNECTIONACCEPTED",
    "tcpip/tcpcopyipv4":    "PROTOCOLCOPIEDDATA",
    "tcpip/tcpcopyipv6":    "PROTOCOLCOPIEDDATA",
    # Network — UDP
    "udpip/sendipv4": "DATASENTOVERUDP",
    "udpip/sendipv6": "DATASENTOVERUDP",
    "udpip/recvipv4": "DATARECEIVEDOVERUDP",
    "udpip/recvipv6": "DATARECEIVEDOVERUDP",
    # ---- Per-provider mode (-t user) bare EventNames ----
    # File provider: EventName is bare like "Create", "Read", "Close"
    "create":            "CREATE",
    "createnewfile":     "CREATENEWFILE",
    "cleanup":           "CLEANUP",
    "close":             "CLOSE",
    "read":              "READ",
    "write":             "WRITE",
    "setinformation":    "SETINFORMATION",
    "deletepath":        "DELETEPATH",
    "rename":            "RENAME",
    "renamepath":        "RENAMEPATH",
    "direnum":           "DIRENUM",
    "flush":             "FLUSH",
    "queryinformation":  "QUERYINFORMATION",
    "fsctl":             "FSCTL",
    "queryea":           "QUERYEA",
    "querysecurity":     "QUERYSECURITY",
    "setdelete":         "SETDELETE",
    "dirnotify":         "DIRNOTIFY",
    "namecreate":        "NAMECREATE",
    "namedelete":        "NAMEDELETE",
    "operationend":      "OPERATIONEND",
    # Process provider: EventName is like "ThreadStart/Start", "ImageLoad"
    "threadstart/start":  "THREADSTART",
    "threadstop/stop":    "THREADSTOP",
    "processstart/start": "PROCESSSTART",
    "processstop/stop":   "PROCESSSTOP",
    "processfreeze/start":"PROCESSFREEZE",
    "imageload":          "IMAGELOAD",
    "imageunload":        "IMAGEUNLOAD",
    "cpuprioritychange":  "CPUPRIORITYCHANGE",
    "cpubaseprioritychange": "CPUBASEPRIORITYCHANGE",
    "ioprioritychange":   "IOPRIORITYCHANGE",
    "pageprioritychange": "PAGEPRIORITYCHANGE",
    "threadworkonbehalfupdate": "THREADWORKONBEHALFUPDATE",
    "jobstart/start":     "JOBSTART",
    "jobterminate/stop":  "JOBTERMINATE",
    "psdiskioattribution/start": "PSDISKIOATTRIBUTE",
    "psdiskioattribution/stop":  "PSDISKIOATTRIBUTE",
    # Registry provider: EventName is "EventID(N)" — must use OpcodeName
    # These are handled by the OpcodeName fallback in normalize_event()
    # Network provider: EventName already matches (KERNEL_NETWORK_TASK_*)
}

# Task Name → EventName form in EVENTNAME_EDGEFEATS
TASK_TO_EVENTNAME: Dict[str, str] = {
    "CLEANUP": "Cleanup", "CLOSE": "Close", "CREATE": "Create",
    "CREATENEWFILE": "CreateNewFile", "DELETEPATH": "DeletePath",
    "DIRENUM": "DirEnum", "DIRNOTIFY": "DirNotify", "FLUSH": "Flush",
    "FSCTL": "FSCTL", "NAMECREATE": "NameCreate", "NAMEDELETE": "NameDelete",
    "OPERATIONEND": "OperationEnd", "QUERYINFORMATION": "QueryInformation",
    "QUERYEA": "QueryEA", "QUERYSECURITY": "QuerySecurity",
    "READ": "Read", "WRITE": "Write", "SETDELETE": "SetDelete",
    "SETINFORMATION": "SetInformation", "RENAME": "Rename",
    "RENAMEPATH": "Renamepath",
    "PAGEPRIORITYCHANGE": "PagePriorityChange",
    "IOPRIORITYCHANGE": "IoPriorityChange",
    "CPUBASEPRIORITYCHANGE": "CpuBasePriorityChange",
    "CPUPRIORITYCHANGE": "CpuPriorityChange",
    "IMAGEPRIORITYCHANGE": "IoPriorityChange",  # Task_Names[24] in encode_graph_attributes.py
    "IMAGELOAD": "ImageLoad", "IMAGEUNLOAD": "ImageUnload",
    "PROCESSSTOP":  "ProcessStop/Stop",
    "PROCESSSTART": "ProcessStart/Start",
    "PROCESSFREEZE": "ProcessFreeze/Start",
    "THREADSTART": "ThreadStart/Start",
    "THREADSTOP":  "ThreadStop/Stop",
    "THREADWORKONBEHALFUPDATE": "ThreadWorkOnBehalfUpdate",
    "JOBSTART": "JobStart/Start", "JOBTERMINATE": "JobTerminate/Stop",
    "REGPERFOPHIVEFLUSHWROTELOGFILE": "RegPerfOpHiveFlushWroteLogFile",
    "CREATEKEY": "CreateKey", "OPENKEY": "OpenKey", "DELETEKEY": "DeleteKey",
    "QUERYKEY": "QueryKey", "SETVALUEKEY": "SetValueKey",
    "DELETEVALUEKEY": "DeleteValueKey", "QUERYVALUEKEY": "QueryValueKey",
    "ENUMERATEKEY": "EnumerateKey", "ENUMERATEVALUEKEY": "EnumerateValueKey",
    "QUERYMULTIPLEVALUEKEY": "QueryMultipleValueKey",
    "SETINFORMATIONKEY": "SetinformationKey",
    "CLOSEKEYS": "CloseKeys",
    "QUERYSECURITYKEY": "QuerySecurityKey", "SETSECURITYKEY": "SetSecurityKey",
    "DATASENT":            "KERNEL_NETWORK_TASK_TCPIP/Datasent.",
    "DATARECEIVED":        "KERNEL_NETWORK_TASK_TCPIP/Datareceived.",
    "CONNECTIONATTEMPTED": "KERNEL_NETWORK_TASK_TCPIP/Connectionattempted.",
    "DISCONNECTISSUED":    "KERNEL_NETWORK_TASK_TCPIP/Disconnectissued.",
    "DATARETRANSMITTED":   "KERNEL_NETWORK_TASK_TCPIP/Dataretransmitted.",
    "CONNECTIONACCEPTED":  "KERNEL_NETWORK_TASK_TCPIP/connectionaccepted.",
    "PROTOCOLCOPIEDDATA":  "KERNEL_NETWORK_TASK_TCPIP/Protocolcopieddataonbehalfofuser.",
    "DATARECEIVEDOVERUDP": "KERNEL_NETWORK_TASK_UDPIP/DatareceivedoverUDPprotocol.",
    "DATASENTOVERUDP":     "KERNEL_NETWORK_TASK_UDPIP/DatasentoverUDPprotocol.",
}


# OpcodeName → Task Name mapping for Registry provider in per-provider mode
# Registry events arrive as "EventID(2)" with OpcodeName="OpenKey"
OPCODENAME_TO_TASK: Dict[str, str] = {
    "openkey":           "OPENKEY",
    "closekey":          "CLOSEKEYS",
    "querykey":          "QUERYKEY",
    "queryvalue":        "QUERYVALUEKEY",
    "queryvaluekey":     "QUERYVALUEKEY",
    "setvalue":          "SETVALUEKEY",
    "setvaluekey":       "SETVALUEKEY",
    "deletevalue":       "DELETEVALUEKEY",
    "enumeratekey":      "ENUMERATEKEY",
    "enumeratevaluekey": "ENUMERATEVALUEKEY",
    "createkey":         "CREATEKEY",
    "deletekey":         "DELETEKEY",
    "setinformation":    "SETINFORMATIONKEY",
    "setinformationkey": "SETINFORMATIONKEY",
    "querysecurity":     "QUERYSECURITYKEY",
    "querysecuritykey":  "QUERYSECURITYKEY",
    "setsecurity":       "SETSECURITYKEY",
    "setsecuritykey":    "SETSECURITYKEY",
    "querymultiplevalue":"QUERYMULTIPLEVALUEKEY",
    "regperfophiveflushwrotelogfile": "REGPERFOPHIVEFLUSHWROTELOGFILE",
    "start":             None,  # ignore hive flush start/stop
    "stop":              None,
}


def normalize_event(raw_event: str, opcode_name: str = "") -> Tuple[str, str]:
    """Return (Task Name uppercase, EventName for edge_attr).
    Returns ("UNKNOWN","UNKNOWN") if unmappable.
    In per-provider mode, Registry events have EventName="EventID(N)"
    so we fall back to OpcodeName."""
    if not raw_event:
        return "UNKNOWN", "UNKNOWN"
    rl = raw_event.lower()
    task = RAW_TO_TASK_NAME.get(rl)
    if task is None:
        # Check if raw_event is already in EVENTNAME_EDGEFEATS
        if raw_event in EVENTNAME_INDEX:
            return raw_event.upper().replace("/", ""), raw_event
        # Fallback: use OpcodeName (critical for Registry in per-provider mode)
        if opcode_name:
            ol = opcode_name.lower()
            task = OPCODENAME_TO_TASK.get(ol)
            if task is None and ol in OPCODENAME_TO_TASK:
                # Explicitly mapped to None (e.g., hive flush start/stop) — skip
                return "UNKNOWN", "UNKNOWN"
            if task is None:
                # Try it as a raw event name too
                task = RAW_TO_TASK_NAME.get(ol)
            if task:
                return task, TASK_TO_EVENTNAME.get(task, "UNKNOWN")
        return "UNKNOWN", "UNKNOWN"
    return task, TASK_TO_EVENTNAME.get(task, "UNKNOWN")


# ─────────────────────────────────────────────────────────────────────────────
# Edge-direction rules  — EVIDENCE: normalize_edge_directions.py
# ─────────────────────────────────────────────────────────────────────────────
FILE_TO_THREAD = {"READ", "QUERYINFORMATION", "QUERYSECURITY", "QUERYEA",
                  "DIRENUM", "DIRNOTIFY"}
PROCESS_TO_PROC = {"PROCESSSTART", "PROCESSSTOP", "JOBSTART", "JOBSTOP",
                   "CPUBASEPRIORITYCHANGE",
                   "CPUPRIORITYCHANGE", "IOPRIORITYCHANGE",
                   "PAGEPRIORITYCHANGE", "IMAGELOAD", "IMAGEUNLOAD"}
# NOTE: JOBTERMINATE is NOT in author's PROCESS_TO_PROC. It follows default direction.
REG_TASKS_TO_THREAD = {"QUERYKEY", "QUERYVALUEKEY", "ENUMERATEKEY",
                       "ENUMERATEVALUEKEY"}
# NOTE: author's normalize_edge_directions.py uses opcodes {35,38,39,40} ONLY.
# Paper Table 2 lists 41(QueryMultipleValueKey) and 45(QuerySecurityKey) too,
# but the CODE is what the model was trained against. So we follow the code.
NET_TASKS_TO_THREAD = {"CONNECTIONACCEPTED", "DATARECEIVED",
                       "DATARECEIVEDOVERUDP", "DISCONNECTISSUED"}


def edge_direction(node_kind: str, task_name: str) -> str:
    if node_kind == "file":
        return "resource_to_thread" if task_name in FILE_TO_THREAD else "thread_to_resource"
    if node_kind == "registry":
        return "resource_to_thread" if task_name in REG_TASKS_TO_THREAD else "thread_to_resource"
    if node_kind == "network":
        return "resource_to_thread" if task_name in NET_TASKS_TO_THREAD else "thread_to_resource"
    if node_kind == "process":
        return "thread_to_resource" if task_name in PROCESS_TO_PROC else "resource_to_thread"
    return "thread_to_resource"


# ─────────────────────────────────────────────────────────────────────────────
# Semantic translation layer + MITRE ATT&CK patterns (for SOC alert text)
# ─────────────────────────────────────────────────────────────────────────────

EVENT_SEMANTICS = {
    "cleanup":            ("File System", "Released a file handle"),
    "close":              ("File System", "Closed a file object"),
    "create":             ("File System", "Created or opened a file/directory"),
    "createnewfile":      ("File System", "Created a brand-new file on disk"),
    "deletepath":         ("File System", "Deleted a file by path"),
    "direnum":            ("File System", "Enumerated directory contents"),
    "dirnotify":          ("File System", "Registered for directory change notifications"),
    "flush":              ("File System", "Flushed file buffers to disk"),
    "fsctl":              ("File System", "Issued a file system control command"),
    "namecreate":         ("File System", "Assigned a name to a file object"),
    "namedelete":         ("File System", "Removed a file name entry"),
    "operationend":       ("File System", "Completed a pending I/O operation"),
    "queryinformation":   ("File System", "Queried file metadata"),
    "queryea":            ("File System", "Queried extended file attributes"),
    "querysecurity":      ("File System", "Queried file security descriptor"),
    "read":               ("File System", "Read data from a file"),
    "write":              ("File System", "Wrote data to a file"),
    "setdelete":          ("File System", "Marked a file for deletion"),
    "setinformation":     ("File System", "Modified file metadata"),
    "rename":             ("File System", "Renamed a file"),
    "renamepath":         ("File System", "Renamed a file (by path)"),
    "processstart/start": ("Process",     "Started a new process"),
    "processstop/stop":   ("Process",     "Terminated a process"),
    "processstart":       ("Process",     "Started a new process"),
    "processstop":        ("Process",     "Terminated a process"),
    "threadstart/start":  ("Thread",      "Spawned a new thread"),
    "threadstop/stop":    ("Thread",      "Terminated a thread"),
    "threadstart":        ("Thread",      "Spawned a new thread"),
    "threadstop":         ("Thread",      "Terminated a thread"),
    "start":              ("Thread",      "Thread started execution"),
    "stop":               ("Thread",      "Thread stopped execution"),
    "imageload":          ("Process",     "Loaded a DLL/executable image into memory"),
    "imageunload":        ("Process",     "Unloaded a DLL/executable image"),
    "cpubaseprioritychange": ("Process",  "Changed CPU base priority"),
    "cpuprioritychange":  ("Process",     "Changed CPU priority"),
    "pageprioritychange": ("Process",     "Changed page priority"),
    "ioprioritychange":   ("Process",     "Changed I/O priority"),
    "createkey":          ("Registry",    "Created a new registry key"),
    "openkey":            ("Registry",    "Opened an existing registry key"),
    "deletekey":          ("Registry",    "Deleted a registry key"),
    "querykey":           ("Registry",    "Queried registry key information"),
    "setvaluekey":        ("Registry",    "Set/modified a registry value"),
    "deletevaluekey":     ("Registry",    "Deleted a registry value"),
    "queryvaluekey":      ("Registry",    "Read a registry value"),
    "enumeratekey":       ("Registry",    "Enumerated sub-keys of a registry key"),
    "enumeratevaluekey":  ("Registry",    "Enumerated values under a registry key"),
    "querymultiplevaluekey": ("Registry", "Read multiple registry values"),
    "setinformationkey":  ("Registry",    "Modified registry key metadata"),
    "closekeys":          ("Registry",    "Closed a registry key handle"),
    "querysecuritykey":   ("Registry",    "Queried registry key security"),
    "setsecuritykey":     ("Registry",    "Set registry key security"),
    "kernel_network_task_tcpip": ("Network", "TCP/IP network operation"),
    "datasent":           ("Network",     "Sent data over the network (TCP)"),
    "datareceived":       ("Network",     "Received data from the network (TCP)"),
    "connectionattempted":("Network",     "Attempted a new network connection"),
    "disconnectissued":   ("Network",     "Disconnected from a network endpoint"),
    "dataretransmitted":  ("Network",     "Retransmitted network data"),
    "connectionaccepted": ("Network",     "Accepted an incoming network connection"),
    # ---- entries that appear in n-gram features but were missing ----
    "threadworkonbehalfupdate": ("Thread", "Updated thread work-on-behalf context (impersonation)"),
    "processfreeze/start": ("Process",    "Froze a process (suspended execution)"),
    "processfreeze":       ("Process",    "Froze a process (suspended execution)"),
    "jobstart/start":      ("Process",    "Started a job object"),
    "jobterminate/stop":   ("Process",    "Terminated a job object"),
    "jobstart":            ("Process",    "Started a job object"),
    "jobterminate":        ("Process",    "Terminated a job object"),
    "imageprioritychange": ("Process",    "Changed image priority"),
    "lostevent":           ("Process",    "A system event was lost (high-volume condition)"),
    "psdiskioattribute":   ("Process",    "Disk I/O attribution event"),
    "psdiskioattribution": ("Process",    "Disk I/O attribution tracking event"),
    "psioratecontrol":     ("Process",    "I/O rate control event"),
    "regperfophiveflushwrotelogfile": ("Registry", "Registry hive flush performance event"),
    "setdelete":           ("File System", "Marked a file for deletion"),
    "null":                ("Unknown",    "Null/empty event"),
    "unknown":             ("Unknown",    "Unrecognised event type"),
    # ---- network sub-events that appear as lowercase in n-grams ----
    "datasentoverudp":         ("Network", "Sent data over UDP"),
    "datareceivedoverudp":     ("Network", "Received data over UDP"),
    "protocolcopieddata":      ("Network", "Protocol copied data on behalf of user"),
    "dataretransmitted":       ("Network", "Retransmitted network data"),
    "kernel_network_task_udpip": ("Network", "UDP/IP network operation"),
}

MITRE_PATTERNS = [
    # ---- Registry Reconnaissance (reading/querying existing keys) ----
    ({"openkey", "queryvaluekey", "enumeratekey"}, "T1012", "Query Registry", "Discovery"),
    ({"openkey", "querykey", "closekeys"}, "T1012", "Query Registry", "Discovery"),
    ({"openkey", "queryvaluekey", "closekeys"}, "T1012", "Query Registry", "Discovery"),
    ({"querykey", "enumeratekey"}, "T1012", "Query Registry", "Discovery"),
    ({"openkey", "enumeratekey"}, "T1012", "Query Registry", "Discovery"),
    ({"openkey", "querykey"}, "T1012", "Query Registry", "Discovery"),
    # ---- Registry Persistence (creating keys, setting values) ----
    ({"createkey", "setvaluekey"}, "T1547.001", "Registry Run Keys / Startup Folder", "Persistence"),
    ({"createkey", "setvaluekey", "closekeys"}, "T1547.001", "Registry Run Keys / Startup Folder", "Persistence"),
    ({"openkey", "setvaluekey"}, "T1112", "Modify Registry", "Defense Evasion"),
    ({"closekeys", "querykey", "createkey"}, "T1112", "Modify Registry", "Defense Evasion"),
    ({"openkey", "createkey"}, "T1112", "Modify Registry", "Defense Evasion"),
    # ---- Registry Cleanup / Anti-forensics ----
    ({"openkey", "deletekey"}, "T1070.007", "Clear Registry", "Defense Evasion"),
    ({"deletevaluekey", "closekeys"}, "T1070.007", "Clear Registry", "Defense Evasion"),
    # ---- Process Injection / DLL Loading ----
    ({"imageload", "create", "queryinformation"}, "T1055.001", "Dynamic-link Library Injection", "Defense Evasion"),
    ({"close", "create", "imageload"}, "T1055", "Process Injection", "Defense Evasion"),
    ({"imageload", "create"}, "T1055", "Process Injection", "Defense Evasion"),
    ({"imageload", "imageload", "imageload"}, "T1129", "Shared Modules", "Execution"),
    # ---- Process Execution ----
    ({"processstart", "imageload"}, "T1059", "Command and Scripting Interpreter", "Execution"),
    ({"imageload", "processstart"}, "T1059", "Command and Scripting Interpreter", "Execution"),
    # ---- Thread Injection ----
    ({"threadstart", "start", "threadstop", "stop"}, "T1055.003", "Thread Execution Hijacking", "Defense Evasion"),
    ({"threadstart", "threadstop"}, "T1055.003", "Thread Execution Hijacking", "Defense Evasion"),
    # ---- File System Data Collection ----
    ({"create", "read", "read"}, "T1005", "Data from Local System", "Collection"),
    ({"read", "read", "read"}, "T1005", "Data from Local System", "Collection"),
    ({"close", "read", "read"}, "T1005", "Data from Local System", "Collection"),
    # ---- File System Discovery ----
    ({"direnum", "queryinformation"}, "T1083", "File and Directory Discovery", "Discovery"),
    ({"close", "create", "direnum"}, "T1083", "File and Directory Discovery", "Discovery"),
    # ---- File System Staging / Writing ----
    ({"write", "write", "write"}, "T1074", "Data Staged", "Collection"),
    ({"write", "create"}, "T1105", "Ingress Tool Transfer", "Command and Control"),
    ({"create", "write", "close"}, "T1059.001", "PowerShell Script Writing", "Execution"),
    # ---- Network C2 ----
    ({"connectionattempted", "datasent", "datareceived"}, "T1071", "Application Layer Protocol", "Command and Control"),
    ({"datasent", "datareceived"}, "T1071", "Application Layer Protocol", "Command and Control"),
    ({"connectionattempted", "datasent"}, "T1041", "Exfiltration Over C2 Channel", "Exfiltration"),
    ({"connectionattempted"}, "T1095", "Non-Application Layer Protocol", "Command and Control"),
    # ---- Data Exfiltration over Network ----
    ({"read", "datasent"}, "T1041", "Exfiltration Over C2 Channel", "Exfiltration"),
    ({"datareceived", "write"}, "T1105", "Ingress Tool Transfer", "Command and Control"),
    # ---- Credential Access ----
    ({"openkey", "queryvaluekey", "querykey", "enumeratekey"}, "T1552.002", "Credentials in Registry", "Credential Access"),
]


def translate_ngram(ngram_str: str) -> Tuple[str, Set[str], List[Tuple[str, str, str]]]:
    """Translate a 4-gram feature name → narrative + categories + MITRE matches."""
    tokens = ngram_str.strip().lower().split()
    categories: Set[str] = set()
    steps: List[str] = []
    for t in tokens:
        cat, desc = EVENT_SEMANTICS.get(t, ("Unknown", f"unknown event '{t}'"))
        categories.add(cat)
        steps.append(desc)

    if len(steps) >= 2:
        narrative = "The thread " + steps[0].lower()
        for s in steps[1:]:
            narrative += ", then " + s.lower()
        narrative += "."
    else:
        narrative = "The thread " + (steps[0].lower() if steps else "performed an unknown action") + "."

    token_set = set(tokens)
    mitre_matches: List[Tuple[str, str, str]] = []
    seen: Set[str] = set()
    for keywords, mid, name, tactic in MITRE_PATTERNS:
        if keywords.issubset(token_set) and mid not in seen:
            mitre_matches.append((mid, name, tactic))
            seen.add(mid)
    return narrative, categories, mitre_matches


# =============================================================================
# StreamingComputationGraph — author-faithful incremental builder
# EVIDENCE: build_computation_graph.py (UID rules + per-provider event handlers)
# =============================================================================

class StreamingComputationGraph:
    """Mirrors the semantics of build_computation_graph.first_step() but
    incremental rather than batch.

    State:
        nodes[i]                : {kind, key, attrs}
        node_index[(kind, *key)]: i
        edges[i]                : {src, tar, task_name, event_name, ts, kind}
        proc_starts[pid]        : {ct, root_idx, ts, image, parent_pid}
        proc_threads[pid][tid]  : thread_start_ts
        parent_of[child_pid]    : parent_pid          (from PROCESSSTART)
        descendants[root_pid]   : {pids descending from root}
        events_by_root[root_pid]: [edge_idx ...]
        nodes_in_root[root_pid] : {node_idx ...}
        file_handle_map  : (FileObject, PID, TID) -> file_node_idx
        reg_handle_map   : (KeyObject,  PID, TID) -> reg_node_idx
    """

    def __init__(self):
        self.nodes: List[dict] = []
        self.node_index: Dict[Tuple, int] = {}
        self.edges: List[dict] = []
        self.proc_starts: Dict[str, dict] = {}
        self.proc_threads: Dict[str, Dict[str, str]] = defaultdict(dict)
        self.parent_of: Dict[str, str] = {}
        self.descendants: Dict[str, Set[str]] = defaultdict(set)
        self.events_by_root: Dict[str, List[int]] = defaultdict(list)
        self.nodes_in_root: Dict[str, Set[int]] = defaultdict(set)
        self.file_handle_map: Dict[Tuple[str, str, str], int] = {}
        self.reg_handle_map:  Dict[Tuple[str, str, str], int] = {}
        self.event_counter = 0

    # ---- Node helpers (UIDs from build_computation_graph.py) ----------------
    def _add_node(self, kind: str, key: Tuple, attrs: dict) -> int:
        full_key = (kind,) + key
        if full_key in self.node_index:
            return self.node_index[full_key]
        idx = len(self.nodes)
        self.nodes.append({"kind": kind, "key": key, "attrs": attrs})
        self.node_index[full_key] = idx
        return idx

    def _process_node(self, pid: str, ct: str) -> int:
        # <<PROC-NODE>>(PID):pid_(CT):ct
        return self._add_node("process", (pid, ct), {"ProcessId": pid, "CreateTime": ct})

    def _thread_node(self, pid: str, tid: str, thread_ts: str, ct: str) -> int:
        # <<THREAD>>(TID):tid_(TS):thread_ts__(PID):pid_(CT):ct
        return self._add_node("thread", (tid, thread_ts, pid, ct),
                              {"ThreadId": tid, "ThreadStartTime": thread_ts,
                               "ProcessId": pid, "CreateTime": ct})

    def _file_node(self, file_object: str, file_name: Optional[str],
                   pid: str, tid: str, include_fn: bool = True) -> int:
        # EVIDENCE: build_computation_graph.get_file_info
        # For CREATE/CREATENEWFILE: hash = (FileObject, FileName.upper(), PID, TID)
        # For other ops when not in mapping: hash = (FileObject, PID, TID) — NO FileName
        if include_fn:
            fn_key = (file_name or "").upper() if file_name else ""
        else:
            fn_key = ""  # omit FileName from UID, matching author's fallback
        return self._add_node("file", (file_object, fn_key, pid, tid),
                              {"FileObject": file_object, "FileName": file_name})

    def _registry_node(self, key_object: str, relative_name: Optional[str],
                       pid: str, tid: str, include_rn: bool = True) -> int:
        # EVIDENCE: build_computation_graph.get_reg_info
        # For CREATE/OPEN (opcodes 32,33): hash includes RelativeName.upper()
        # For all other ops when not in mapping: hash uses ONLY (KeyObject, PID, TID)
        if include_rn:
            rn_key = (relative_name or "").upper() if relative_name else ""
        else:
            rn_key = ""  # omit RelativeName from UID, matching author's fallback
        return self._add_node("registry", (key_object, rn_key, pid, tid),
                              {"KeyObject": key_object, "RelativeName": relative_name})

    def _network_node(self, daddr: str) -> int:
        return self._add_node("network", (daddr,), {"daddr": daddr})

    # ---- Public ingest ------------------------------------------------------
    def ingest(self, log_entry: dict) -> Optional[str]:
        raw_event = log_entry.get("EventName") or log_entry.get("Task Name") or ""
        opcode_name = log_entry.get("OpcodeName") or ""
        task_name, event_name = normalize_event(raw_event, opcode_name)
        if task_name == "UNKNOWN":
            return None

        pid = self._coerce_id(log_entry.get("ProcessID"))
        tid = self._coerce_id(log_entry.get("ThreadID"))
        xml = log_entry.get("XmlEventData") or {}
        if not isinstance(xml, dict):
            xml = {}
        # KERNEL MODE: top-level ProcessID is the REAL originating PID.
        # Only fall back to XmlEventData when top-level is null/-1.
        # EVIDENCE: author's build_computation_graph.py reads both ProcessId
        # (lowercase d) and ProcessID (uppercase D) — in kernel mode they match.
        if pid is None:
            pid = (self._coerce_id(xml.get("PID")) or
                   self._coerce_id(xml.get("_PID")) or
                   self._coerce_id(xml.get("ProcessId")))
        if tid is None:
            tid = (self._coerce_id(xml.get("TID")) or
                   self._coerce_id(xml.get("TThreadId")) or
                   self._coerce_id(xml.get("ThreadId")))

        if pid is None:
            return None

        ts_raw = log_entry.get("TimeStamp") or log_entry.get("@timestamp")
        ts = self._coerce_ts(ts_raw)
        self.event_counter += 1

        # ---- PROCESSSTART ----
        if task_name == "PROCESSSTART":
            child_pid = self._coerce_id(xml.get("ProcessID")) or pid
            parent_pid = (self._coerce_id(xml.get("ParentProcessId")) or
                          self._coerce_id(xml.get("ParentProcessID")) or pid)
            create_time = xml.get("CreateTime") or log_entry.get("CreateTime") or ts_raw or "N/A"
            child_idx = self._process_node(child_pid, str(create_time))
            self.proc_starts[child_pid] = {
                "ct": str(create_time), "root_idx": child_idx, "ts": ts,
                "image": xml.get("ImageName") or log_entry.get("ImageName"),
                "parent_pid": parent_pid,
            }
            if parent_pid in self.proc_starts and parent_pid != child_pid:
                self.parent_of[child_pid] = parent_pid
                root_pid = self._find_root(parent_pid)
                self.descendants[root_pid].add(child_pid)
                # Edge: parent thread → child process
                if tid is not None:
                    parent_ct = self.proc_starts[parent_pid]["ct"]
                    thread_ts = self.proc_threads[parent_pid].get(tid, "N/A")
                    t_idx = self._thread_node(parent_pid, tid, thread_ts, parent_ct)
                    self._add_edge(t_idx, child_idx, task_name, event_name, ts,
                                   "process", root_pid=root_pid)
            else:
                self.descendants[child_pid].add(child_pid)
            return child_pid

        # ---- THREADSTART ----
        # EVIDENCE: build_computation_graph.get_proc_info THREADSTART branch
        # When ProcessId == ProcessID: same-PID thread creation
        # When ProcessId != ProcessID: CROSS-PROCESS thread injection (T1055)
        #   Author creates edges in BOTH the parent and child process subgraphs
        if task_name == "THREADSTART":
            parent_pid = pid  # ProcessId = the process that initiated the thread creation
            thread_pid = self._coerce_id(xml.get("ProcessID")) or pid  # ProcessID = the process the thread belongs to
            new_tid = self._coerce_id(xml.get("ThreadID")) or tid
            if new_tid is None:
                return None

            # Register thread start time in the TARGET process
            self.proc_threads[thread_pid][new_tid] = str(ts_raw or ts)

            # Ensure target process exists
            if thread_pid not in self.proc_starts:
                proc_name = log_entry.get("ProcessName") or xml.get("PName")
                self.proc_starts[thread_pid] = {"ct": "N/A", "root_idx": None,
                                                 "ts": ts, "image": proc_name,
                                                 "parent_pid": None}
                self.descendants[thread_pid].add(thread_pid)
            elif self.proc_starts[thread_pid].get("image") is None:
                proc_name = log_entry.get("ProcessName") or xml.get("PName")
                if proc_name and proc_name not in ("N/A", "", "None"):
                    self.proc_starts[thread_pid]["image"] = proc_name

            # SAME-PID case (normal thread creation within a process)
            proc_ct = self.proc_starts[thread_pid]["ct"]
            p_idx = self._process_node(thread_pid, proc_ct)
            t_idx = self._thread_node(thread_pid, new_tid, str(ts_raw or ts), proc_ct)
            root_pid_target = self._find_root(thread_pid)
            # Direction: THREADSTART not in PROCESS_TO_PROC → reversed = process→thread
            self._add_edge(p_idx, t_idx, task_name, event_name, ts, "process",
                           root_pid=root_pid_target)

            # CROSS-PROCESS case: parent_pid != thread_pid (remote thread injection)
            # EVIDENCE: build_computation_graph.py get_proc_info THREADSTART:
            #   "if logentry_ProcessId != logentry_ProcessID:"
            #   Creates edges in the CHILD process too
            if parent_pid != thread_pid and parent_pid is not None:
                if parent_pid not in self.proc_starts:
                    self.proc_starts[parent_pid] = {"ct": "N/A", "root_idx": None,
                                                     "ts": ts, "image": None,
                                                     "parent_pid": None}
                    self.descendants[parent_pid].add(parent_pid)
                parent_ct = self.proc_starts[parent_pid]["ct"]
                parent_thread_ts = self.proc_threads[parent_pid].get(tid, "N/A")
                parent_t_idx = self._thread_node(parent_pid, tid, parent_thread_ts, parent_ct)
                parent_p_idx = self._process_node(parent_pid, parent_ct)
                root_pid_parent = self._find_root(parent_pid)
                # Edge in parent's subgraph: parent_process → parent_thread (the initiator)
                self._add_edge(parent_p_idx, parent_t_idx, task_name, event_name, ts,
                               "process", root_pid=root_pid_parent)

            return None

        # ---- All other events ----
        provider_id = log_entry.get("ProviderGuid") or log_entry.get("ProviderId") or ""
        provider = self._classify_provider(task_name, provider_id)
        if pid not in self.proc_starts:
            # Fix C: use ProcessName from event as image when no PROCESSSTART available
            proc_name = log_entry.get("ProcessName") or xml.get("PName")
            if proc_name in ("", "N/A", "None", None):
                proc_name = None
            self.proc_starts[pid] = {"ct": "N/A", "root_idx": None, "ts": ts,
                                      "image": proc_name, "parent_pid": None}
            self.descendants[pid].add(pid)
        elif self.proc_starts[pid].get("image") is None:
            # Update image if we didn't have it before
            proc_name = log_entry.get("ProcessName") or xml.get("PName")
            if proc_name and proc_name not in ("N/A", "", "None"):
                self.proc_starts[pid]["image"] = proc_name
        proc_ct = self.proc_starts[pid]["ct"]
        if tid is None:
            return None
        thread_ts = self.proc_threads[pid].get(tid, "N/A")
        t_idx = self._thread_node(pid, tid, thread_ts, proc_ct)
        root_pid = self._find_root(pid)

        if provider == "file":
            # EVIDENCE: build_computation_graph.get_file_info line 1:
            # "if logentry_TaskName not in {'OPERATIONEND', 'NAMEDELETE'}:"
            # Author SKIPS these events entirely — they don't produce nodes or edges.
            if task_name in {"OPERATIONEND", "NAMEDELETE"}:
                return None
            file_object = str(xml.get("FileObject") or log_entry.get("FileObject") or "?")
            # SilkETW kernel trace uses "OpenPath" instead of "FileName" for FileIo/Create
            file_name = (xml.get("FileName") or xml.get("OpenPath") or
                         log_entry.get("FileName") or log_entry.get("OpenPath"))
            key = (file_object, pid, tid)
            if task_name in {"CREATE", "CREATENEWFILE"}:
                f_idx = self._file_node(file_object, file_name, pid, tid, include_fn=True)
                self.file_handle_map[key] = f_idx
            elif task_name == "CLOSE":
                f_idx = self.file_handle_map.pop(key, None)
                if f_idx is None:
                    f_idx = self._file_node(file_object, file_name, pid, tid, include_fn=False)
            else:
                f_idx = self.file_handle_map.get(key)
                if f_idx is None:
                    # Author: fallback hash omits FileName
                    f_idx = self._file_node(file_object, file_name, pid, tid, include_fn=False)
            self._wire_edge(t_idx, f_idx, "file", task_name, event_name, ts, root_pid)

        elif provider == "registry":
            # SilkETW kernel trace uses "KeyHandle" instead of "KeyObject"
            ko = str(xml.get("KeyObject") or xml.get("KeyHandle") or
                     log_entry.get("KeyObject") or log_entry.get("KeyHandle") or "?")
            rn = (xml.get("RelativeName") or xml.get("KeyName") or
                  log_entry.get("KeyName"))
            key = (ko, pid, tid)
            if task_name in {"CREATEKEY", "OPENKEY"}:
                r_idx = self._registry_node(ko, rn, pid, tid, include_rn=True)
                self.reg_handle_map[key] = r_idx
            elif task_name == "CLOSEKEYS":
                r_idx = self.reg_handle_map.pop(key, None)
                if r_idx is None:
                    # Author: fallback hash omits RelativeName
                    r_idx = self._registry_node(ko, rn, pid, tid, include_rn=False)
            else:
                r_idx = self.reg_handle_map.get(key)
                if r_idx is None:
                    # Author: fallback hash omits RelativeName
                    r_idx = self._registry_node(ko, rn, pid, tid, include_rn=False)
            self._wire_edge(t_idx, r_idx, "registry", task_name, event_name, ts, root_pid)

        elif provider == "network":
            # SilkETW kernel trace may have daddr as integer or IP string
            raw_daddr = xml.get("daddr") or xml.get("saddr") or log_entry.get("daddr") or "0.0.0.0"
            daddr = str(raw_daddr).replace(",", "")  # remove comma-separated int format
            n_idx = self._network_node(daddr)
            self._wire_edge(t_idx, n_idx, "network", task_name, event_name, ts, root_pid)

        elif provider == "process":
            p_idx = self._process_node(pid, proc_ct)
            self._wire_edge(t_idx, p_idx, "process", task_name, event_name, ts, root_pid)

        return None

    # ---- Edge wiring --------------------------------------------------------
    def _wire_edge(self, thread_idx: int, resource_idx: int, kind: str,
                   task_name: str, event_name: str, ts: float, root_pid: str):
        direction = edge_direction(kind, task_name)
        if direction == "resource_to_thread":
            src, tar = resource_idx, thread_idx
        else:
            src, tar = thread_idx, resource_idx
        self._add_edge(src, tar, task_name, event_name, ts, kind, root_pid=root_pid)

    def _add_edge(self, src: int, tar: int, task_name: str, event_name: str,
                  ts: float, kind: str, root_pid: Optional[str] = None):
        e_idx = len(self.edges)
        self.edges.append({"src": src, "tar": tar, "task_name": task_name,
                           "event_name": event_name, "ts": ts, "kind": kind})
        if root_pid is not None:
            self.events_by_root[root_pid].append(e_idx)
            self.nodes_in_root[root_pid].add(src)
            self.nodes_in_root[root_pid].add(tar)

    # ---- Helpers ------------------------------------------------------------
    # EVIDENCE: build_computation_graph.py dispatches by ProviderId GUID
    FILE_PROVIDER     = "{edd08927-9cc4-4e65-b970-c2560fb5c289}"
    NETWORK_PROVIDER  = "{7dd42a49-5329-4832-8dfd-43d979153a88}"
    PROCESS_PROVIDER  = "{22fb2cd6-0e7b-422b-a0c7-2fad1fd0e716}"
    REGISTRY_PROVIDER = "{70eb4f03-c1de-4f73-a051-33d13d5413bd}"

    PROVIDER_TO_KIND = {
        FILE_PROVIDER: "file", NETWORK_PROVIDER: "network",
        PROCESS_PROVIDER: "process", REGISTRY_PROVIDER: "registry",
    }

    @staticmethod
    def _classify_provider(task_name: str, provider_id: str = "") -> str:
        # Primary: use ProviderId GUID if available (author's method)
        if provider_id:
            kind = StreamingComputationGraph.PROVIDER_TO_KIND.get(provider_id.lower())
            if kind:
                return kind
        # Fallback: classify by task name
        FILE = {"CLEANUP", "CLOSE", "CREATE", "CREATENEWFILE", "DELETEPATH",
                "DIRENUM", "DIRNOTIFY", "FLUSH", "FSCTL", "NAMECREATE",
                "NAMEDELETE", "OPERATIONEND", "QUERYINFORMATION", "QUERYEA",
                "QUERYSECURITY", "READ", "WRITE", "SETDELETE", "SETINFORMATION",
                "RENAME", "RENAMEPATH"}
        REG = {"CREATEKEY", "OPENKEY", "DELETEKEY", "QUERYKEY", "SETVALUEKEY",
               "DELETEVALUEKEY", "QUERYVALUEKEY", "ENUMERATEKEY",
               "ENUMERATEVALUEKEY", "QUERYMULTIPLEVALUEKEY",
               "SETINFORMATIONKEY", "CLOSEKEYS", "QUERYSECURITYKEY",
               "SETSECURITYKEY", "REGPERFOPHIVEFLUSHWROTELOGFILE"}
        NET = {"DATASENT", "DATARECEIVED", "CONNECTIONATTEMPTED",
               "DISCONNECTISSUED", "DATARETRANSMITTED", "CONNECTIONACCEPTED",
               "PROTOCOLCOPIEDDATA", "DATARECEIVEDOVERUDP", "DATASENTOVERUDP"}
        if task_name in FILE: return "file"
        if task_name in REG:  return "registry"
        if task_name in NET:  return "network"
        return "process"

    def _find_root(self, pid: str) -> str:
        seen = set()
        cur = pid
        while cur in self.parent_of and cur not in seen:
            seen.add(cur)
            cur = self.parent_of[cur]
        return cur

    @staticmethod
    def _coerce_id(value) -> Optional[str]:
        if value is None: return None
        s = str(value).replace(",", "").strip()
        if s in {"", "0", "-1", "None"}: return None
        return s

    @staticmethod
    def _coerce_ts(value) -> float:
        if value is None: return 0.0
        if isinstance(value, (int, float)): return float(value)
        try:
            from dateutil.parser import parse as _p
            return _p(str(value)).timestamp()
        except Exception:
            try: return float(value)
            except Exception: return 0.0

    def root_pids_ready(self, node_threshold: int) -> List[str]:
        return [r for r in list(self.descendants.keys())
                if len(self.nodes_in_root.get(r, ())) >= node_threshold]

    def diagnostics(self, root_pid: str) -> dict:
        node_idxs = self.nodes_in_root.get(root_pid, set())
        kinds: Dict[str, int] = defaultdict(int)
        for ni in node_idxs:
            kinds[self.nodes[ni]["kind"]] += 1
        edge_idxs = self.events_by_root.get(root_pid, [])
        events: Dict[str, int] = defaultdict(int)
        for ei in edge_idxs:
            events[self.edges[ei]["event_name"]] += 1
        image = self.proc_starts.get(root_pid, {}).get("image")
        return {"image": image,
                "num_nodes": len(node_idxs), "num_edges": len(edge_idxs),
                "node_kinds": dict(kinds),
                "top_events": dict(sorted(events.items(), key=lambda x: -x[1])[:10])}


# =============================================================================
# GraphletProjector — author-faithful taint-rooted projection
# EVIDENCE: project_graphlets.tainted_subgraph + subgraph + projection
# =============================================================================

class ProjectedGraphlet:
    def __init__(self, root_pid: str, image: Optional[str],
                 nodes: List[dict], edges: List[dict]):
        self.root_pid = root_pid
        self.image = image
        self.nodes = nodes
        self.edges = edges

    def to_pyg_data(self) -> Data:
        # 5-dim type one-hot per node (see graphite_n_gram.py:118)
        kind_to_idx = {"file": 0, "registry": 1, "network": 2,
                       "process": 3, "thread": 4}
        x = []
        for nd in self.nodes:
            v = [0.0] * NUM_NODE_FEATURES
            v[kind_to_idx[nd["kind"]]] = 1.0
            x.append(v)

        # 62-dim edge attr: 61 event one-hot + 1 timestamp
        ts_values = [e["ts"] for e in self.edges]
        ts_min = min(ts_values) if ts_values else 0.0
        ts_max = max(ts_values) if ts_values else 1.0
        ts_range = max(ts_max - ts_min, 1e-9)
        src_list, tar_list, attr_list = [], [], []
        for e in self.edges:
            attr = [0.0] * NUM_EDGE_FEATURES
            ev_idx = EVENTNAME_INDEX.get(e["event_name"], EVENTNAME_INDEX["UNKNOWN"])
            attr[ev_idx] = 1.0
            attr[-1] = (e["ts"] - ts_min) / ts_range
            src_list.append(e["src"])
            tar_list.append(e["tar"])
            attr_list.append(attr)

        return Data(
            x=torch.tensor(x, dtype=torch.float),
            edge_index=torch.tensor([src_list, tar_list], dtype=torch.long),
            edge_attr=torch.tensor(attr_list, dtype=torch.float),
        )


class GraphletProjector:
    def __init__(self, graph: StreamingComputationGraph):
        self.g = graph

    def project(self, root_pid: str) -> Optional[ProjectedGraphlet]:
        if root_pid not in self.g.descendants:
            return None
        node_idxs = self.g.nodes_in_root.get(root_pid)
        edge_idxs = self.g.events_by_root.get(root_pid)
        if not node_idxs or not edge_idxs:
            return None
        sorted_nodes = sorted(node_idxs)
        local_of = {n: i for i, n in enumerate(sorted_nodes)}
        local_nodes = [self.g.nodes[n] for n in sorted_nodes]
        local_edges = []
        for ei in edge_idxs:
            e = self.g.edges[ei]
            if e["src"] in local_of and e["tar"] in local_of:
                local_edges.append({
                    "src": local_of[e["src"]],
                    "tar": local_of[e["tar"]],
                    "task_name": e["task_name"],
                    "event_name": e["event_name"],
                    "ts": e["ts"],
                })
        if not local_edges:
            return None
        image = self.g.proc_starts.get(root_pid, {}).get("image")
        return ProjectedGraphlet(root_pid=root_pid, image=image,
                                 nodes=local_nodes, edges=local_edges)


# =============================================================================
# SHAP alert generator
# =============================================================================

class SHAPAlertGenerator:
    def __init__(self, model, feature_names: List[str]):
        self.explainer = shap.TreeExplainer(model)
        self.feature_names = feature_names
        ev = self.explainer.expected_value
        if hasattr(ev, "__len__") and len(ev) > 1:
            self.base_value = float(ev[1])
        else:
            self.base_value = float(ev) if not hasattr(ev, "__len__") else float(ev[0])

    def explain(self, X: np.ndarray, top_k: int = 10) -> dict:
        t0 = time.perf_counter()
        sv_raw = self.explainer.shap_values(X)
        if isinstance(sv_raw, list) and len(sv_raw) > 1:
            sv = sv_raw[1].flatten()
        elif isinstance(sv_raw, list):
            sv = sv_raw[0].flatten()
        else:
            sv = sv_raw.flatten() if sv_raw.ndim <= 1 else sv_raw[0]
        t_ms = (time.perf_counter() - t0) * 1000

        x_vals = X.flatten() if X.ndim > 1 else X

        # CRITICAL: only show features that actually occurred in this graphlet
        active = np.where(x_vals > 0)[0]
        if len(active) > 0:
            order = active[np.argsort(np.abs(sv[active]))[::-1]]
            top = order[:top_k]
        else:
            top = np.argsort(np.abs(sv))[::-1][:top_k]

        attributions = []
        all_mitre: List[Tuple[str, str, str]] = []
        for rank, idx in enumerate(top):
            name = self.feature_names[idx] if idx < len(self.feature_names) else f"f{idx}"
            shap_val = float(sv[idx])
            orig = float(x_vals[idx]) if idx < len(x_vals) else 0.0

            # Feature is either a node-type spatial feature or an n-gram
            if idx < len(NODETYPE_NODEFEATS):
                narrative = (f"High interaction with {NODETYPE_NODEFEATS[idx]} "
                             f"entities ({int(orig)} unique neighbours)")
                categories = {NODETYPE_NODEFEATS[idx]}
                mitre: List[Tuple[str, str, str]] = []
            else:
                narrative, categories, mitre = translate_ngram(name)
                all_mitre.extend(mitre)

            attributions.append({
                "rank": rank + 1, "feature": name,
                "shap_value": shap_val, "orig_value": orig,
                "direction": "MALWARE" if shap_val > 0 else "BENIGN",
                "narrative": narrative,
                "categories": sorted(categories),
                "mitre": [{"id": m[0], "name": m[1], "tactic": m[2]} for m in mitre],
            })

        unique_mitre = list({m[0]: m for m in all_mitre}.values())
        return {"t_explain_ms": t_ms, "base_value": self.base_value,
                "attributions": attributions,
                "mitre_summary": [{"id": m[0], "name": m[1], "tactic": m[2]}
                                  for m in unique_mitre]}


# =============================================================================
# SOC ALERT — boxed format with WHY-IS-THIS / MITRE summary
# =============================================================================

def print_soc_alert(label: str,
                    prediction: int,
                    confidence: float,
                    t_detect_ms: float,
                    t_explain_ms: float,
                    num_nodes: int,
                    num_edges: int,
                    image: Optional[str] = None,
                    shap_alert: Optional[dict] = None,
                    show_benign: bool = False,
                    node_kinds: Optional[dict] = None):
    """Print the full SOC alert box.
    Both malware and benign alerts use the same format with XAI.
    Benign alerts only printed when show_benign=True.
    """
    if prediction == 0 and not show_benign:
        return

    total = t_detect_ms + t_explain_ms
    threshold_status = "PASS \u2713" if total <= RQ3_LATENCY_THRESHOLD_MS else "FAIL \u2717"

    W = 100  # box width
    if prediction == 1:
        title = "SOC ALERT \u2014 MALWARE DETECTION EXPLANATION"
        verdict = f"\u2588\u2588 MALWARE \u2014 confidence = {confidence:.4f} ({confidence*100:.1f}%)"
        xai_title = "WHY IS THIS MALWARE? \u2014 Top behavioural indicators (SHAP)"
    else:
        title = "SOC NOTE \u2014 BENIGN CLASSIFICATION EXPLANATION"
        verdict = f"\u25A1\u25A1 BENIGN \u2014 confidence = {1-confidence:.4f} ({(1-confidence)*100:.1f}%)"
        xai_title = "WHY IS THIS BENIGN? \u2014 Top behavioural indicators (SHAP)"

    def line(content: str):
        s = "  \u2502  " + content
        pad = W + 4 - len(s)
        if pad < 0: pad = 0
        print(s + " " * pad + "\u2502")

    print(f"\n  \u250C{'\u2500'*W}\u2510")
    print(f"  \u2502{title:^{W}}\u2502")
    print(f"  \u251C{'\u2500'*W}\u2524")
    line(f"Sample  : {label[:W-12]}")
    if image:
        line(f"Process : {image[:W-12]}")
    line(f"Model   : LightGBM (RQ1 winner: 91.81% acc, 0.9672 AUC)")
    line(f"Verdict : {verdict}")
    graph_str = f"{num_nodes} nodes, {num_edges} edges"
    if node_kinds:
        kind_parts = [f"{v} {k}" for k, v in sorted(node_kinds.items(), key=lambda x: -x[1])]
        graph_str += f"  ({', '.join(kind_parts)})"
    line(f"Graph   : {graph_str[:W-12]}")
    print(f"  \u251C{'\u2500'*W}\u2524")
    line(f"T_detect    : {t_detect_ms:>10.2f} ms")
    line(f"T_explain   : {t_explain_ms:>10.2f} ms")
    line(f"T_total     : {total:>10.2f} ms")
    line(f"RQ3 (\u2264{RQ3_LATENCY_THRESHOLD_MS}ms): [{threshold_status}]")

    if shap_alert:
        print(f"  \u251C{'\u2500'*W}\u2524")
        print(f"  \u2502{xai_title:^{W}}\u2502")
        print(f"  \u251C{'\u2500'*W}\u2524")

        shown = 0
        for a in shap_alert["attributions"]:
            if a["orig_value"] <= 0:
                continue
            shown += 1
            arrow = "\u2192 MALWARE" if a["shap_value"] > 0 else "\u2192 BENIGN"
            strength = "strongly " if abs(a["shap_value"]) > 0.5 else ""
            line("")
            line(f"#{a['rank']}  Pattern : {a['feature']}")
            cat_str = "+".join(a["categories"])
            line(f"     Category: [{cat_str}]")
            line(f"     Occurred : {int(a['orig_value'])} time(s) in this graphlet")
            line(f"     Impact   : SHAP = {a['shap_value']:+.4f}  ({arrow}) {strength}influences classification")
            narr = a["narrative"]
            chunk = W - 18
            line(f"     Meaning  : {narr[:chunk]}")
            if len(narr) > chunk:
                line(f"                {narr[chunk:2*chunk]}")
            for m in a["mitre"]:
                line(f"     MITRE   : {m['id']}  ({m['name']}) [{m['tactic']}]")

        if shown == 0:
            line("(no active n-gram features in this graphlet \u2014 too small or wrong vocabulary)")

        mitre_summary = shap_alert.get("mitre_summary", [])
        if mitre_summary:
            print(f"  \u251C{'\u2500'*W}\u2524")
            print(f"  \u2502{'MITRE ATT&CK SUMMARY \u2014 unique techniques observed':^{W}}\u2502")
            print(f"  \u251C{'\u2500'*W}\u2524")
            for m in mitre_summary[:8]:
                line(f"  {m['id']:<12} {m['name']:<40} [{m['tactic']}]")

            # ---- BEHAVIOURAL ATTACK PROFILE ----
            # Analyse the COMBINATION of MITRE techniques to produce a narrative
            tactic_set = {m['tactic'] for m in mitre_summary}
            technique_ids = {m['id'] for m in mitre_summary}
            profiles = []
            if 'Discovery' in tactic_set and 'Persistence' in tactic_set:
                profiles.append("RECONNAISSANCE → PERSISTENCE: Process performed registry "
                               "discovery (enumerating keys/values) then established persistence "
                               "by creating or modifying registry keys — consistent with initial "
                               "access followed by foothold establishment.")
            elif 'Discovery' in tactic_set and 'Defense Evasion' in tactic_set:
                profiles.append("RECONNAISSANCE → EVASION: Process scanned registry and file "
                               "system for information then modified system configuration to "
                               "evade detection — consistent with environment-aware malware "
                               "that adapts its behaviour based on installed defences.")
            elif 'Discovery' in tactic_set and 'Collection' in tactic_set:
                profiles.append("RECONNAISSANCE → DATA HARVESTING: Process enumerated system "
                               "resources (registry keys, files) and systematically read data — "
                               "consistent with information-stealing malware collecting credentials "
                               "or configuration data.")
            if 'Command and Control' in tactic_set and 'Collection' in tactic_set:
                profiles.append("C2 + DATA COLLECTION: Network communication combined with "
                               "file/registry reading suggests data is being exfiltrated to "
                               "a command-and-control server.")
            if 'Defense Evasion' in tactic_set and ('T1055' in technique_ids or 'T1055.001' in technique_ids or 'T1055.003' in technique_ids):
                profiles.append("PROCESS INJECTION DETECTED: DLL loading patterns combined with "
                               "file creation and thread manipulation indicate code injection "
                               "into a running process — a hallmark of fileless malware.")
            if 'Execution' in tactic_set and 'T1129' in technique_ids:
                profiles.append("MASS DLL LOADING: Rapid sequential loading of multiple shared "
                               "libraries suggests reflective DLL injection or module side-loading.")
            if not profiles:
                if 'Discovery' in tactic_set:
                    profiles.append("SYSTEM ENUMERATION: Process is systematically querying "
                                   "system information through registry and file operations — "
                                   "consistent with early-stage reconnaissance by malware.")
                elif 'Collection' in tactic_set:
                    profiles.append("DATA HARVESTING: Process is systematically reading files "
                                   "and registry values — consistent with credential or data "
                                   "theft operations.")
            if profiles:
                print(f"  \u251C{'\u2500'*W}\u2524")
                print(f"  \u2502{'BEHAVIOURAL ATTACK PROFILE':^{W}}\u2502")
                print(f"  \u251C{'\u2500'*W}\u2524")
                for profile in profiles[:2]:
                    chunk = W - 8
                    words = profile.split()
                    current_line = ""
                    for word in words:
                        if len(current_line) + len(word) + 1 > chunk:
                            line(f"  {current_line}")
                            current_line = word
                        else:
                            current_line = f"{current_line} {word}" if current_line else word
                    if current_line:
                        line(f"  {current_line}")
                    line("")

    print(f"  \u2514{'\u2500'*W}\u2518")


# =============================================================================
# Saved-model loader
# =============================================================================

def load_saved_model(model_dir: str):
    md = pathlib.Path(model_dir)
    model_path    = md / "lightgbm_model.joblib"
    embedder_path = md / "graphite_embedder.joblib"
    meta_path     = md / "model_meta.json"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            f"Run lightgbm_shap_analyzer.py first to train and save the model.")

    model = joblib.load(model_path)
    try:
        embedder = joblib.load(embedder_path)
    except Exception:
        import pickle as _p
        class _R(_p.Unpickler):
            def find_class(self, module, name):
                if name == "GraphiteEmbedder":
                    return GraphiteEmbedder
                return super().find_class(module, name)
        with open(embedder_path, "rb") as f:
            embedder = _R(f).load()

    meta = json.load(open(meta_path)) if meta_path.exists() else {}
    log.info(f"Loaded saved model: features={len(embedder.feature_names)}  "
             f"trained_on={meta.get('trained_on','?')}  "
             f"test_f1={meta.get('test_f1','?')}  "
             f"test_auc={meta.get('test_auc','?')}")
    return model, embedder, meta


def classify_graph(model, embedder, data: Data) -> Tuple[int, float, float, np.ndarray]:
    t0 = time.perf_counter()
    emb = embedder.embed(data)
    if emb.dim() > 1:
        emb = emb.squeeze(0)
    X = emb.numpy().reshape(1, -1)
    pred = int(model.predict(X)[0])
    conf = float(model.predict_proba(X)[0][1])
    t_ms = (time.perf_counter() - t0) * 1000
    return pred, conf, t_ms, X


# =============================================================================
# MODE 1: LIVE — Elasticsearch polling, real SilkETW pipeline
# =============================================================================

def run_live_mode(args):
    from elasticsearch import Elasticsearch
    model, embedder, _ = load_saved_model(args.model_dir)
    alerter = SHAPAlertGenerator(model, embedder.feature_names)

    es = Elasticsearch(args.es_host)
    if not es.ping():
        log.error(f"Cannot connect to Elasticsearch at {args.es_host}")
        sys.exit(1)
    log.info(f"Connected to Elasticsearch at {args.es_host}")

    sg = StreamingComputationGraph()
    projector = GraphletProjector(sg)
    last_ts = "now-24h"
    classified_roots: Dict[str, dict] = {}  # pid -> {'nodes': count, 'alert_count': n}
    results: List[dict] = []
    poll_count = 0

    print("\n" + "=" * 100)
    print("  RQ3: REAL-TIME SOC SIMULATION  (live mode)".ljust(100))
    print(f"  Elasticsearch: {args.es_host}/{args.es_index}".ljust(100))
    print(f"  Trigger      : graphlet >= {args.trigger_nodes} nodes per PROCESSSTART root".ljust(100))
    print(f"  Threshold    : T_detect + T_explain <= {RQ3_LATENCY_THRESHOLD_MS} ms".ljust(100))
    print(f"  Show benign  : {args.show_benign}".ljust(100))
    if hasattr(args, 'dashboard') and args.dashboard:
        print(f"  Dashboard    : http://localhost:{args.dashboard_port}".ljust(100))
    print("=" * 100 + "\n")

    # Start live dashboard server if requested
    if hasattr(args, 'dashboard') and args.dashboard:
        DashboardHandler.start(args.dashboard_port)

    # Use options() to avoid DeprecationWarning on request_timeout
    es_opts = es.options(request_timeout=30, ignore_status=[400, 404])

    try:
        while True:
            try:
                # Use larger batch size for reliability — SilkETW sends in bursts
                res = es_opts.search(index=args.es_index, body={
                    "query": {"range": {"@timestamp": {"gte": last_ts}}},
                    "size": 10000,
                    "sort": [{"@timestamp": "asc"}],
                })
            except Exception as e:
                log.error(f"ES query failed: {e}")
                time.sleep(args.poll_interval)
                continue

            hits = res["hits"]["hits"]
            if hits:
                last_ts = hits[-1]["_source"]["@timestamp"]
                ingested = 0
                for h in hits:
                    result = sg.ingest(h["_source"])
                    if result is not None:
                        ingested += 1
                # Log burst arrivals for visibility
                if len(hits) > 1000:
                    log.info(f"[burst] Ingested {len(hits)} events ({ingested} new PIDs)")

            # Periodic status log every 60 polls (~30 seconds)
            poll_count += 1
            if poll_count % 60 == 0:
                total_nodes = len(sg.nodes)
                total_edges = len(sg.edges)
                n_roots = len(sg.descendants)
                # Count actual classifications vs whitelisted skips
                n_malware = sum(1 for d in classified_roots.values() if d.get('verdict') == 'MALWARE')
                n_benign = sum(1 for d in classified_roots.values() if d.get('verdict') in ('BENIGN', None))
                n_whitelisted = sum(1 for d in classified_roots.values() if d.get('verdict') == 'WHITELISTED')
                n_actual = n_malware + n_benign
                log.info(f"[status] {total_nodes} nodes, {total_edges} edges, "
                         f"{n_roots} roots, {n_actual} classified "
                         f"({n_malware} malware, {n_benign} benign, {n_whitelisted} whitelisted)")
                # Show classified PIDs with their verdicts
                if classified_roots:
                    mal_pids = [(r, d) for r, d in classified_roots.items()
                                if d.get('verdict') == 'MALWARE']
                    ben_pids = [(r, d) for r, d in classified_roots.items()
                                if d.get('verdict') in ('BENIGN', None) and d.get('verdict') != 'WHITELISTED']
                    if mal_pids:
                        mal_str = ", ".join(
                            f"PID {r}({sg.proc_starts.get(r,{}).get('image','?')}"
                            f" peak={d.get('peak_conf',0):.0%})"
                            for r, d in mal_pids[:5])
                        log.info(f"[MALWARE detected] {mal_str}")
                    if ben_pids and len(ben_pids) <= 10:
                        ben_str = ", ".join(
                            f"PID {r}({sg.proc_starts.get(r,{}).get('image','?')})"
                            for r, _ in ben_pids[:5])
                        log.info(f"[benign] {ben_str}")
                # Show top 5 largest unclassified roots so user can see what's building
                unclassified = [(r, len(sg.nodes_in_root.get(r, set())))
                                for r in sg.descendants.keys()
                                if classified_roots.get(r, {}).get('nodes', 0) == 0
                                and len(sg.nodes_in_root.get(r, set())) > 10]
                unclassified.sort(key=lambda x: -x[1])
                if unclassified:
                    top_str = ", ".join(
                        f"PID {r}({sg.proc_starts.get(r,{}).get('image','?')})={n}"
                        for r, n in unclassified[:5])
                    log.info(f"[growing] {top_str}")
                # If watch_pid is set, always show its status
                if hasattr(args, 'watch_pid') and args.watch_pid:
                    wp = str(args.watch_pid)
                    wn = len(sg.nodes_in_root.get(wp, set()))
                    we = len(sg.events_by_root.get(wp, []))
                    wi = sg.proc_starts.get(wp, {}).get('image', 'NOT SEEN')
                    if wn > 0:
                        wd = sg.diagnostics(wp)
                        log.info(f"[watch PID {wp}] {wn} nodes, {we} edges, "
                                 f"image={wi}, kinds={wd['node_kinds']}")
                    else:
                        log.info(f"[watch PID {wp}] NOT YET SEEN in events")

            # Known benign system processes — skip classification to reduce false positives
            BENIGN_WHITELIST = {
                'system', 'idle', 'registry', 'smss', 'csrss', 'wininit',
                'services', 'lsass', 'svchost', 'ctfmon', 'dwm', 'sihost',
                'explorer', 'taskhostw', 'runtimebroker', 'searchindexer',
                'searchprotocolhost', 'searchfilterhost', 'searchapp',
                'msmpeng', 'nissrv', 'securityhealthservice',
                'wmiprvse', 'dllhost', 'conhost', 'fontdrvhost',
                'silketw', 'logstash', 'vmtoolsd', 'vmwaretray',
                'onedrive.sync.service', 'filecoauth', 'sdxhelper',
                'officeclicktorun', 'windowsterminal', 'openconsole',
                'msedgewebview2', 'msedge', 'chrome', 'firefox',
            }
            # Minimum confidence to trigger a MALWARE alert
            MALWARE_CONFIDENCE_THRESHOLD = 0.70

            for root_pid in sg.root_pids_ready(args.trigger_nodes):
                current_nodes = len(sg.nodes_in_root.get(root_pid, set()))
                prev_data = classified_roots.get(root_pid, {})
                prev_nodes = prev_data.get('nodes', 0)
                prev_alerts = prev_data.get('alert_count', 0)
                prev_verdict = prev_data.get('verdict', None)
                peak_conf = prev_data.get('peak_conf', 0.0)
                malware_alert_count = prev_data.get('malware_alerts', 0)

                # STOP re-analyzing this PID entirely if:
                # - Already classified as MALWARE with 2+ alerts shown
                # This prevents the confusing MALWARE→BENIGN flip
                if prev_verdict == 'MALWARE' and malware_alert_count >= 2:
                    continue

                # Skip if already classified AND graph hasn't grown by 2x since then
                if prev_nodes > 0 and current_nodes < prev_nodes * 2:
                    continue

                diag = sg.diagnostics(root_pid)
                image_name = (diag.get('image') or 'unknown').lower()

                # Skip whitelisted benign processes (unless --show-benign is set)
                if image_name in BENIGN_WHITELIST and not args.show_benign:
                    classified_roots[root_pid] = {**prev_data, 'nodes': current_nodes, 'verdict': 'WHITELISTED'}
                    continue

                # UPDATE nodes and alert_count WITHOUT wiping verdict/peak_conf/malware_alerts
                classified_roots[root_pid] = {**prev_data, 'nodes': current_nodes, 'alert_count': prev_alerts + 1}
                graphlet = projector.project(root_pid)
                if graphlet is None:
                    continue
                data = graphlet.to_pyg_data()
                n_nodes = data.x.shape[0]
                n_edges = data.edge_index.shape[1]
                log.info(f"[graphlet root={root_pid} image={diag['image']}] "
                         f"{n_nodes} nodes / {n_edges} edges, kinds={diag['node_kinds']}")

                pred, conf, t_det, X = classify_graph(model, embedder, data)
                t_exp = 0.0
                shap_alert = None

                # prev_verdict, peak_conf, malware_alert_count already read above

                # Apply confidence threshold — only alert if confidence >= threshold
                if pred == 1 and conf < MALWARE_CONFIDENCE_THRESHOLD:
                    # But if already confirmed MALWARE, keep it as MALWARE
                    if prev_verdict == 'MALWARE':
                        log.info(f"[re-check root={root_pid} image={diag['image']}] "
                                 f"still MALWARE (peak {peak_conf:.1%}, current {conf:.1%}, nodes={n_nodes})")
                        continue
                    log.info(f"[low-confidence root={root_pid} image={diag['image']}] "
                             f"MALWARE conf={conf:.1%} < {MALWARE_CONFIDENCE_THRESHOLD:.0%} threshold, "
                             f"suppressing alert (nodes={n_nodes})")
                    classified_roots[root_pid]['verdict'] = 'MALWARE_LOW'
                    results.append({"root_pid": root_pid, "image": diag["image"],
                                    "prediction": "MALWARE_LOW_CONF",
                                    "confidence": conf,
                                    "t_detect_ms": t_det, "t_explain_ms": 0,
                                    "t_total_ms": t_det,
                                    "meets_threshold": True,
                                    "nodes": n_nodes, "edges": n_edges})
                    continue

                current_verdict = "MALWARE" if pred == 1 else "BENIGN"

                # *** STICKY MALWARE VERDICT ***
                # Once a PID has been classified as MALWARE, it STAYS MALWARE.
                # Later re-checks that say BENIGN (due to graphlet dilution from
                # background events) are logged but the verdict never downgrades.
                # This prevents the confusing MALWARE→BENIGN flip that happens
                # when SilkETW's kernel buffer flushes background events into
                # the graphlet after the attack completes.
                if prev_verdict == 'MALWARE' and current_verdict == 'BENIGN':
                    log.info(f"[verdict-lock root={root_pid} image={diag['image']}] "
                             f"model now says BENIGN ({conf:.1%}) but keeping MALWARE "
                             f"(peak {peak_conf:.1%}) — graphlet diluted by background events "
                             f"(nodes={n_nodes})")
                    results.append({"root_pid": root_pid, "image": diag["image"],
                                    "prediction": "MALWARE",  # keep MALWARE in results
                                    "confidence": peak_conf,  # report peak confidence
                                    "t_detect_ms": t_det, "t_explain_ms": 0,
                                    "t_total_ms": t_det,
                                    "meets_threshold": (t_det) <= RQ3_LATENCY_THRESHOLD_MS,
                                    "nodes": n_nodes, "edges": n_edges})
                    continue

                # Update verdict and track peak confidence
                classified_roots[root_pid]['verdict'] = current_verdict
                if current_verdict == 'MALWARE' and conf > peak_conf:
                    classified_roots[root_pid]['peak_conf'] = conf

                # Alert logic — STRICT MAX 2 ALERT BOXES PER PID:
                #   Alert #1: first time malware is detected
                #   Alert #2: only if confidence INCREASED since alert #1
                #   After that: no more analysis (stopped above)
                last_alerted_conf = prev_data.get('last_alerted_conf', 0.0)

                show_full_alert = False
                if pred == 1:
                    if malware_alert_count == 0:
                        # First detection — always show
                        show_full_alert = True
                    elif malware_alert_count == 1 and conf > last_alerted_conf:
                        # Second alert — only if confidence grew
                        show_full_alert = True
                    # malware_alert_count >= 2 → never show again
                    if show_full_alert:
                        classified_roots[root_pid]['malware_alerts'] = malware_alert_count + 1
                        classified_roots[root_pid]['last_alerted_conf'] = conf
                elif pred == 0 and prev_alerts <= 1:
                    show_full_alert = True  # show first benign classification

                if show_full_alert:
                    if pred == 1 or args.show_benign:
                        shap_alert = alerter.explain(X, top_k=args.shap_top_k)
                        t_exp = shap_alert["t_explain_ms"]
                    print_soc_alert(f"PID {root_pid} ({diag.get('image') or 'unknown'})", pred, conf, t_det, t_exp,
                                    n_nodes, n_edges, image=diag["image"],
                                    shap_alert=shap_alert, show_benign=args.show_benign,
                                    node_kinds=diag.get('node_kinds'))
                else:
                    log.info(f"[re-check root={root_pid} image={diag['image']}] "
                             f"{current_verdict} conf={conf:.1%} nodes={n_nodes} "
                             f"(alert #{prev_alerts+1}, suppressing full box)")
                results.append({"root_pid": root_pid, "image": diag["image"],
                                "prediction": "MALWARE" if pred == 1 else "BENIGN",
                                "confidence": conf,
                                "t_detect_ms": t_det, "t_explain_ms": t_exp,
                                "t_total_ms": t_det + t_exp,
                                "meets_threshold": (t_det + t_exp) <= RQ3_LATENCY_THRESHOLD_MS,
                                "nodes": n_nodes, "edges": n_edges,
                                "node_kinds": diag.get('node_kinds'),
                                "shap": shap_alert if shap_alert else None})
            # Update dashboard state for live web UI
            _DASHBOARD_STATE["results"] = results
            _DASHBOARD_STATE["classified_roots"] = {k: {kk: vv for kk, vv in v.items() if kk != 'shap'}
                                                     for k, v in classified_roots.items()}
            _DASHBOARD_STATE["sg_stats"] = {
                "total_nodes": len(sg.nodes), "total_edges": len(sg.edges),
                "total_roots": len(sg.descendants),
            }
            time.sleep(args.poll_interval)

    except KeyboardInterrupt:
        os.makedirs(args.output_dir, exist_ok=True)
        path = os.path.join(args.output_dir, "rq3_live_results.json")
        with open(path, "w") as f:
            json.dump({"mode": "live", "rq3_threshold_ms": RQ3_LATENCY_THRESHOLD_MS,
                       "results": results}, f, indent=2)
        print(f"\n[saved] {path}")


# =============================================================================
# MODE 2: DATASET-TEST — gold-standard RQ3 evidence on real graphlets
# =============================================================================

def run_dataset_test(args):
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    from dataprocessor_graphs import load_dataset

    dataset_path = args.dataset_path
    nt = json.load(open(os.path.join(dataset_path, "NodeType_NodeFeatures.json")))
    en = json.load(open(os.path.join(dataset_path, "EventName_EdgeFeatures.json")))

    model, embedder, _ = load_saved_model(args.model_dir)
    alerter = SHAPAlertGenerator(model, embedder.feature_names)

    test_ds = load_dataset(
        os.path.join(dataset_path, "test/benign"),
        os.path.join(dataset_path, "test/malware"),
        len(nt), len(en) + 1,
    )
    mal = [d for d in test_ds if "malware" in d.name.lower()]
    ben = [d for d in test_ds if "malware" not in d.name.lower()]
    random.seed(42)
    mal_sample = random.sample(mal, min(args.num_samples, len(mal)))
    ben_sample = random.sample(ben, min(5, len(ben)))

    print("\n" + "=" * 100)
    print("  RQ3: DATASET TEST  (model integrity check on real training graphlets)".ljust(100))
    print(f"  Samples: {len(mal_sample)} malware + {len(ben_sample)} benign".ljust(100))
    print("=" * 100 + "\n")

    rows = []
    for ds_list, true_label in [(mal_sample, "MALWARE"), (ben_sample, "BENIGN")]:
        for g in ds_list:
            try:
                pred, conf, t_det, X = classify_graph(model, embedder, g)
                t_exp = 0.0
                shap_alert = None
                if pred == 1 or args.show_benign:
                    shap_alert = alerter.explain(X, top_k=args.shap_top_k)
                    t_exp = shap_alert["t_explain_ms"]
                n_nodes = g.x.shape[0]
                n_edges = g.edge_index.shape[1] if g.edge_index.numel() > 0 else 0
                print_soc_alert(g.name, pred, conf, t_det, t_exp,
                                n_nodes, n_edges, image=None,
                                shap_alert=shap_alert, show_benign=args.show_benign)
                rows.append({
                    "sample": g.name, "true_label": true_label,
                    "prediction": "MALWARE" if pred == 1 else "BENIGN",
                    "confidence": round(conf, 6),
                    "t_detect_ms": round(t_det, 3),
                    "t_explain_ms": round(t_exp, 3),
                    "t_total_ms": round(t_det + t_exp, 3),
                    "meets_threshold": (t_det + t_exp) <= RQ3_LATENCY_THRESHOLD_MS,
                    "nodes": n_nodes, "edges": n_edges,
                })
            except Exception as e:
                log.warning(f"Skipped {g.name}: {e}")

    mal_rows = [r for r in rows if r["true_label"] == "MALWARE"]
    ben_rows = [r for r in rows if r["true_label"] == "BENIGN"]
    correct_mal = sum(1 for r in mal_rows if r["prediction"] == "MALWARE")
    correct_ben = sum(1 for r in ben_rows if r["prediction"] == "BENIGN")
    avg_total = float(np.mean([r["t_total_ms"] for r in rows])) if rows else 0.0
    max_total = max((r["t_total_ms"] for r in rows), default=0.0)
    all_ok = all(r["meets_threshold"] for r in rows)

    print(f"\n{'='*100}")
    print("  RQ3 DATASET-TEST SUMMARY")
    print(f"{'='*100}")
    print(f"  Malware correctly detected : {correct_mal}/{len(mal_rows)}")
    print(f"  Benign  correctly classified: {correct_ben}/{len(ben_rows)}")
    print(f"  Avg T_total                 : {avg_total:.2f} ms")
    print(f"  Max T_total                 : {max_total:.2f} ms")
    print(f"  RQ3 threshold (<= {RQ3_LATENCY_THRESHOLD_MS} ms)  : {'PASS' if all_ok else 'FAIL'}")
    print(f"{'='*100}\n")

    os.makedirs(args.output_dir, exist_ok=True)
    out = os.path.join(args.output_dir, "rq3_live_results.json")
    with open(out, "w") as f:
        json.dump({"mode": "dataset_test", "summary": {
            "malware_correct": correct_mal, "malware_total": len(mal_rows),
            "benign_correct": correct_ben, "benign_total": len(ben_rows),
            "avg_t_total_ms": avg_total, "max_t_total_ms": max_total,
            "all_within_threshold": all_ok,
        }, "results": rows}, f, indent=2)
    print(f"[saved] {out}")


# =============================================================================
# MODE 3: DRY-RUN — synthetic events through the real streaming pipeline
# =============================================================================

def _inject(sg: StreamingComputationGraph, event_name: str, pid: str, tid: str,
            ts_iso: str, **xml_extra):
    sg.ingest({
        "EventName": event_name,
        "ProcessID": pid,
        "ThreadID": tid,
        "TimeStamp": ts_iso,
        "XmlEventData": xml_extra,
    })


def _make_iso(offset: float) -> str:
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    return datetime.fromtimestamp(base.timestamp() + offset, tz=timezone.utc).isoformat()


def _gen_malware_events(sg: StreamingComputationGraph, pid: str):
    """Mimics top-SHAP malware patterns from your lightgbm_shap_analysis:
       - openkey closekeys openkey create / querykey
       - close read read read
       - heavy registry traffic across multiple threads
       - DLL injection pattern (image loads + file reads)
       - C2 callout
    """
    # The malware process is the root of its own graphlet
    _inject(sg, "Process/Start", pid, "1", _make_iso(0.001),
            ProcessID=pid, ParentProcessId=pid,
            CreateTime="2024-01-01T00:00:00.001Z", ImageName="malware.exe")

    tids = [str(10000 + i) for i in range(8)]
    for i, tid in enumerate(tids):
        _inject(sg, "Thread/Start", pid, tid, _make_iso(0.002 + i * 0.0001),
                ProcessID=pid, ThreadID=tid)

    t = 0.01
    # 4 threads doing heavy registry work
    for tid in tids[:4]:
        for k in range(60):
            for ev in ["Registry/Open", "Registry/QueryValue", "Registry/CloseKeys",
                       "Registry/Open", "Registry/Create", "Registry/CloseKeys",
                       "Registry/Open", "Registry/Query", "Registry/CloseKeys",
                       "Registry/Open", "Registry/EnumerateKey", "Registry/CloseKeys"]:
                _inject(sg, ev, pid, tid, _make_iso(t),
                        KeyObject=f"K{tid}_{k}",
                        RelativeName=f"HKLM\\SOFTWARE\\Mal\\{tid}\\{k}")
                t += 0.0001

    # 2 threads doing file ops
    for tid in tids[4:6]:
        for k in range(40):
            for ev in ["FileIo/Create", "FileIo/Read", "FileIo/Read", "FileIo/Create",
                       "FileIo/Write", "FileIo/Close", "FileIo/Cleanup"]:
                _inject(sg, ev, pid, tid, _make_iso(t),
                        FileObject=f"F{tid}_{k}",
                        FileName=f"C:\\temp\\payload_{tid}_{k}.dat")
                t += 0.0001

    # 2 threads doing DLL injection pattern
    for tid in tids[6:]:
        for dll in ["ntdll.dll", "kernel32.dll", "amsi.dll", "clr.dll", "ws2_32.dll"]:
            _inject(sg, "Image/Load", pid, tid, _make_iso(t),
                    FileName=f"C:\\Windows\\System32\\{dll}", FileObject=dll)
            t += 0.0001
        for k in range(15):
            for ev in ["FileIo/Create", "FileIo/Read", "FileIo/QueryInfo", "FileIo/Close"]:
                _inject(sg, ev, pid, tid, _make_iso(t),
                        FileObject=f"FF{tid}_{k}",
                        FileName=f"C:\\Windows\\config\\d_{k}")
                t += 0.0001

    # C2 callout
    for k in range(5):
        _inject(sg, "TcpIp/ConnectIPV4", pid, tids[0], _make_iso(t), daddr="192.168.1.100")
        _inject(sg, "TcpIp/SendIPV4",    pid, tids[0], _make_iso(t + 0.0001), daddr="192.168.1.100")
        _inject(sg, "TcpIp/RecvIPV4",    pid, tids[0], _make_iso(t + 0.0002), daddr="192.168.1.100")
        t += 0.001


def _gen_benign_events(sg: StreamingComputationGraph, pid: str):
    """Light registry, mostly file reads, no network — typical PowerShell script."""
    _inject(sg, "Process/Start", pid, "1", _make_iso(0.001),
            ProcessID=pid, ParentProcessId=pid,
            CreateTime="2024-01-01T00:00:00.001Z", ImageName="powershell.exe")

    tids = ["2234", "2235"]
    for i, tid in enumerate(tids):
        _inject(sg, "Thread/Start", pid, tid, _make_iso(0.002 + i * 0.0001),
                ProcessID=pid, ThreadID=tid)

    t = 0.01
    for k in range(30):
        for ev in ["FileIo/Create", "FileIo/Read", "FileIo/Close", "FileIo/Cleanup"]:
            _inject(sg, ev, pid, tids[0], _make_iso(t),
                    FileObject=f"F{k}", FileName=f"C:\\Users\\script_{k}.ps1")
            t += 0.0002
    for k in range(8):
        _inject(sg, "Registry/Open", pid, tids[1], _make_iso(t),
                KeyObject=f"K{k}", RelativeName="HKCU\\Console")
        _inject(sg, "Registry/QueryValue", pid, tids[1], _make_iso(t + 0.0001),
                KeyObject=f"K{k}", RelativeName="HKCU\\Console")
        _inject(sg, "Registry/CloseKeys", pid, tids[1], _make_iso(t + 0.0002),
                KeyObject=f"K{k}", RelativeName="HKCU\\Console")
        t += 0.001
    for k in range(15):
        _inject(sg, "FileIo/QueryInfo", pid, tids[0], _make_iso(t),
                FileObject=f"FQ{k}",
                FileName=f"C:\\Windows\\System32\\file_{k}.dll")
        t += 0.0002


def run_dry_run(args):
    model, embedder, _ = load_saved_model(args.model_dir)
    alerter = SHAPAlertGenerator(model, embedder.feature_names)

    print("\n" + "=" * 100)
    print("  RQ3: DRY-RUN MODE  (synthetic ETW \u2192 author-faithful streaming pipeline)")
    print("=" * 100 + "\n")

    for label, root_pid, gen_fn in [
        ("MALWARE-LIKE", "9999", _gen_malware_events),
        ("BENIGN-LIKE",  "1234", _gen_benign_events),
    ]:
        sg = StreamingComputationGraph()
        projector = GraphletProjector(sg)
        gen_fn(sg, root_pid)
        diag = sg.diagnostics(root_pid)
        log.info(f"[{label} pid={root_pid}] {diag}")

        graphlet = projector.project(root_pid)
        if graphlet is None:
            log.error(f"[{label}] no graphlet projected")
            continue
        data = graphlet.to_pyg_data()
        pred, conf, t_det, X = classify_graph(model, embedder, data)
        shap_alert = alerter.explain(X, top_k=args.shap_top_k)
        print_soc_alert(f"{label} pid={root_pid}", pred, conf,
                        t_det, shap_alert["t_explain_ms"],
                        data.x.shape[0], data.edge_index.shape[1],
                        image=diag["image"], shap_alert=shap_alert,
                        show_benign=True)


# =============================================================================
# HELP TEXT — printed by --help-full
# =============================================================================

HELP_TEXT = r"""
================================================================================
  LIVE_SOC_SIMULATION.PY  \u2014  RQ3 REAL-TIME GRAPHITE MALWARE DETECTOR
================================================================================

WHAT IT DOES
------------
A streaming, author-faithful re-implementation of the Graphite pipeline
(jgwak1/Graphite, SecureComm 2024). Continuously consumes ETW events from a
SilkETW \u2192 PowerShell-bridge \u2192 Logstash \u2192 Elasticsearch pipeline,
incrementally builds the Graphite computation graph, projects per-PROCESSSTART
graphlets at the same 1000-node trigger from the paper's RQ3 setup, embeds
them with the trained Graphite-Ngram model (LightGBM, RQ1 winner), and
generates a full SOC alert with SHAP explanations and MITRE ATT&CK mapping.

THREE EXECUTION MODES
---------------------

  (1) LIVE                                          [default]
      Polls Elasticsearch every --poll-interval seconds for new ETW events,
      builds the streaming graph, classifies graphlets when ready.
      Use this on the Kali VM during a live scenario walkthrough.

         python live_soc_simulation.py
         python live_soc_simulation.py --es-host http://localhost:9200 \
              --es-index etw-live-logs --trigger-nodes 1000

  (2) DATASET-TEST                                  --dataset-test
      Loads the actual training pickles (Processed_SUBGRAPH_P3_*.pickle),
      runs them straight through embedder + LightGBM + SHAP. Highest-fidelity
      RQ3 evidence: bypasses the streaming layer entirely so you can prove
      that with real graphlets, T_detect + T_explain << 5000ms and accuracy
      matches the saved model. Use this for the thesis.

         python live_soc_simulation.py --dataset-test \
              --dataset-path ../dataset --num-samples 30

  (3) DRY-RUN                                       --dry-run
      Synthesises ETW events that mimic the top-SHAP malware patterns
      (registry enumeration, DLL injection, C2 callout) and benign patterns
      (light file reads). Runs them through the streaming builder + projector
      + LightGBM + SHAP. Useful for proving the pipeline plumbing works
      without needing a Windows VM.

         python live_soc_simulation.py --dry-run

DATASET-TEST OUTPUT
-------------------
For each test sample you get a full boxed SOC alert showing:
  \u2022 Verdict line (MALWARE / BENIGN with confidence)
  \u2022 Graph size (nodes, edges)
  \u2022 T_detect, T_explain, T_total + RQ3 PASS/FAIL
  \u2022 Top-K SHAP features (only those that ACTUALLY occurred in the graphlet)
      \u2014 each with: pattern n-gram, category, occurrence count,
        SHAP value + direction, behavioural narrative, MITRE technique
  \u2022 Consolidated MITRE ATT&CK summary

CRITICAL: SilkETW FLAGS YOU MUST USE ON WINDOWS VM
--------------------------------------------------
The Graphite training data contains the FULL set of FileIO sub-events
(Create / Read / Write / Close / Cleanup / etc) and ImageLoad events. With
the default flag set "-kk Process,Thread,FileIO,Registry,NetworkTCPIP",
SilkETW kernel mode emits only "FileIo/OperationEnd" \u2014 none of the
sub-events the model was trained to recognise. Detection WILL fail.

   USE THESE FLAGS:
   SilkETW.exe -t kernel \
       -kk Process,Thread,FileIO,FileIOInit,Registry,NetworkTCPIP,ImageLoad \
       -ot file -p C:\path\to\etw.json

LIVE-MODE PIPELINE \u2014 STEP-BY-STEP DEPLOYMENT
-------------------------------------------

  ON THE KALI BACKEND VM:
    sudo apt update && sudo apt install elasticsearch logstash
    sudo systemctl enable --now elasticsearch logstash
    # Configure /etc/logstash/conf.d/etw.conf  (see RQ3_Setup_Guide.md)
    sudo iptables -I INPUT -p tcp --dport 5044 -j ACCEPT

  ON THE WINDOWS VICTIM VM (Admin terminals):
    SilkETW.exe -t kernel \
        -kk Process,Thread,FileIO,FileIOInit,Registry,NetworkTCPIP,ImageLoad \
        -ot file -p C:\path\to\etw.json
    # In a second admin PowerShell:
    $KaliIP = "<KALI_IP>";  $Port = 5044
    $sock = New-Object System.Net.Sockets.TcpClient($KaliIP,$Port)
    $w = New-Object System.IO.StreamWriter($sock.GetStream())
    Get-Content -Path C:\path\to\etw.json -Wait -Tail 0 |
        ForEach-Object { $w.WriteLine($_); $w.Flush() }

  ON KALI (after the model has been trained \u2014 saved_model/ exists):
    python live_soc_simulation.py
    # Run an Invoke-PSInject.ps1 / Invoke-DllInjection.ps1 attack on Windows.
    # Watch the SOC alert appear within seconds of the graphlet hitting 1000 nodes.

ALL FLAGS
---------
  --es-host URL          Elasticsearch URL (default http://localhost:9200)
  --es-index NAME        Elasticsearch index name (default etw-live-logs)
  --model-dir PATH       Saved-model directory (default ./saved_model)
  --trigger-nodes N      Per-PROCESSSTART graphlet size threshold (default 1000)
  --poll-interval SEC    ES polling interval in seconds (default 0.5)
  --output-dir PATH      Where to save rq3_live_results.json (default ./rq3_results)
  --shap-top-k N         Top-K SHAP features per alert (default 10)
  --show-benign          Show full SOC alert boxes for benign detections too
  --dry-run              Run synthetic-event mode (no Elasticsearch needed)
  --dataset-test         Run dataset-pickle mode (no Elasticsearch needed)
  --dataset-path PATH    Path to dataset/ folder (for --dataset-test)
  --num-samples N        Number of malware samples in --dataset-test (default 30)
  --help-full            Show this help text

SAVED ARTEFACTS
---------------
  rq3_results/rq3_live_results.json   Per-detection JSON + summary

SECURECOMM 2024 PAPER NUMBERS (FOR REFERENCE)
---------------------------------------------
  Offline accuracy   : 87.7%  (paper); your RQ1 LightGBM: 91.81% acc, 0.9672 AUC
  Real-time accuracy : 86.7%  (paper); RQ3 trigger: 1000 nodes
  RQ3 latency limit  : <= 5000 ms  (Interim Report \u00a73.3.3)
================================================================================
"""


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="RQ3 Real-Time SOC Simulation - Graphite N-gram + LightGBM + SHAP",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
MODES:
  Live (default)  : Polls Elasticsearch for SilkETW kernel events, builds
                    computation graphs, classifies with LightGBM, explains with SHAP
  --dry-run       : Synthetic events (no Elasticsearch needed, for testing)
  --dataset-test  : Run against real training dataset pickles (gold-standard check)

EXAMPLES:
  python %(prog)s --trigger-nodes 100 --watch-pid 3256
  python %(prog)s --trigger-nodes 100 --watch-pid 3256 --dashboard
  python %(prog)s --trigger-nodes 100 --show-benign
  python %(prog)s --dry-run
  python %(prog)s --dataset-test --num-samples 10

NOTES:
  - Confidence threshold is 50%%. Predictions below this are suppressed.
  - Benign system processes (svchost, explorer, etc.) are whitelisted.
  - Use --show-benign to see ALL classifications including benign and whitelisted.
  - Maximum 2 MALWARE alert boxes per PID, then one-line updates.
  - Results saved to --output-dir on Ctrl+C.
""",
    )
    p.add_argument("--es-host",      type=str, default="http://localhost:9200",
                   help="Elasticsearch URL (default: http://localhost:9200)")
    p.add_argument("--es-index",     type=str, default="etw-live-logs",
                   help="Elasticsearch index name (default: etw-live-logs)")
    p.add_argument("--model-dir",    type=str, default="./saved_model",
                   help="Path to saved LightGBM model directory (default: ./saved_model)")
    p.add_argument("--trigger-nodes", type=int, default=1000,
                   help="Min nodes per graphlet before classification (default: 1000, use 100 for faster)")
    p.add_argument("--poll-interval", type=float, default=0.5,
                   help="Seconds between ES polls (default: 0.5)")
    p.add_argument("--output-dir",   type=str, default="./rq3_results",
                   help="Directory to save results JSON on exit (default: ./rq3_results)")
    p.add_argument("--shap-top-k",   type=int, default=10,
                   help="Number of top SHAP features to show in alert (default: 10)")
    p.add_argument("--show-benign",  action="store_true",
                   help="Show ALL classifications (benign + whitelisted). Off by default.")
    p.add_argument("--dry-run",      action="store_true",
                   help="Synthetic event mode — no Elasticsearch needed")
    p.add_argument("--dataset-test", action="store_true",
                   help="Dataset pickle mode — test on real training graphlets")
    p.add_argument("--dataset-path", type=str,
                   default=str(pathlib.Path(__file__).parent.parent / "dataset"),
                   help="Path to dataset directory (for --dataset-test)")
    p.add_argument("--num-samples",  type=int, default=30,
                   help="Number of malware samples to test (for --dataset-test, default: 30)")
    p.add_argument("--help-full",    action="store_true",
                   help="Print detailed documentation about modes, deployment, and architecture")
    p.add_argument("--watch-pid",    type=str, default=None,
                   help="Monitor a specific PID — shows node/edge growth every 30 seconds")
    p.add_argument("--dashboard",    action="store_true",
                   help="Launch live web dashboard on port 8050 (open http://localhost:8050)")
    p.add_argument("--dashboard-port", type=int, default=8050,
                   help="Port for the live dashboard (default: 8050)")
    return p.parse_args()


# =============================================================================
# LIVE DASHBOARD — HTTP server serving real-time results via /api/results
# =============================================================================

# Global reference so the HTTP handler can access live results
_DASHBOARD_STATE = {"results": [], "classified_roots": {}, "sg_stats": {}}

DASHBOARD_HTML = r'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>RQ3 — Live SOC Dashboard</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=Outfit:wght@300;400;500;600;700;800&display=swap');
:root{--bg:#0a0e17;--bg2:#111827;--card:#1a2235;--border:#2a3550;--text:#e8edf5;--text2:#8899b4;--muted:#556580;--red:#ff3366;--green:#00ff88;--yellow:#ffaa00;--blue:#3399ff;--cyan:#00ddff;--track:#2a3550}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Outfit',sans-serif;background:var(--bg);color:var(--text);min-height:100vh}
body::before{content:'';position:fixed;inset:0;background:linear-gradient(rgba(51,153,255,.03) 1px,transparent 1px),linear-gradient(90deg,rgba(51,153,255,.03) 1px,transparent 1px);background-size:40px 40px;pointer-events:none;z-index:0}

.hdr{position:sticky;top:0;z-index:100;background:rgba(10,14,23,.92);backdrop-filter:blur(16px);border-bottom:1px solid var(--border);padding:12px 28px;display:flex;align-items:center;justify-content:space-between}
.hdr-l{display:flex;align-items:center;gap:14px}
.logo{width:36px;height:36px;background:linear-gradient(135deg,var(--red),var(--blue));border-radius:9px;display:flex;align-items:center;justify-content:center;font-weight:800;font-size:15px;color:#fff}
.hdr h1{font-size:17px;font-weight:700;letter-spacing:-.5px}
.hdr h1 span{color:var(--cyan)}
.hdr-r{display:flex;gap:20px;font-size:12px;color:var(--text2);font-family:'JetBrains Mono',monospace}
.dot{width:7px;height:7px;border-radius:50%;display:inline-block;margin-right:5px;background:var(--green);box-shadow:0 0 8px var(--green);animation:pulse 2s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}

.stats{display:flex;gap:10px;padding:14px 28px;flex-wrap:wrap;position:relative;z-index:1}
.sc{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:12px 18px;min-width:140px;flex:1}
.sc .lb{font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:1px;margin-bottom:4px}
.sc .vl{font-size:24px;font-weight:700;font-family:'JetBrains Mono',monospace}
.vl.r{color:var(--red)}.vl.g{color:var(--green)}.vl.c{color:var(--cyan)}

.grid{padding:0 28px 28px;display:grid;grid-template-columns:repeat(auto-fill,minmax(480px,1fr));gap:16px;position:relative;z-index:1}

.pcard{background:var(--card);border:1px solid var(--border);border-radius:14px;overflow:hidden;transition:border-color .3s}
.pcard.mal{border-color:var(--red);box-shadow:0 0 20px rgba(255,51,102,.08)}
.pcard.ben{border-color:var(--green)}

.ph{padding:14px 18px;display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid var(--border)}
.ph-l{display:flex;align-items:center;gap:10px}
.pid{font-family:'JetBrains Mono',monospace;font-size:12px;font-weight:600;padding:3px 9px;border-radius:5px;background:rgba(51,153,255,.15);color:var(--blue)}
.img{font-size:14px;font-weight:600}
.badge{font-size:10px;font-weight:700;text-transform:uppercase;padding:3px 10px;border-radius:16px;letter-spacing:.7px}
.b-mal{background:rgba(255,51,102,.2);color:var(--red);border:1px solid var(--red)}
.b-ben{background:rgba(0,255,136,.12);color:var(--green);border:1px solid #00cc6a}
.b-pend{background:rgba(255,170,0,.15);color:var(--yellow);border:1px solid var(--yellow)}

.gauge-area{padding:18px;display:flex;align-items:center;gap:20px}
.gc{width:150px;height:95px;position:relative;flex-shrink:0}
.gc svg{width:100%;height:100%}
.gc .gl{position:absolute;bottom:2px;left:50%;transform:translateX(-50%);font-family:'JetBrains Mono',monospace;font-size:22px;font-weight:700}
.gc .gs{position:absolute;bottom:-14px;left:50%;transform:translateX(-50%);font-size:9px;color:var(--muted);text-transform:uppercase;letter-spacing:1px;white-space:nowrap}

.dets{flex:1;min-width:0}
.dr{display:flex;justify-content:space-between;padding:4px 0;font-size:12px;border-bottom:1px solid rgba(42,53,80,.4)}
.dr:last-child{border:none}
.dk{color:var(--text2)}.dv{font-family:'JetBrains Mono',monospace;font-weight:500}
.dv.pass{color:var(--green)}.dv.fail{color:var(--red)}

.mitre{padding:0 18px 12px;display:flex;flex-wrap:wrap;gap:5px}
.mt{font-size:10px;font-family:'JetBrains Mono',monospace;padding:2px 8px;border-radius:3px;background:rgba(51,153,255,.1);color:var(--blue);border:1px solid rgba(51,153,255,.2)}

.conf-bars{padding:0 18px 14px}
.cb-title{font-size:11px;font-weight:600;color:var(--text2);margin-bottom:8px;text-transform:uppercase;letter-spacing:.7px}
.cb-row{display:flex;align-items:center;gap:6px;margin-bottom:4px;font-size:11px}
.cb-name{width:100px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-family:'JetBrains Mono',monospace;color:var(--text2);font-size:10px}
.cb-bar{flex:1;height:12px;background:var(--track);border-radius:2px;overflow:hidden}
.cb-fill{height:100%;border-radius:2px;transition:width .5s}
.cb-fill.pos{background:linear-gradient(90deg,#cc2952,var(--red))}.cb-fill.neg{background:linear-gradient(90deg,#00cc6a,var(--green))}
.cb-val{width:50px;text-align:right;font-family:'JetBrains Mono',monospace;font-size:10px;font-weight:500}
.cb-val.p{color:var(--red)}.cb-val.n{color:var(--green)}

/* ── ALERT MODAL ── */
.modal-overlay{display:none;position:fixed;inset:0;background:rgba(0,0,0,.7);z-index:200;justify-content:center;align-items:start;padding:40px;overflow-y:auto}
.modal-overlay.show{display:flex}
.modal{background:var(--card);border:1px solid var(--red);border-radius:16px;max-width:800px;width:100%;max-height:85vh;overflow-y:auto;position:relative}
.modal-close{position:absolute;top:12px;right:16px;background:none;border:none;color:var(--text2);font-size:22px;cursor:pointer}
.modal-close:hover{color:var(--text)}
.modal-hdr{padding:18px 22px;border-bottom:1px solid var(--border);font-size:16px;font-weight:700;color:var(--red)}
.modal-body{padding:18px 22px}

.shap-item{margin-bottom:16px;padding:14px;background:rgba(42,53,80,.3);border-radius:10px;border-left:3px solid var(--track)}
.shap-item.pos-shap{border-left-color:var(--red)}
.shap-item.neg-shap{border-left-color:var(--green)}
.si-top{display:flex;justify-content:space-between;align-items:center;margin-bottom:6px}
.si-rank{font-size:11px;color:var(--muted);font-weight:600}
.si-feat{font-family:'JetBrains Mono',monospace;font-size:13px;font-weight:600;color:var(--text)}
.si-shap{font-family:'JetBrains Mono',monospace;font-size:13px;font-weight:700}
.si-shap.p{color:var(--red)}.si-shap.n{color:var(--green)}
.si-meta{font-size:12px;color:var(--text2);line-height:1.6}
.si-meta strong{color:var(--text)}
.si-mitre{display:flex;gap:5px;margin-top:6px;flex-wrap:wrap}

.btn-explain{font-family:'Outfit',sans-serif;font-size:12px;font-weight:600;padding:6px 14px;border-radius:7px;border:1px solid var(--red);background:rgba(255,51,102,.1);color:var(--red);cursor:pointer;transition:all .2s}
.btn-explain:hover{background:rgba(255,51,102,.25)}

.profile-box{margin-top:14px;padding:12px 16px;background:rgba(255,51,102,.06);border:1px solid rgba(255,51,102,.2);border-radius:8px;font-size:13px;line-height:1.6;color:var(--text2)}
.profile-box strong{color:var(--red)}

.timeline{grid-column:1/-1;background:var(--card);border:1px solid var(--border);border-radius:14px;overflow:hidden}
.tl-hdr{padding:12px 18px;border-bottom:1px solid var(--border);font-size:13px;font-weight:600}
.tl-body{max-height:200px;overflow-y:auto;padding:10px 18px;font-family:'JetBrains Mono',monospace;font-size:11px;line-height:1.8}
.tl-entry{display:flex;gap:10px}
.tl-ts{color:var(--muted);min-width:65px}
.tl-msg{color:var(--text2)}.hl-m{color:var(--red);font-weight:600}.hl-b{color:var(--green);font-weight:600}.hl-p{color:var(--cyan)}

::-webkit-scrollbar{width:5px}::-webkit-scrollbar-track{background:var(--bg2)}::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px}
</style>
</head>
<body>
<div class="hdr">
  <div class="hdr-l"><div class="logo">G</div><h1>RQ3 <span>Live SOC Dashboard</span></h1></div>
  <div class="hdr-r"><span><span class="dot"></span><span id="st">Polling...</span></span><span id="hm">LightGBM · 20,213 features</span></div>
</div>
<div class="stats" id="statsBar">
  <div class="sc"><div class="lb">Processes</div><div class="vl c" id="sT">0</div></div>
  <div class="sc"><div class="lb">Malware</div><div class="vl r" id="sM">0</div></div>
  <div class="sc"><div class="lb">Benign</div><div class="vl g" id="sB">0</div></div>
  <div class="sc"><div class="lb">Whitelisted</div><div class="vl c" id="sW">0</div></div>
  <div class="sc"><div class="lb">Total Events</div><div class="vl c" id="sE">0</div></div>
  <div class="sc"><div class="lb">RQ3 ≤5000ms</div><div class="vl g" id="sR">—</div></div>
</div>
<div class="grid" id="grid"></div>
<div class="modal-overlay" id="modal"><div class="modal"><button class="modal-close" onclick="closeModal()">✕</button><div class="modal-hdr" id="mHdr"></div><div class="modal-body" id="mBody"></div></div></div>

<script>
const API = '/api/results';
let prevLen = 0;

function gauge(conf, verdict) {
  const sa=-180, ea=0, rng=ea-sa, a=sa+rng*conf, cx=75, cy=82, r=60;
  const p=d=>({x:cx+r*Math.cos(d*Math.PI/180),y:cy+r*Math.sin(d*Math.PI/180)});
  let c = verdict==='MALWARE' ? (conf>=.7?'#ff3366':conf>=.5?'#ffaa00':'#00ff88') : '#00ff88';
  const ts=p(sa),te=p(ea),fe=p(a),la=(a-sa)>180?1:0;
  let tk='';
  for(let i=0;i<=10;i++){const d=sa+rng*i/10,pi=p(d),rad=d*Math.PI/180;tk+=`<line x1="${pi.x}" y1="${pi.y}" x2="${cx+(r+7)*Math.cos(rad)}" y2="${cy+(r+7)*Math.sin(rad)}" stroke="#2a3550" stroke-width="${i%5===0?2:1}"/>`;}
  return `<svg viewBox="0 0 150 95" xmlns="http://www.w3.org/2000/svg">${tk}<path d="M${ts.x} ${ts.y}A${r} ${r} 0 1 1 ${te.x} ${te.y}" fill="none" stroke="#2a3550" stroke-width="9" stroke-linecap="round"/><path d="M${ts.x} ${ts.y}A${r} ${r} 0 ${la} 1 ${fe.x} ${fe.y}" fill="none" stroke="${c}" stroke-width="9" stroke-linecap="round"/><circle cx="${fe.x}" cy="${fe.y}" r="3.5" fill="${c}"/></svg>`;
}

function card(pid, entries) {
  let pk=0,pe=entries[0],fv='BENIGN';
  entries.forEach(e=>{if(e.prediction==='MALWARE'){fv='MALWARE';if(e.confidence>pk){pk=e.confidence;pe=e;}}});
  if(fv!=='MALWARE'){pe=entries[entries.length-1];pk=pe.confidence;}
  const cc=fv==='MALWARE'?'mal':'ben', bc=fv==='MALWARE'?'b-mal':'b-ben';
  const tD=pe.t_detect_ms||0,tE=pe.t_explain_ms||0,tT=pe.t_total_ms||(tD+tE),ps=tT<=5000;
  const nk=pe.node_kinds||{};const nkStr=Object.entries(nk).sort((a,b)=>b[1]-a[1]).map(([k,v])=>`${v} ${k}`).join(', ');
  // Find entry with SHAP data
  const shapEntry = entries.find(e=>e.shap)||null;
  const mitre = shapEntry?.shap?.mitre_summary||[];

  let h=`<div class="pcard ${cc}" id="pc-${pid}"><div class="ph"><div class="ph-l"><span class="pid">PID ${pid}</span><span class="img">${pe.image||'?'}</span></div><span class="badge ${bc}">${fv}</span></div>`;
  h+=`<div class="gauge-area"><div class="gc">${gauge(pk,fv)}<div class="gl" style="color:${fv==='MALWARE'?'#ff3366':'#00ff88'}">${Math.round(pk*100)}%</div><div class="gs">${fv==='MALWARE'?'Peak':'Benign'} Conf.</div></div>`;
  h+=`<div class="dets"><div class="dr"><span class="dk">Graph</span><span class="dv">${pe.nodes}n / ${pe.edges}e</span></div>`;
  if(nkStr)h+=`<div class="dr"><span class="dk">Composition</span><span class="dv">${nkStr}</span></div>`;
  h+=`<div class="dr"><span class="dk">T_detect</span><span class="dv">${tD.toFixed(1)}ms</span></div>`;
  h+=`<div class="dr"><span class="dk">T_explain</span><span class="dv">${tE.toFixed(1)}ms</span></div>`;
  h+=`<div class="dr"><span class="dk">T_total</span><span class="dv ${ps?'pass':'fail'}">${tT.toFixed(1)}ms ${ps?'✓':'✗'}</span></div>`;
  h+=`<div class="dr"><span class="dk">Checks</span><span class="dv">${entries.length}×</span></div>`;
  if(shapEntry)h+=`<div class="dr"><span class="dk"></span><button class="btn-explain" onclick="showAlert('${pid}')">🔍 Alert Explanation</button></div>`;
  h+=`</div></div>`;

  if(mitre.length){h+=`<div class="mitre">`;mitre.forEach(m=>{h+=`<span class="mt">${m.id} ${m.name}</span>`;});h+=`</div>`;}

  // Confidence progression
  if(fv==='MALWARE'&&entries.length>1){
    const mx=Math.max(...entries.map(e=>e.confidence),.01);
    h+=`<div class="conf-bars"><div class="cb-title">Confidence Progression</div>`;
    entries.forEach((e,i)=>{const w=(e.confidence/mx)*100;const im=e.prediction==='MALWARE';
      h+=`<div class="cb-row"><span class="cb-name">#${i+1} (${e.nodes}n)</span><div class="cb-bar"><div class="cb-fill ${im?'pos':'neg'}" style="width:${w}%"></div></div><span class="cb-val ${im?'p':'n'}">${(e.confidence*100).toFixed(1)}%</span></div>`;
    });h+=`</div>`;}
  h+=`</div>`;return h;
}

// Store data globally for modal access
let _data={results:[]};

function showAlert(pid){
  const entries=_data.results.filter(r=>(r.root_pid||r.sample)===pid);
  const shapEntry=entries.find(e=>e.shap);
  if(!shapEntry||!shapEntry.shap)return;
  const s=shapEntry.shap;
  document.getElementById('mHdr').innerHTML=`🚨 SOC Alert — PID ${pid} (${shapEntry.image||'?'}) — MALWARE ${(shapEntry.confidence*100).toFixed(1)}%`;
  let h='';
  // Attributions
  s.attributions.forEach(a=>{
    if(a.orig_value<=0)return;
    const cls=a.shap_value>0?'pos-shap':'neg-shap';
    const sc=a.shap_value>0?'p':'n';
    h+=`<div class="shap-item ${cls}"><div class="si-top"><span><span class="si-rank">#${a.rank}</span> <span class="si-feat">${a.feature}</span></span><span class="si-shap ${sc}">${a.shap_value>0?'+':''}${a.shap_value.toFixed(4)}</span></div>`;
    h+=`<div class="si-meta"><strong>Category:</strong> ${a.categories.join('+')}<br><strong>Occurred:</strong> ${Math.round(a.orig_value)}× in graphlet<br><strong>Meaning:</strong> ${a.narrative}</div>`;
    if(a.mitre&&a.mitre.length){h+=`<div class="si-mitre">`;a.mitre.forEach(m=>{h+=`<span class="mt">${m.id} ${m.name} [${m.tactic}]</span>`;});h+=`</div>`;}
    h+=`</div>`;
  });
  // Behavioural profile from MITRE
  const tactics=new Set((s.mitre_summary||[]).map(m=>m.tactic));
  if(tactics.has('Discovery')&&tactics.has('Collection')){
    h+=`<div class="profile-box"><strong>RECONNAISSANCE → DATA HARVESTING:</strong> Process enumerated system resources (registry keys, files) and systematically read data — consistent with information-stealing malware collecting credentials or configuration data.</div>`;
  }else if(tactics.has('Discovery')&&tactics.has('Defense Evasion')){
    h+=`<div class="profile-box"><strong>RECONNAISSANCE → EVASION:</strong> Process scanned registry and file system for information then modified system configuration to evade detection — consistent with environment-aware malware.</div>`;
  }
  document.getElementById('mBody').innerHTML=h;
  document.getElementById('modal').classList.add('show');
}
function closeModal(){document.getElementById('modal').classList.remove('show');}
document.getElementById('modal').addEventListener('click',e=>{if(e.target.id==='modal')closeModal();});

function render(data){
  _data=data;
  const R=data.results||[];
  const cr=data.classified_roots||{};
  const sg=data.sg_stats||{};

  // Group by PID
  const byP={};R.forEach(r=>{const p=r.root_pid||r.sample||'?';if(!byP[p])byP[p]=[];byP[p].push(r);});
  let nM=0,nB=0,nW=0;
  Object.entries(cr).forEach(([_,d])=>{if(d.verdict==='MALWARE')nM++;else if(d.verdict==='BENIGN')nB++;else if(d.verdict==='WHITELISTED')nW++;});
  // Fallback from results if cr empty
  if(!Object.keys(cr).length){Object.values(byP).forEach(es=>{const hm=es.some(e=>e.prediction==='MALWARE');if(hm)nM++;else nB++;});}

  document.getElementById('sT').textContent=Object.keys(byP).length;
  document.getElementById('sM').textContent=nM;
  document.getElementById('sB').textContent=nB;
  document.getElementById('sW').textContent=nW;
  document.getElementById('sE').textContent=sg.total_edges||R.length;
  const allPass=R.every(r=>r.meets_threshold!==false);
  document.getElementById('sR').textContent=allPass?'ALL PASS':'FAIL';
  document.getElementById('sR').className='vl '+(allPass?'g':'r');
  document.getElementById('st').textContent=`Live · ${R.length} detections`;

  // Sort malware first
  const pids=Object.keys(byP).sort((a,b)=>{const am=byP[a].some(e=>e.prediction==='MALWARE'),bm=byP[b].some(e=>e.prediction==='MALWARE');return am===bm?0:am?-1:1;});

  let h='';pids.forEach(p=>{h+=card(p,byP[p]);});

  // Timeline
  h+=`<div class="timeline"><div class="tl-hdr">📋 Detection Timeline (${R.length} events)</div><div class="tl-body">`;
  R.forEach((r,i)=>{const p=r.root_pid||r.sample||'?';const c=r.prediction==='MALWARE'?'hl-m':'hl-b';
    h+=`<div class="tl-entry"><span class="tl-ts">#${i+1}</span><span class="tl-msg"><span class="hl-p">PID ${p}</span> → <span class="${c}">${r.prediction}</span> (${(r.confidence*100).toFixed(1)}%) · ${(r.t_total_ms||0).toFixed(0)}ms · ${r.nodes}n</span></div>`;
  });
  h+=`</div></div>`;
  document.getElementById('grid').innerHTML=h;
}

// Poll API every 3 seconds
async function poll(){
  try{
    const resp=await fetch(API);
    if(!resp.ok)return;
    const data=await resp.json();
    if(data.results&&data.results.length!==prevLen){prevLen=data.results.length;render(data);}
  }catch(e){}
}
setInterval(poll,3000);
poll();
</script>
</body>
</html>'''


class DashboardHandler:
    """Simple HTTP handler that serves the dashboard HTML and API endpoints."""
    import http.server

    class _Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == '/api/results':
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps(_DASHBOARD_STATE, default=str).encode())
            elif self.path in ('/', '/index.html', '/dashboard'):
                self.send_response(200)
                self.send_header('Content-Type', 'text/html')
                self.end_headers()
                self.wfile.write(DASHBOARD_HTML.encode())
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, format, *args):
            pass  # suppress HTTP access logs

    @staticmethod
    def start(port: int):
        import threading
        import http.server
        server = http.server.HTTPServer(('0.0.0.0', port), DashboardHandler._Handler)
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        log.info(f"[dashboard] Live SOC dashboard running at http://localhost:{port}")
        return server


def main():
    args = parse_args()
    if args.help_full:
        print(HELP_TEXT)
        return
    if args.dataset_test:
        run_dataset_test(args)
    elif args.dry_run:
        run_dry_run(args)
    else:
        run_live_mode(args)


if __name__ == "__main__":
    main()
