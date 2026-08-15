# Real-Time SOC Detection Pipeline (RQ3 and RQ4)

This folder holds the live detection stage. It takes the LightGBM model trained in the
offline experiments and runs it as a streaming detector, answering two research questions:

- **RQ3**: can the model detect malware on ETW events captured in real time?
- **RQ4**: what is the detection and explanation latency, and does it stay inside the
  operational budget of 5,000 ms?

`live_soc_simulation.py` is a streaming re-implementation of the batch Graphite pipeline.
It rebuilds the computation graph incrementally as events arrive, projects a graphlet for
each process tree, embeds it with the same N-gram embedder used in training, classifies
with LightGBM, explains the decision with SHAP, and maps the drivers to MITRE ATT&CK to
produce a SOC alert.

## Files

- `live_soc_simulation.py`: the detection engine. Contains the streaming graph builder, the
  graphlet projector, the SilkETW event-name translation tables, the SHAP alert generator,
  and an optional live web dashboard.
- `logstash_pipeline_example.conf`: the ingestion config that receives events over HTTP and
  forwards them to Elasticsearch.
- `saved_model/`: the saved artefacts the engine loads at start up
  (`lightgbm_model.joblib`, `graphite_embedder.joblib`, `model_meta.json`).
- `rq3&4_results/`: the live detection results.

The SilkETW to Graphite mapping (referred to as `SILKETW_TO_GRAPHITE` in Appendix A.3) is
defined inside `live_soc_simulation.py`, so there is no separate lookup file.

## Model folder name

The engine loads the model from `./saved_model` by default. If your folder is named
`models` instead, either rename it to `saved_model` or add `--model-dir models` to every
command. The commands below assume `saved_model`, matching the default.

## Three run modes

Only the live mode needs the full capture stack. The other two run from the model and the
dataset alone, which makes them the ones an examiner can reproduce.

### 1. Live mode (default)

Polls Elasticsearch for new SilkETW events, builds the streaming graph, and classifies a
process once its graphlet reaches the trigger size.

```bash
python live_soc_simulation.py --trigger-nodes 200 --dashboard --watch-pid 5380
```

The dashboard then serves at `http://localhost:8050`. This mode needs SilkETW, Logstash,
and Elasticsearch running (see deployment).

### 2. Dataset-test mode

Runs the real test graphlets straight through the embedder, LightGBM, and SHAP. No
Elasticsearch or Windows VM needed. This is the highest-fidelity RQ3 and RQ4 evidence.

```bash
python live_soc_simulation.py --dataset-test --dataset-path ../dataset --num-samples 30
```

This mode imports `dataprocessor_graphs.py`, so run with `PYTHONPATH=../core` or copy that
file into this folder.

### 3. Dry-run mode

Synthesises ETW events that mimic malware and benign patterns, then runs them through the
full streaming pipeline. No Elasticsearch or Windows VM needed.

```bash
python live_soc_simulation.py --dry-run
```

## Live deployment

The live pipeline runs across two virtual machines:

```
Windows victim VM                              Kali Linux VM
-----------------                              -------------
attack script (PowerShell)                     Logstash  (HTTP input, port 5044)
   -> ETW kernel events                           -> Elasticsearch  (index: etw-live-logs)
   -> SilkETW (kernel mode, specific flags)          -> live_soc_simulation.py
   -> forwarded as JSON  --------------------->          -> SOC alert / dashboard
```

SilkETW runs in kernel mode so events are attributed to the originating attack process
rather than to the SilkETW consumer.

### Critical SilkETW flags

The model was trained on the full set of FileIO sub-events and ImageLoad events. The
default kernel flag set emits only `FileIo/OperationEnd` and detection will fail. Capture
with the full set, including `FileIOInit` and `ImageLoad`:

```
SilkETW.exe -t kernel \
    -kk Process,Thread,FileIO,FileIOInit,Registry,NetworkTCPIP,ImageLoad \
    -ot file -p C:\path\to\etw.json
```

The full deployment walkthrough is printed by `python live_soc_simulation.py --help-full`.

## logstash_pipeline_example.conf

```conf
input {
  http {
    port => 5044
    codec => json
  }
}
filter {
}
output {
  elasticsearch {
    hosts => ["http://localhost:9200"]
    index => "etw-live-logs"
  }
}
```

An HTTP listener on port 5044 reads each event as JSON. The filter is left empty on
purpose, all grouping and noise handling is done inside `live_soc_simulation.py` after the
events are read back from Elasticsearch. Every event is written into the `etw-live-logs`
index. Edit the host and index to point at a different Elasticsearch instance.

## How live_soc_simulation.py works

Once events are indexed, the engine polls the index, translates each SilkETW event name into
the Graphite vocabulary, updates the streaming computation graph, and once a process root
reaches the trigger size projects its graphlet. It embeds the graphlet, classifies it with
LightGBM, runs SHAP on flagged processes, maps the top drivers to MITRE ATT&CK, and prints
a structured SOC alert (and pushes it to the dashboard if enabled). Common Windows system
processes are whitelisted, a verdict holds as malware once flagged so it does not flip back
to benign as later background events dilute the graphlet, and at most two full alert boxes
are shown per process.

## Key flags

| Flag | Default | Purpose |
|------|---------|---------|
| `--trigger-nodes` | `1000` | Nodes a process root must reach before it is scored. |
| `--es-host` | `http://localhost:9200` | Elasticsearch URL (live mode). |
| `--es-index` | `etw-live-logs` | Elasticsearch index (live mode). |
| `--shap-top-k` | `10` | Number of top SHAP features per alert. |
| `--show-benign` | off | Also show full alert boxes for benign and whitelisted processes. |
| `--dashboard` | off | Launch the live web dashboard on port 8050. |
| `--watch-pid` | none | Log node and edge growth for one PID every 30 seconds. |
| `--dataset-test` | off | Run against real dataset graphlets. |
| `--dry-run` | off | Run against synthetic events. |
| `--help-full` | | Print the full deployment and architecture guide. |

## Results

- All four attack scenarios were detected.
- Detection plus explanation latency ranged from about 312 ms to 1,300 ms, well inside the
  5,000 ms budget (RQ4).
- Live mode suppresses alerts below a 0.70 malware confidence gate to reduce false
  positives. This 0.70 gate differs from the 75% figure stated in the thesis text, worth
  reconciling before the viva.
