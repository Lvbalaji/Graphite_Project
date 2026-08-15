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

- `live_soc_simulation.py`: the detection engine. Contains the streaming graph builder,
  the graphlet projector, the SilkETW event-name translation tables, the SHAP alert
  generator, and an optional live web dashboard.
- `logstash_pipeline_example.conf`: the ingestion config that receives events over HTTP
  and forwards them to Elasticsearch.
- `saved_model/`: the saved artefacts the engine loads at start up.
  - `lightgbm_model.joblib`: the trained LightGBM classifier (RQ1 winner).
  - `graphite_embedder.joblib`: the fitted N-gram embedder (CountVectorizer plus the
    thread-centred pooling).
  - `model_meta.json`: model configuration and metrics recorded at training time.

The SilkETW to Graphite event-name mapping (referred to as `SILKETW_TO_GRAPHITE` in
Appendix A.3) is defined inside `live_soc_simulation.py` as the `RAW_TO_TASK_NAME`,
`TASK_TO_EVENTNAME`, and `OPCODENAME_TO_TASK` tables, so there is no separate lookup file.

## Model directory

The engine loads the model from `./saved_model` by default, so the saved-model folder in
this repo must be named `saved_model/` and sit next to `live_soc_simulation.py`. It holds
`lightgbm_model.joblib`, `graphite_embedder.joblib`, and `model_meta.json`. All commands
below use the default, matching how the tool is run in practice with no extra flags.

## Three run modes

The engine has three modes. Only the live mode needs the full capture stack; the other
two run from the model and the dataset alone, which makes them the ones an examiner can
actually reproduce.

### 1. Live mode (default)

Polls Elasticsearch for new SilkETW events, builds the streaming graph, and classifies a
process once its graphlet reaches the trigger size.

```bash
python live_soc_simulation.py
```

With the options you are likely to use in a walkthrough (smaller trigger for a faster
alert, the live dashboard, and a watched PID):

```bash
python live_soc_simulation.py --trigger-nodes 200 --dashboard --watch-pid 5380
```

The dashboard then serves at `http://localhost:8050`. This mode needs SilkETW, Logstash,
and Elasticsearch running (see the deployment section).

### 2. Dataset-test mode

Loads the real test graphlets and runs them straight through the embedder, LightGBM, and
SHAP. No Elasticsearch or Windows VM needed. This is the highest-fidelity RQ3 and RQ4
evidence, because it proves detection and latency on genuine graphlets rather than
synthetic ones.

```bash
python live_soc_simulation.py --dataset-test --dataset-path ../dataset --num-samples 30
```

This mode imports `dataprocessor_graphs.py`, so that file must be importable. It lives in
`core/`, so either copy `dataprocessor_graphs.py` into this folder or run with
`PYTHONPATH=../core`.

### 3. Dry-run mode

Synthesises ETW events that mimic the top-SHAP malware patterns (registry enumeration,
DLL injection, a C2 callout) and a benign pattern, then runs them through the full
streaming pipeline. No Elasticsearch or Windows VM needed. Useful for proving the plumbing
end to end.

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
rather than to the SilkETW consumer, which matters for per-process analysis.

### Critical SilkETW flags

The model was trained on the full set of FileIO sub-events and ImageLoad events. The
default SilkETW kernel flag set emits only `FileIo/OperationEnd` and no sub-events, and
detection will fail. Capture with the full set, including `FileIOInit` and `ImageLoad`:

```
SilkETW.exe -t kernel \
    -kk Process,Thread,FileIO,FileIOInit,Registry,NetworkTCPIP,ImageLoad \
    -ot file -p C:\path\to\etw.json
```

The captured events are then forwarded as JSON to the Logstash input on the Kali VM. The
full step-by-step deployment, including the event forwarder, is printed by:

```bash
python live_soc_simulation.py --help-full
```

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

The input opens an HTTP listener on port 5044 and reads each event as JSON. The filter is
left empty on purpose, all grouping and noise handling is done inside
`live_soc_simulation.py` after the events are read back from Elasticsearch. The output
writes every event into the `etw-live-logs` index. The host and index are set for a local
lab, so edit those two values to point at a different Elasticsearch instance.

## How live_soc_simulation.py works

Once events are indexed, the engine runs this loop:

1. **Poll**: read the newest batch of events from the `etw-live-logs` index.
2. **Translate**: convert each SilkETW event name into the Graphite vocabulary using the
   built-in mapping tables, falling back to the opcode name for registry events.
3. **Build**: update the streaming computation graph, applying the author's node UID rules
   and edge-direction rules, and track each process tree back to its root.
4. **Trigger**: once a process root accumulates at least `--trigger-nodes` nodes (default
   1000), project its graphlet.
5. **Embed and classify**: embed the graphlet with `graphite_embedder.joblib` and score it
   with `lightgbm_model.joblib`.
6. **Explain**: run SHAP on flagged processes, ranking only the N-gram features that
   actually occurred in that graphlet.
7. **Map and alert**: translate the top drivers into MITRE ATT&CK techniques, build a
   short behavioural profile, and print a structured SOC alert (and push it to the
   dashboard if enabled).

To keep alerts readable, the engine whitelists common Windows system processes, holds a
verdict as malware once a process has been flagged so it does not flip back to benign when
later background events dilute the graphlet, and shows at most two full alert boxes per
process before switching to one-line updates.

## Key flags

| Flag | Default | Purpose |
|------|---------|---------|
| `--model-dir` | `./saved_model` | Folder with the saved model (default is used here). |
| `--trigger-nodes` | `1000` | Nodes a process root must reach before it is scored. |
| `--es-host` | `http://localhost:9200` | Elasticsearch URL (live mode). |
| `--es-index` | `etw-live-logs` | Elasticsearch index (live mode). |
| `--poll-interval` | `0.5` | Seconds between Elasticsearch polls. |
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
  5,000 ms budget. Latency is measured from the point events are read out of Elasticsearch
  through to the generated alert (RQ4).
- Alerts are suppressed below a 0.70 malware confidence gate in live mode to reduce false
  positives. Note this 0.70 gate differs from the 75% figure stated in the thesis text,
  worth reconciling before the viva.
