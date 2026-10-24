# FlyBrake: BANC v888 connectome-constrained warning AI (starter project)

## What it does
- **User-downloaded 2026 BANC v888** directed synapse adjacency is fixed as a neural network mask.
- Neurotransmitter predictions: GABA presynaptic neurons get **negative** connections, acetylcholine neurons **positive** connections. **This is an engineering simplification**: transmitter alone does not establish the actual postsynaptic effect.
- Trains a small recurrent neural network using an independent room occupancy dataset (office data, **not school data**).
- Compares (A) real-connectome signed model, (B) same-degree rewired signed control, (C) real-connectome unsigned ablation, (D) logistic regression baseline.
- Compares sensor corruption tests and optional abstention safety gate.
- Streamlit demo reports only **suggestions and abstention**, never controls a real device.

## Setup
Use Python 3.10+ in a new virtual environment:
```
python -m pip install -r requirements.txt
python smoke_test.py
```

## Get original data (no fabricated BANC graph)
Download BANC v888 from [Harvard Dataverse](https://doi.org/10.7910/DVN/7WTH1N).
Required files:
- `banc_888_meta.feather` (~49 MB)
- `banc_888_edgelist_simple_v2.feather` (~285 MB)
- `banc_888_neurotransmitter_prediction_v2.csv` (~21 MB)

Reference schema: [BANC Dataset Documentation](https://github.com/sjcabs/fly_connectome_data_tutorial/blob/main/data/dataset_documentation/banc_data.md).

Download UCI [Occupancy Detection ID 357](https://archive.ics.uci.edu/dataset/357/occupancy-detection) and unzip. It has `datatraining.txt`, `datatest.txt`, `datatest2.txt`.

Put all files under `data/` or adjust CLI paths.

## Build the real-BANC graph
```
python build_graph.py \
  --meta data/banc_888_meta.feather \
  --nt data/banc_888_neurotransmitter_prediction_v2.csv \
  --edges data/banc_888_edgelist_simple_v2.feather \
  --n 96 --out data/banc_graph.npz
```
The selection algorithm chooses a densely connected graph inside the ventral nerve cord from intrinsic neurons with confident GABA/ACh predictions. This is **not automatically a biological 'alarm-suppressing circuit'**. If it cannot find a sufficiently mixed/connected graph, it fails and the selected anatomical region/cell types need revision. Verify original file provenance and log selected neuron IDs from `banc_graph.npz` in the report. Full edgelist may require 1–2 GB RAM.

## Train, compare, and run the interface
```
python train.py --graph data/banc_graph.npz \
  --train data/datatraining.txt --test data/datatest2.txt
streamlit run app.py
```
View `results/metrics.csv` and `results/experiment_info.json` for actual measured values **after training**. The file contains separate clean / simulated sensor fault measurements and safety-gate on/off conditions.

## Scientific controls and safety
- Real BANC vs degree-preserved rewired controls isolates *some* connectivity structure effect.
- Signed vs unsigned BANC is an ablation of the **GABA sign engineering assumption**, not a test of true receptor physiology.
- The logistic regression baseline shows whether the fancy graph model adds value vs a simple algorithm.
- Report clean test performance, false alarms (empty room predicted occupied), unsafe false-empty rate (occupied room predicted empty), F1 on decided samples, **abstention rate and coverage**. Low false alarm rate due only to abstentions is **not** an unqualified improvement.
- Original occupancy dataset uses office measurements and photographic labels (no images are used for *model input*). We do not collect identity, personal records, or images.
- Each dataset is chronological, train/validation/test remain separate; test set is not used for tuning.
- Sensor failures (`missing_light`, `co2_spike`) are artificially injected into measured office data; they are not real accident labels.
- Thresholds (`0.2` and `0.8`) are **starting settings only**, not calibrated probability thresholds. Tune only on validation; evaluate on held-out test after freezing all choices.
- No claim of a certified safety device and no deployment in schools without separate validation and teacher approval.

## Known prototype limitations
1. We do **not** recreate receptor-specific neurotransmission, synaptic temporal delays, or detailed Drosophila physiology.
2. The input/readout mappings are engineered. BANC is used for hidden recurrent adjacency and predicted presynaptic transmitters only.
3. The automatically selected densely connected neighborhood may lack proven functional relevance to gating/alert suppression.
4. Real connectome topology is not guaranteed to outperform controls. A null result still answers the research question.
5. Leakage from temporal correlations, domain shift between office and school, experimental variance (run 3+ seeds), calibration and multiple-testing issues deserve careful study.

## Citation
Bates et al. (2026), *Nature*, "Distributed control circuits across a brain-and-cord connectome", https://doi.org/10.1038/s41586-026-10735-w.
Candanedo (2016), UCI Occupancy Detection, https://doi.org/10.24432/C5X01N.

### Provenance warning
The software verifies input columns and structure only, **not the external provenance** of source files. It does not automatically authenticate that the supplied input is from BANC. Save Dataverse download URLs, release number, file hash, and selected IDs in your project report.

## 2026-10-24 audit update
See [AUDIT_KO.md](AUDIT_KO.md) for measured verification, reproducibility limits, and revised commands. `banc_unsigned` remains the historical all-positive control; `banc_free` is the new unconstrained-sign control. Use `python run_5_seeds.py --extended-controls --outdir results/revised_native_5seeds`. Existing output folders cannot be overwritten. The dashboard reads measured CSVs and displays real supplied graph edges and neuron IDs. The extracted region is VNC, not the entire brain.
