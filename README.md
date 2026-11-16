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

## FlyBrake Lab — 1단계 (2026-11-02)

기존 홈 화면은 그대로 유지하며 Streamlit 사이드바의 **FlyBrake Lab** 페이지에서
저장된 결과 묶음·학습 시드·모델 A/B·센서 조건·안전장치를 선택해 비교합니다.
Lab의 시드 선택은 카드, 표, 그래프, 동일 coverage 분석에 모두 적용됩니다.
홈 화면의 기존 시드 선택은 여전히 모델/그래프 확인용입니다. 시드별 성능 비교는 Lab을 이용하세요.
Lab은 체크포인트를 불러오거나 학습하지 않습니다.

PyCharm Terminal (PowerShell):

```powershell
cd D:\flybrake
.\.venv\Scripts\python.exe -m streamlit run app.py
```

새 환경에 Plotly만 빠졌다면 기존 패키지를 업그레이드하지 않고 추가합니다.

```powershell
.\.venv\Scripts\python.exe -m pip install --no-deps plotly==5.24.1
.\.venv\Scripts\python.exe -m pip check
```

테스트:

```powershell
.\.venv\Scripts\python.exe -m unittest test_research test_lab_results test_lab_ui -v
.\.venv\Scripts\python.exe smoke_test.py
```

- `src/lab_results.py`: 읽기 전용 결과 검색, 스키마 검사, 필터와 표시용 집계.
- `src/lab_charts.py`: 카드/표와 같은 수치로 Plotly 그림 생성.
- `pages/1_FlyBrake_Lab.py`: 독립 멀티페이지 UI.
- `test_lab_results.py`, `test_lab_ui.py`: 실제 CSV 수치 및 화면 조작 회귀 검사.

묶음의 `metrics_5seeds.csv`가 있으면 그것만 사용하고, 없으면 해당 묶음의 `seed*/metrics.csv`를 읽습니다.
요약 CSV를 원시 측정값으로 취급하지 않습니다. 누락된 지표는 N/A, 잘못된 필수 열·중복 행은 안내로 표시합니다.
일부 시드만 있는 조건은 전체 시드 평균으로 표시하지 않습니다. 평균의 표준편차는 ddof=1이며 신뢰구간이 아닙니다.
CSV 캐시는 경로·수정시각(ns)·크기로 구분하며, 사이드바의 새로 읽기 버튼으로 명시적으로 갱신할 수 있습니다.

동일 coverage는 `test_score_top_k_no_labels` 분석만 사용합니다. 검증 임계값 분석은 별도 접힌 섹션이며
안전장치 OFF 조건임을 명시합니다. 동일 coverage라도 선택된 표본이 같다는 뜻은 아닙니다.

## FlyBrake Lab — 2단계 뉴런 실험 (2026-11-02)

기존 실행 명령 `python -m streamlit run app.py`의 왼쪽 메뉴에서 **FlyBrake Neurons**를 선택합니다.
실제 연결망의 노드 클릭 또는 드롭다운으로 뉴런을 고르고, 단일 뉴런 비활성화 ON/OFF로 같은 입력의 모델 점수를 비교합니다.
각 4단계 재귀 계산 후 선택 뉴런의 활성값을 0으로 고정하며 가중치를 수정하거나 재학습하지 않습니다.
뉴런/체크포인트 선택을 바꾸면 비활성화는 OFF로 초기화됩니다.

현재 Python 3.11 환경에서는 **기존 결과 · results**의 시드 42~46을 사용하세요.
`revised_audit_20261024`는 전처리 저장 버전 차이로 그래프 탐색만 가능하며, 추론은 호환 환경이 필요합니다.
단일 입력의 점수 변화는 전체 테스트 성능이나 생물학적 기능을 입증하지 않습니다.

```powershell
.\.venv\Scripts\python.exe -m unittest test_research test_lab_results test_lab_ui test_lab_ablation test_lab_neurons_ui -v
```

구현·검증·제한 사항은 [LAB_PHASE2_REPORT_KO.md](LAB_PHASE2_REPORT_KO.md)에 기록했습니다.
