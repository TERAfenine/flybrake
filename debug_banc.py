
import pandas as pd

meta = pd.read_feather(
    "data/banc_888_meta.feather"
)
nt = pd.read_csv(
    "data/banc_888_neurotransmitter_prediction_v2.csv"
)

print("=== 1. 전체 데이터 ===")
print("뉴런 메타데이터:", len(meta))
print("신경전달물질 데이터:", len(nt))

print("\n=== 2. 뇌 영역 ===")
print(meta["region"].value_counts().head(10))

print("\n=== 3. 뉴런 종류 ===")
print(meta["super_class"].value_counts().head(10))

print("\n=== 4. 신경전달물질 종류 ===")
print(nt["neurotransmitter_predicted"].value_counts().head(10))

print("\n=== 5. 조건별 남는 뉴런 수 ===")
m1 = meta[meta["region"] == "ventral_nerve_cord"]
m2 = m1[m1["super_class"] == "ventral_nerve_cord_intrinsic"]

n1 = nt[nt["neurotransmitter_predicted"].isin(
    ["acetylcholine", "gaba"]
)]
n2 = n1[n1["neurotransmitter_score"] >= 0.8]
n3 = n2[n2["count"] >= 20]

print("VNC 영역:", len(m1))
print("VNC + intrinsic:", len(m2))
print("ACh 또는 GABA:", len(n1))
print("예측 점수 0.8 이상:", len(n2))
print("시냅스 수 20 이상:", len(n3))

merged = m2.merge(
    n3,
    left_on="banc_888_id",
    right_on="root_id"
)

print("최종 일치 뉴런:", len(merged))
