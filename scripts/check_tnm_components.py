#!/usr/bin/env python3
"""
Check T, N, M component availability in hospital and SEER data.
Analyze whether separating TNM components could improve model performance.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# Paths
PROJECT_DIR = Path("/Users/stan/Documents/UOM/myn_project")
DATA_DIR = PROJECT_DIR / "data/processed"
HOSPITAL_PATH = PROJECT_DIR / "ACC数据/ACC随访资料需完善2025.2.28.xlsx"

print("=" * 70)
print("T, N, M COMPONENT ANALYSIS")
print("=" * 70)

# =============================================================================
# 1. Check Hospital Data T, N, M
# =============================================================================
print("\n" + "=" * 70)
print("1. HOSPITAL DATA: T, N, M COMPLETENESS")
print("=" * 70)

# Load hospital TNM sheet
df_tnm = pd.read_excel(HOSPITAL_PATH, sheet_name="筛选2", header=1)
print(f"\nHospital TNM sheet loaded: {df_tnm.shape}")

# Check T, N, M columns
for col in ['T', 'N', 'M']:
    if col in df_tnm.columns:
        total = len(df_tnm)
        non_null = df_tnm[col].notna().sum()
        non_empty = df_tnm[col].apply(lambda x: pd.notna(x) and str(x).strip() not in ['', 'nan', 'NaN']).sum()
        print(f"\n{col} column:")
        print(f"  Total records: {total}")
        print(f"  Non-null: {non_null} ({non_null/total*100:.1f}%)")
        print(f"  Non-empty: {non_empty} ({non_empty/total*100:.1f}%)")
        print(f"  Distribution:")
        print(df_tnm[col].value_counts(dropna=False).head(10).to_string())
    else:
        print(f"\n{col} column: NOT FOUND")

# Check Clinical stage column
if 'Clinical stage' in df_tnm.columns:
    print("\nClinical stage column:")
    print(df_tnm['Clinical stage'].value_counts(dropna=False).to_string())

# =============================================================================
# 2. Check SEER Data T, N, M
# =============================================================================
print("\n" + "=" * 70)
print("2. SEER DATA: T, N, M DISTRIBUTIONS")
print("=" * 70)

trackA_train = pd.read_pickle(DATA_DIR / "trackA_train.pkl")
print(f"\nSEER Track A training data: {trackA_train.shape}")
print(f"Columns: {trackA_train.columns.tolist()}")

# Check if T, N, M are in SEER processed data
for col in ['T', 'N', 'M']:
    if col in trackA_train.columns:
        print(f"\n{col} distribution in SEER:")
        print(trackA_train[col].value_counts(dropna=False).to_string())
    else:
        print(f"\n{col}: NOT in processed SEER data")

# =============================================================================
# 3. Load Raw SEER to check T, N, M
# =============================================================================
print("\n" + "=" * 70)
print("3. RAW SEER DATA: T, N, M CHECK")
print("=" * 70)

raw_seer_path = PROJECT_DIR / "ACC数据/r分析seer/SEER纯ACC数据.xlsx"
if raw_seer_path.exists():
    df_raw = pd.read_excel(raw_seer_path)
    print(f"\nRaw SEER data loaded: {df_raw.shape}")

    # Check T, N, M columns
    for col in ['T', 'N', 'M']:
        if col in df_raw.columns:
            print(f"\n{col} distribution in raw SEER:")
            vc = df_raw[col].value_counts(dropna=False)
            print(vc.head(15).to_string())
            print(f"  Total categories: {len(vc)}")
        else:
            print(f"\n{col}: NOT FOUND in raw SEER")

# =============================================================================
# 4. Comparison Summary
# =============================================================================
print("\n" + "=" * 70)
print("4. SUMMARY: CAN WE USE SEPARATE T, N, M?")
print("=" * 70)

# Check hospital completeness
hospital_t_complete = df_tnm['T'].notna().sum() if 'T' in df_tnm.columns else 0
hospital_n_complete = df_tnm['N'].notna().sum() if 'N' in df_tnm.columns else 0
hospital_m_complete = df_tnm['M'].notna().sum() if 'M' in df_tnm.columns else 0
hospital_total = len(df_tnm)

print(f"\nHospital T, N, M completeness:")
print(f"  T: {hospital_t_complete}/{hospital_total} ({hospital_t_complete/hospital_total*100:.1f}%)")
print(f"  N: {hospital_n_complete}/{hospital_total} ({hospital_n_complete/hospital_total*100:.1f}%)")
print(f"  M: {hospital_m_complete}/{hospital_total} ({hospital_m_complete/hospital_total*100:.1f}%)")

# Check if SEER has T, N, M
seer_has_tnm = all(col in trackA_train.columns for col in ['T', 'N', 'M'])
print(f"\nSEER processed data has T, N, M: {seer_has_tnm}")

if hospital_t_complete > 0 and hospital_n_complete > 0 and hospital_m_complete > 0:
    print("\n✓ Hospital data HAS T, N, M components")
    print("  → Separate T, N, M analysis is FEASIBLE")
else:
    print("\n✗ Hospital data is MISSING T, N, M components")
    print("  → Separate T, N, M analysis would NOT help external validation")
