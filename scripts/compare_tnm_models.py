#!/usr/bin/env python3
"""
Compare model performance: Combined TNMstage vs Separate T, N, M components.

This script:
1. Loads SEER data and adds T, N, M components
2. Builds Cox models with both approaches
3. Compares discrimination (C-index) on validation set
4. Tests external validation on hospital data
"""

import pandas as pd
import numpy as np
from pathlib import Path
from lifelines import CoxPHFitter
import json
import warnings
warnings.filterwarnings("ignore")

# Paths
PROJECT_DIR = Path("/Users/stan/Documents/UOM/myn_project")
DATA_DIR = PROJECT_DIR / "data/processed"
HOSPITAL_PATH = PROJECT_DIR / "ACC数据/ACC随访资料需完善2025.2.28.xlsx"
RAW_SEER_PATH = PROJECT_DIR / "ACC数据/r分析seer/SEER纯ACC数据.xlsx"

print("=" * 70)
print("MODEL COMPARISON: Combined TNMstage vs Separate T, N, M")
print("=" * 70)

# =============================================================================
# 1. Load and prepare SEER data with T, N, M
# =============================================================================
print("\n" + "-" * 70)
print("1. LOADING SEER DATA WITH T, N, M COMPONENTS")
print("-" * 70)

# Load processed SEER data
trackA_train = pd.read_pickle(DATA_DIR / "trackA_train.pkl")
trackA_val = pd.read_pickle(DATA_DIR / "trackA_val.pkl")

# Load raw SEER to get T, N, M
df_raw = pd.read_excel(RAW_SEER_PATH)

# Create ID to T, N, M mapping
tnm_map = df_raw[['编号', 'T', 'N', 'M']].copy()
tnm_map.columns = ['ID', 'T', 'N', 'M']

# Merge T, N, M into train and val
trackA_train = trackA_train.merge(tnm_map, on='ID', how='left')
trackA_val = trackA_val.merge(tnm_map, on='ID', how='left')

print(f"Training set: {len(trackA_train)} records")
print(f"Validation set: {len(trackA_val)} records")

# Check T, N, M completeness in Track A
print(f"\nT, N, M completeness in Track A training:")
for col in ['T', 'N', 'M']:
    non_null = trackA_train[col].notna().sum()
    print(f"  {col}: {non_null}/{len(trackA_train)} ({non_null/len(trackA_train)*100:.1f}%)")

# =============================================================================
# 2. Prepare T, N, M categories
# =============================================================================
print("\n" + "-" * 70)
print("2. T, N, M CATEGORY DISTRIBUTIONS")
print("-" * 70)

# Clean and standardize T, N, M values
def clean_tnm_value(val, component):
    """Clean T, N, M values for modeling."""
    if pd.isna(val):
        return f"{component}X"  # Unknown
    val = str(val).strip().upper()
    if val in ['', 'NAN', 'NA']:
        return f"{component}X"
    return val

for col in ['T', 'N', 'M']:
    trackA_train[col] = trackA_train[col].apply(lambda x: clean_tnm_value(x, col[0]))
    trackA_val[col] = trackA_val[col].apply(lambda x: clean_tnm_value(x, col[0]))

print("\nTraining set distributions:")
for col in ['T', 'N', 'M']:
    print(f"\n{col}:")
    print(trackA_train[col].value_counts().to_string())

# =============================================================================
# 3. Build Cox Models - Helper Functions
# =============================================================================
def build_cox_model(train_data, variables, time_col, event_col):
    """Build Cox model with specified variables."""
    model_df = train_data[[time_col, event_col]].copy()
    feature_names = []

    for var in variables:
        if var not in train_data.columns:
            continue
        dummies = pd.get_dummies(train_data[var].astype(str), prefix=var, drop_first=True, dtype=float)
        for col in dummies.columns:
            model_df[col] = dummies[col].values
            feature_names.append(col)

    model_df = model_df.dropna()
    cph = CoxPHFitter(penalizer=0.01)
    cph.fit(model_df, duration_col=time_col, event_col=event_col)

    return cph, feature_names, model_df

def prepare_data_for_scoring(data, variables, time_col, event_col, expected_features):
    """Prepare data for model scoring with proper feature alignment."""
    model_df = data[[time_col, event_col]].copy()

    for var in variables:
        if var not in data.columns:
            continue
        var_features = [f for f in expected_features if f.startswith(f"{var}_")]
        for feat in var_features:
            model_df[feat] = 0.0
        for idx in data.index:
            val = str(data.loc[idx, var])
            feature_name = f"{var}_{val}"
            if feature_name in var_features:
                model_df.loc[idx, feature_name] = 1.0

    for feature in expected_features:
        if feature not in model_df.columns:
            model_df[feature] = 0.0

    ordered_cols = [time_col, event_col] + expected_features
    model_df = model_df[ordered_cols]
    model_df = model_df.dropna()
    return model_df

# =============================================================================
# 4. Build and Compare OS Models
# =============================================================================
print("\n" + "-" * 70)
print("3. OS MODEL COMPARISON")
print("-" * 70)

# Current model variables (with combined TNMstage)
current_vars = ['TNMstage', 'age', 'chemotherapy', 'radiotherapy', 'marital_status', 'grade']

# Alternative model variables (with separate T, N, M)
separate_vars = ['T', 'N', 'M', 'age', 'chemotherapy', 'radiotherapy', 'marital_status', 'grade']

# Build current model (combined TNMstage)
print("\n--- Model A: Combined TNMstage ---")
cph_combined, features_combined, _ = build_cox_model(
    trackA_train, current_vars, 'time_os', 'event_os'
)
print(f"Training C-index: {cph_combined.concordance_index_:.4f}")

# Validate on SEER validation set
val_combined = prepare_data_for_scoring(
    trackA_val, current_vars, 'time_os', 'event_os', features_combined
)
cindex_combined_val = cph_combined.score(val_combined, scoring_method='concordance_index')
print(f"Validation C-index: {cindex_combined_val:.4f}")

# Build separate T, N, M model
print("\n--- Model B: Separate T, N, M ---")
cph_separate, features_separate, _ = build_cox_model(
    trackA_train, separate_vars, 'time_os', 'event_os'
)
print(f"Training C-index: {cph_separate.concordance_index_:.4f}")

# Validate on SEER validation set
val_separate = prepare_data_for_scoring(
    trackA_val, separate_vars, 'time_os', 'event_os', features_separate
)
cindex_separate_val = cph_separate.score(val_separate, scoring_method='concordance_index')
print(f"Validation C-index: {cindex_separate_val:.4f}")

# =============================================================================
# 5. External Validation on Hospital Data
# =============================================================================
print("\n" + "-" * 70)
print("4. EXTERNAL VALIDATION ON HOSPITAL DATA")
print("-" * 70)

# Load and prepare hospital data
from datetime import datetime
FOLLOWUP_END_DATE = datetime(2024, 10, 17)

df_main = pd.read_excel(HOSPITAL_PATH, sheet_name="需要搜集的数据", header=0)
df_tnm = pd.read_excel(HOSPITAL_PATH, sheet_name="筛选2", header=1)

# Normalize patient IDs
def normalize_patient_id(series):
    return series.astype(str).str.replace(r'\.0$', '', regex=True).str.strip()

if '住院号' in df_main.columns:
    df_main['住院号'] = normalize_patient_id(df_main['住院号'])
if '住院号' in df_tnm.columns:
    df_tnm['住院号'] = normalize_patient_id(df_tnm['住院号'])

# Clean IDs
df_main = df_main[df_main['住院号'].notna() & (df_main['住院号'] != 'nan')]
df_tnm = df_tnm[df_tnm['住院号'].notna() & (df_tnm['住院号'] != 'nan')]

# Merge
df_hospital = df_main.merge(df_tnm[['住院号', 'T', 'N', 'M', 'Clinical stage',
                                     '随访至今是否存活（2024/10/17）', '死亡', '手术时间']],
                            on='住院号', how='left')

print(f"Hospital data: {len(df_hospital)} records")

# Harmonize hospital data
def harmonize_hospital(df):
    """Harmonize hospital data to match SEER encoding."""
    result = pd.DataFrame()
    result['patient_id'] = df['住院号']

    # Age
    def map_age(val):
        if pd.isna(val): return np.nan
        try:
            age = float(val)
            if age < 45: return '＜45'
            elif age <= 59: return '45-59'
            else: return '＞60'
        except: return np.nan
    result['age'] = df['年龄'].apply(map_age) if '年龄' in df.columns else np.nan

    # TNMstage from Clinical stage
    def map_tnm_stage(val):
        if pd.isna(val): return '4NOS'
        s = str(val).strip().upper()
        stage_map = {
            'I': '1', '1': '1', 'II': '2', '2': '2', 'III': '3', '3': '3',
            'IVA': '4A', 'IV A': '4A', '4A': '4A',
            'IVB': '4B', 'IV B': '4B', '4B': '4B',
            'IVC': '4C', 'IV C': '4C', '4C': '4C',
            'IV': '4', '4': '4'
        }
        return stage_map.get(s, '4NOS')
    result['TNMstage'] = df['Clinical stage'].apply(map_tnm_stage) if 'Clinical stage' in df.columns else '4NOS'

    # Separate T, N, M
    def map_t(val):
        if pd.isna(val): return 'TX'
        s = str(val).strip().upper()
        if s in ['1', 'T1']: return 'T1'
        if s in ['2', 'T2']: return 'T2'
        if s in ['3', 'T3']: return 'T3'
        if s in ['4A', 'T4A']: return 'T4a'
        if s in ['4B', 'T4B']: return 'T4b'
        if s in ['4', 'T4']: return 'T4'
        if s in ['X', 'TX', '-']: return 'TX'
        return 'TX'
    result['T'] = df['T'].apply(map_t) if 'T' in df.columns else 'TX'

    def map_n(val):
        if pd.isna(val): return 'NX'
        s = str(val).strip().upper()
        if s in ['0', 'N0']: return 'N0'
        if s in ['1', 'N1']: return 'N1'
        if s in ['2A', 'N2A']: return 'N2a'
        if s in ['2B', 'N2B']: return 'N2b'
        if s in ['2C', 'N2C']: return 'N2c'
        if s in ['2', 'N2']: return 'N2'
        if s in ['3B', 'N3B']: return 'N3b'
        if s in ['X', 'NX', '-']: return 'NX'
        return 'NX'
    result['N'] = df['N'].apply(map_n) if 'N' in df.columns else 'NX'

    def map_m(val):
        if pd.isna(val): return 'MX'
        s = str(val).strip().upper().replace('？', '')
        if s in ['0', 'M0']: return 'M0'
        if s in ['1', 'M1']: return 'M1'
        if s in ['X', 'MX', '-']: return 'MX'
        return 'MX'
    result['M'] = df['M'].apply(map_m) if 'M' in df.columns else 'MX'

    # Grade
    def map_grade(val):
        if pd.isna(val): return '不明'
        s = str(val).strip()
        if '未分化' in s or '间变' in s or '4' in s: return '4未分化间变性'
        if 'III' in s or '低分化' in s or '差' in s: return '3分化差'
        if 'II' in s or '中分化' in s: return '2中分化'
        if 'I' in s or '高分化' in s or '好' in s: return '1分化好'
        return '不明'
    result['grade'] = df['组织学分级（高低分化）'].apply(map_grade) if '组织学分级（高低分化）' in df.columns else '不明'

    # Marital status
    def map_marital(val):
        if pd.isna(val): return '未知'
        s = str(val).strip()
        if '已婚' in s: return '已婚'
        if '未婚' in s and '同居' not in s: return '未婚'
        if '离' in s: return '离异'
        if '丧偶' in s: return '丧偶'
        if '分居' in s: return '分居'
        if '同居' in s: return '同居未婚'
        return '未知'
    result['marital_status'] = df['婚姻'].apply(map_marital) if '婚姻' in df.columns else '未知'

    # Treatment
    treatment_text = df['治疗方式'].fillna('').astype(str) if '治疗方式' in df.columns else pd.Series([''] * len(df), index=df.index)

    def map_radiotherapy(row_idx):
        val = df.loc[row_idx, '放疗'] if '放疗' in df.columns else None
        treat = treatment_text.loc[row_idx] if row_idx in treatment_text.index else ''
        if pd.isna(val): val = ''
        s = str(val).strip()
        if s in ['1', '1.0', '是', '有'] or '放疗' in treat: return 'Yes'
        return 'No/Unknow'
    result['radiotherapy'] = [map_radiotherapy(i) for i in df.index]

    def map_chemotherapy(row_idx):
        val = df.loc[row_idx, '化疗（术前？术后？）'] if '化疗（术前？术后？）' in df.columns else None
        treat = treatment_text.loc[row_idx] if row_idx in treatment_text.index else ''
        if pd.isna(val): val = ''
        s = str(val).strip()
        if s in ['/', '0', '否', '无', '', 'nan']: return 'No/Unknown'
        if s or '化疗' in treat: return 'Yes'
        return 'No/Unknown'
    result['chemotherapy'] = [map_chemotherapy(i) for i in df.index]

    # OS calculation from dates
    surgery_dates = pd.to_datetime(df.get('手术时间'), errors='coerce')
    death_dates = pd.to_datetime(df.get('死亡'), errors='coerce')
    survival_status = df.get('随访至今是否存活（2024/10/17）')

    time_os_list = []
    event_os_list = []

    for idx in df.index:
        surgery_date = surgery_dates.loc[idx] if idx in surgery_dates.index else pd.NaT
        death_date = death_dates.loc[idx] if idx in death_dates.index else pd.NaT
        status = survival_status.loc[idx] if survival_status is not None and idx in survival_status.index else np.nan

        if pd.isna(surgery_date):
            time_os_list.append(np.nan)
            event_os_list.append(np.nan)
            continue

        try:
            status_val = float(status)
        except:
            status_val = np.nan

        if status_val == 0:  # Dead
            if pd.notna(death_date) and death_date.year > 1980:
                os_months = (death_date - surgery_date).days / 30.44
            else:
                os_months = (FOLLOWUP_END_DATE - surgery_date).days / 30.44
            time_os_list.append(os_months)
            event_os_list.append(1)
        elif status_val == 1:  # Alive
            os_months = (FOLLOWUP_END_DATE - surgery_date).days / 30.44
            time_os_list.append(os_months)
            event_os_list.append(0)
        else:
            time_os_list.append(np.nan)
            event_os_list.append(np.nan)

    result['time_os'] = time_os_list
    result['event_os'] = event_os_list

    return result

# Harmonize hospital data
df_harmonized = harmonize_hospital(df_hospital)
df_harmonized = df_harmonized.dropna(subset=['time_os', 'event_os'])
df_harmonized = df_harmonized[df_harmonized['time_os'] > 0]

print(f"\nHarmonized hospital data: {len(df_harmonized)} records")
print(f"OS events: {int(df_harmonized['event_os'].sum())} ({df_harmonized['event_os'].mean()*100:.1f}%)")

# Check T, N, M distributions in hospital
print("\nHospital T, N, M distributions:")
for col in ['T', 'N', 'M']:
    print(f"\n{col}:")
    print(df_harmonized[col].value_counts().to_string())

# External validation - Combined TNMstage model
print("\n--- External Validation: Combined TNMstage Model ---")
hospital_combined = prepare_data_for_scoring(
    df_harmonized, current_vars, 'time_os', 'event_os', features_combined
)
if len(hospital_combined) > 10:
    cindex_combined_ext = cph_combined.score(hospital_combined, scoring_method='concordance_index')
    print(f"Hospital C-index: {cindex_combined_ext:.4f}")
    print(f"N evaluated: {len(hospital_combined)}")
else:
    cindex_combined_ext = np.nan
    print("Insufficient data for evaluation")

# External validation - Separate T, N, M model
print("\n--- External Validation: Separate T, N, M Model ---")
hospital_separate = prepare_data_for_scoring(
    df_harmonized, separate_vars, 'time_os', 'event_os', features_separate
)
if len(hospital_separate) > 10:
    cindex_separate_ext = cph_separate.score(hospital_separate, scoring_method='concordance_index')
    print(f"Hospital C-index: {cindex_separate_ext:.4f}")
    print(f"N evaluated: {len(hospital_separate)}")
else:
    cindex_separate_ext = np.nan
    print("Insufficient data for evaluation")

# =============================================================================
# 6. Summary Comparison
# =============================================================================
print("\n" + "=" * 70)
print("5. SUMMARY COMPARISON")
print("=" * 70)

print("""
╔═══════════════════════════════════════════════════════════════════════╗
║                    OS MODEL COMPARISON RESULTS                        ║
╠═══════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║  Model                    │ SEER Train │ SEER Val │ Hospital Ext     ║
║  ─────────────────────────┼────────────┼──────────┼──────────────────║""")
print(f"║  A: Combined TNMstage    │   {cph_combined.concordance_index_:.4f}   │  {cindex_combined_val:.4f}  │     {cindex_combined_ext:.4f}         ║")
print(f"║  B: Separate T, N, M     │   {cph_separate.concordance_index_:.4f}   │  {cindex_separate_val:.4f}  │     {cindex_separate_ext:.4f}         ║")
print("║                                                                       ║")

diff_val = cindex_separate_val - cindex_combined_val
diff_ext = cindex_separate_ext - cindex_combined_ext if not np.isnan(cindex_separate_ext) else np.nan

print(f"║  Difference (B - A)      │     -      │  {diff_val:+.4f}  │    {diff_ext:+.4f}          ║")
print("║                                                                       ║")
print("╚═══════════════════════════════════════════════════════════════════════╝")

print("\n" + "-" * 70)
print("INTERPRETATION:")
print("-" * 70)
if diff_val > 0.02:
    print("→ Separate T, N, M IMPROVES SEER validation by >{:.1f}%".format(diff_val*100))
elif diff_val < -0.02:
    print("→ Separate T, N, M WORSENS SEER validation by {:.1f}%".format(abs(diff_val)*100))
else:
    print("→ Similar performance on SEER validation (difference < 2%)")

if not np.isnan(diff_ext):
    if diff_ext > 0.02:
        print("→ Separate T, N, M IMPROVES hospital external validation by >{:.1f}%".format(diff_ext*100))
    elif diff_ext < -0.02:
        print("→ Separate T, N, M WORSENS hospital external validation by {:.1f}%".format(abs(diff_ext)*100))
    else:
        print("→ Similar performance on hospital external validation (difference < 2%)")

print("\n" + "-" * 70)
print("RECOMMENDATION:")
print("-" * 70)
if diff_val > 0.01 and (np.isnan(diff_ext) or diff_ext > -0.02):
    print("Consider using separate T, N, M - shows improvement without hurting external validation.")
elif diff_val < -0.01:
    print("Keep combined TNMstage - separate components show worse performance.")
else:
    print("Either approach is reasonable - similar performance observed.")
    print("Combined TNMstage is simpler and more clinically interpretable.")
