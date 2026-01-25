#!/usr/bin/env python3
"""
06_external_validation.py - External validation on hospital cohort.

Input: Fitted models and hospital data
Output: External validation metrics, comparison plots
Pos: Sixth step - tests model transportability to independent cohort

Usage: python scripts/06_external_validation.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter, KaplanMeierFitter
from sklearn.metrics import roc_curve, auc
from datetime import datetime
import pickle
import json
import warnings
warnings.filterwarnings("ignore")

from config import (
    DATA_DIR, MODELS_DIR, TABLES_DIR, ROC_DIR, CALIBRATION_DIR, COMPARISON_DIR,
    HOSPITAL_PATH, FOLLOWUP_END_DATE,
    TIME_POINTS, TIME_LABELS, FIGURE_DPI, COLORS
)


def normalize_patient_id(series):
    """Normalize patient IDs for merging."""
    return series.astype(str).str.replace(r'\.0$', '', regex=True).str.strip()


def harmonize_hospital_data(df):
    """Harmonize hospital data to match SEER encoding."""
    result = pd.DataFrame()
    result['patient_id'] = df['住院号']

    # Age groups
    def map_age(val):
        if pd.isna(val):
            return np.nan
        try:
            age = float(val)
            if age < 45:
                return '＜45'
            elif age <= 59:
                return '45-59'
            else:
                return '＞60'
        except:
            return np.nan

    result['age'] = df['年龄'].apply(map_age) if '年龄' in df.columns else np.nan

    # T stage
    def map_t(val):
        if pd.isna(val):
            return 'TX'
        s = str(val).strip().upper()
        if s in ['1', 'T1']:
            return 'T1'
        if s in ['2', 'T2']:
            return 'T2'
        if s in ['3', 'T3']:
            return 'T3'
        if s in ['4A', 'T4A']:
            return 'T4a'
        if s in ['4B', 'T4B']:
            return 'T4b'
        if s in ['4', 'T4']:
            return 'T4'
        return 'TX'

    result['T'] = df['T'].apply(map_t) if 'T' in df.columns else 'TX'

    # N stage
    def map_n(val):
        if pd.isna(val):
            return 'NX'
        s = str(val).strip().upper()
        if s in ['0', 'N0']:
            return 'N0'
        if s in ['1', 'N1']:
            return 'N1'
        if s in ['2A', 'N2A']:
            return 'N2a'
        if s in ['2B', 'N2B']:
            return 'N2b'
        if s in ['2C', 'N2C']:
            return 'N2c'
        if s in ['2', 'N2']:
            return 'N2'
        if s in ['3B', 'N3B']:
            return 'N3b'
        return 'NX'

    result['N'] = df['N'].apply(map_n) if 'N' in df.columns else 'NX'

    # M stage
    def map_m(val):
        if pd.isna(val):
            return 'MX'
        s = str(val).strip().upper().replace('？', '')
        if s in ['0', 'M0']:
            return 'M0'
        if s in ['1', 'M1']:
            return 'M1'
        return 'MX'

    result['M'] = df['M'].apply(map_m) if 'M' in df.columns else 'MX'

    # Grade
    def map_grade(val):
        if pd.isna(val):
            return '不明'
        s = str(val).strip()
        if '未分化' in s or '间变' in s or '4' in s:
            return '4未分化间变性'
        if 'III' in s or '低分化' in s or '差' in s:
            return '3分化差'
        if 'II' in s or '中分化' in s:
            return '2中分化'
        if 'I' in s or '高分化' in s or '好' in s:
            return '1分化好'
        return '不明'

    result['grade'] = df['组织学分级（高低分化）'].apply(map_grade) if '组织学分级（高低分化）' in df.columns else '不明'

    # Marital status
    def map_marital(val):
        if pd.isna(val):
            return '未知'
        s = str(val).strip()
        if '已婚' in s:
            return '已婚'
        if '未婚' in s and '同居' not in s:
            return '未婚'
        if '离' in s:
            return '离异'
        if '丧偶' in s:
            return '丧偶'
        if '分居' in s:
            return '分居'
        if '同居' in s:
            return '同居未婚'
        return '未知'

    result['marital_status'] = df['婚姻'].apply(map_marital) if '婚姻' in df.columns else '未知'

    # Tumor number
    def map_tumor_number(val):
        if pd.isna(val):
            return '01'
        s = str(val).strip()
        if '多' in s or '>' in s or '＞' in s:
            return '＞1'
        try:
            if float(s) > 1:
                return '＞1'
        except:
            pass
        return '01'

    result['tumor_number'] = df['原发肿瘤数量（单发？多发）'].apply(map_tumor_number) if '原发肿瘤数量（单发？多发）' in df.columns else '01'

    # Treatment variables
    treatment_text = df['治疗方式'].fillna('').astype(str) if '治疗方式' in df.columns else pd.Series([''] * len(df), index=df.index)

    def map_radiotherapy(row_idx):
        val = df.loc[row_idx, '放疗'] if '放疗' in df.columns else None
        treat = treatment_text.loc[row_idx] if row_idx in treatment_text.index else ''
        if pd.isna(val):
            val = ''
        s = str(val).strip()
        if s in ['1', '1.0', '是', '有'] or '放疗' in treat:
            return 'Yes'
        return 'No/Unknow'

    result['radiotherapy'] = [map_radiotherapy(i) for i in df.index]

    def map_chemotherapy(row_idx):
        val = df.loc[row_idx, '化疗（术前？术后？）'] if '化疗（术前？术后？）' in df.columns else None
        treat = treatment_text.loc[row_idx] if row_idx in treatment_text.index else ''
        if pd.isna(val):
            val = ''
        s = str(val).strip()
        if s in ['/', '0', '否', '无', '', 'nan']:
            return 'No/Unknown'
        if s or '化疗' in treat:
            return 'Yes'
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

    # CSS (conservative - only count cancer deaths)
    result['time_css'] = result['time_os'].copy()

    def is_cancer_death(reason):
        if pd.isna(reason):
            return None
        s = str(reason)
        if '非' in s and ('癌' in s or '肿瘤' in s):
            return False
        if '癌' in s or '肿瘤' in s or '转移' in s:
            return True
        return None

    css_events = []
    death_reasons = df.get('死亡原因（需明确是否因为癌症死亡）', pd.Series([None] * len(df), index=df.index))
    for idx, ev_os in zip(df.index, result['event_os']):
        reason = death_reasons.loc[idx] if idx in death_reasons.index else None
        if pd.isna(ev_os) or ev_os == 0:
            css_events.append(0 if ev_os == 0 else np.nan)
        else:
            cancer = is_cancer_death(reason)
            css_events.append(1 if cancer else (0 if cancer is False else np.nan))
    result['event_css'] = css_events

    return result


def prepare_data_for_model(data, variables, time_col, event_col, feature_names):
    """Prepare data with proper feature alignment."""
    model_df = data[[time_col, event_col]].copy()

    for var in variables:
        if var not in data.columns:
            continue
        var_features = [f for f in feature_names if f.startswith(f'{var}_')]
        for feat in var_features:
            model_df[feat] = 0.0
        for idx in data.index:
            val = str(data.loc[idx, var])
            fname = f'{var}_{val}'
            if fname in var_features:
                model_df.loc[idx, fname] = 1.0

    for feat in feature_names:
        if feat not in model_df.columns:
            model_df[feat] = 0.0

    ordered_cols = [time_col, event_col] + feature_names
    model_df = model_df[ordered_cols]
    return model_df.dropna()


def calculate_time_dependent_auc(cph, data, time_col, event_col, time_point):
    """Calculate time-dependent ROC and AUC."""
    mask = (data[time_col] >= time_point) | ((data[time_col] < time_point) & (data[event_col] == 1))
    eval_data = data[mask].copy()

    if len(eval_data) < 20:
        return None, None, np.nan, 0

    eval_data['outcome'] = ((eval_data[time_col] <= time_point) & (eval_data[event_col] == 1)).astype(int)
    n_events = eval_data['outcome'].sum()

    if n_events < 3:
        return None, None, np.nan, n_events

    covariates = eval_data.drop([time_col, event_col, 'outcome'], axis=1)
    risk_scores = cph.predict_log_partial_hazard(covariates)

    fpr, tpr, _ = roc_curve(eval_data['outcome'], risk_scores)
    roc_auc = auc(fpr, tpr)

    return fpr, tpr, roc_auc, n_events


def calculate_calibration(cph, data, time_col, event_col, time_point, n_groups=5):
    """Calculate calibration."""
    baseline_surv = cph.baseline_survival_
    idx = np.abs(baseline_surv.index - time_point).argmin()
    s0_t = baseline_surv.iloc[idx, 0]

    covariates = data.drop([time_col, event_col], axis=1)
    linear_pred = cph.predict_log_partial_hazard(covariates)
    predicted_surv = s0_t ** np.exp(linear_pred)

    data_copy = data.copy()
    data_copy['predicted'] = predicted_surv.values

    try:
        data_copy['group'] = pd.qcut(data_copy['predicted'], q=n_groups, labels=False, duplicates='drop')
    except:
        return None

    results = []
    for group in sorted(data_copy['group'].unique()):
        group_data = data_copy[data_copy['group'] == group]
        mean_predicted = group_data['predicted'].mean()

        kmf = KaplanMeierFitter()
        kmf.fit(group_data[time_col], group_data[event_col])

        if time_point in kmf.survival_function_.index:
            observed = kmf.survival_function_.loc[time_point].values[0]
        else:
            idx_km = np.abs(kmf.survival_function_.index - time_point).argmin()
            observed = kmf.survival_function_.iloc[idx_km].values[0]

        results.append({
            'group': group,
            'n': len(group_data),
            'predicted': mean_predicted,
            'observed': observed
        })

    return pd.DataFrame(results)


def plot_comparison(internal_results, external_results, output_path):
    """Plot internal vs external validation comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    cohorts = ['SEER\nInternal', 'Hospital\nExternal']

    # OS C-index
    ax = axes[0, 0]
    os_vals = [internal_results['os']['c_index'], external_results['os']['c_index']]
    bars = ax.bar(cohorts, os_vals, color=[COLORS['primary'], COLORS['secondary']], alpha=0.8, edgecolor='black')
    ax.set_ylabel('C-index')
    ax.set_title('OS Model: C-index')
    ax.set_ylim(0.4, 0.9)
    ax.axhline(y=0.7, color='gray', linestyle='--', alpha=0.5, label='Good threshold')
    for bar, val in zip(bars, os_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f'{val:.3f}', ha='center', fontweight='bold')

    # CSS C-index
    ax = axes[0, 1]
    css_vals = [internal_results['css']['c_index'], external_results['css']['c_index']]
    bars = ax.bar(cohorts, css_vals, color=[COLORS['primary'], COLORS['secondary']], alpha=0.8, edgecolor='black')
    ax.set_ylabel('C-index')
    ax.set_title('CSS Model: C-index')
    ax.set_ylim(0.4, 0.9)
    ax.axhline(y=0.7, color='gray', linestyle='--', alpha=0.5)
    for bar, val in zip(bars, css_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f'{val:.3f}', ha='center', fontweight='bold')

    # OS AUC comparison
    ax = axes[1, 0]
    x = np.arange(len(TIME_LABELS))
    width = 0.35
    int_aucs = [internal_results['os']['auc'][t] for t in TIME_POINTS]
    ext_aucs = [external_results['os']['auc'][t] for t in TIME_POINTS]
    ax.bar(x - width / 2, int_aucs, width, label='SEER Internal', color=COLORS['primary'], alpha=0.8)
    ax.bar(x + width / 2, ext_aucs, width, label='Hospital External', color=COLORS['secondary'], alpha=0.8)
    ax.set_xlabel('Time Point')
    ax.set_ylabel('AUC')
    ax.set_title('OS Model: Time-Dependent AUC')
    ax.set_xticks(x)
    ax.set_xticklabels(TIME_LABELS)
    ax.legend()
    ax.set_ylim(0.4, 0.9)

    # CSS AUC comparison
    ax = axes[1, 1]
    int_aucs = [internal_results['css']['auc'][t] for t in TIME_POINTS]
    ext_aucs = [external_results['css']['auc'][t] for t in TIME_POINTS]
    ax.bar(x - width / 2, int_aucs, width, label='SEER Internal', color=COLORS['primary'], alpha=0.8)
    ax.bar(x + width / 2, ext_aucs, width, label='Hospital External', color=COLORS['secondary'], alpha=0.8)
    ax.set_xlabel('Time Point')
    ax.set_ylabel('AUC')
    ax.set_title('CSS Model: Time-Dependent AUC')
    ax.set_xticks(x)
    ax.set_xticklabels(TIME_LABELS)
    ax.legend()
    ax.set_ylim(0.4, 0.9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()


def main():
    print("=" * 70)
    print("06. EXTERNAL VALIDATION (HOSPITAL)")
    print("=" * 70)

    # Ensure output directories exist
    for d in [ROC_DIR, CALIBRATION_DIR, COMPARISON_DIR, TABLES_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # Load hospital data
    print(f"\nLoading hospital data: {HOSPITAL_PATH}")
    df_main = pd.read_excel(HOSPITAL_PATH, sheet_name='需要搜集的数据', header=0)
    df_tnm = pd.read_excel(HOSPITAL_PATH, sheet_name='筛选2', header=1)

    # Normalize IDs
    if '住院号' in df_main.columns:
        df_main['住院号'] = normalize_patient_id(df_main['住院号'])
    if '住院号' in df_tnm.columns:
        df_tnm['住院号'] = normalize_patient_id(df_tnm['住院号'])

    # Clean IDs
    df_main = df_main[df_main['住院号'].notna() & (df_main['住院号'] != 'nan')]
    df_tnm = df_tnm[df_tnm['住院号'].notna() & (df_tnm['住院号'] != 'nan')]

    # Merge
    merge_cols = ['住院号', 'T', 'N', 'M', 'Clinical stage', '随访至今是否存活（2024/10/17）', '死亡', '手术时间']
    merge_cols = [c for c in merge_cols if c in df_tnm.columns]
    df_hospital = df_main.merge(df_tnm[merge_cols], on='住院号', how='left')

    print(f"Merged hospital data: {len(df_hospital)} records")

    # Harmonize
    df_harmonized = harmonize_hospital_data(df_hospital)
    df_harmonized = df_harmonized.dropna(subset=['time_os', 'event_os'])
    df_harmonized = df_harmonized[df_harmonized['time_os'] > 0]

    print(f"\nHarmonized data: {len(df_harmonized)} records")
    print(f"OS events: {int(df_harmonized['event_os'].sum())} ({df_harmonized['event_os'].mean() * 100:.1f}%)")

    # Check T, N, M distributions
    print(f"\n--- Hospital T, N, M Distributions ---")
    for col in ['T', 'N', 'M']:
        print(f"\n{col}:")
        print(df_harmonized[col].value_counts())

    # Load internal validation results
    internal_results = {}
    external_results = {}

    all_results = []

    for endpoint in ['os', 'css']:
        time_col = f'time_{endpoint}'
        event_col = f'event_{endpoint}'

        print(f"\n{'=' * 70}")
        print(f"EXTERNAL VALIDATION - {endpoint.upper()}")
        print(f"{'=' * 70}")

        # Load model
        with open(MODELS_DIR / f'multivariate_{endpoint}_model.json', 'r') as f:
            model_spec = json.load(f)

        with open(MODELS_DIR / f'cox_{endpoint}_model.pkl', 'rb') as f:
            cph = pickle.load(f)

        # Load internal results
        with open(MODELS_DIR / f'internal_validation_{endpoint}.json', 'r') as f:
            internal_val = json.load(f)

        feature_names = model_spec['feature_names']
        selected_vars = model_spec['selected_variables']

        # Prepare hospital data
        hospital_prepared = prepare_data_for_model(
            df_harmonized, selected_vars, time_col, event_col, feature_names
        )

        print(f"\nHospital {endpoint.upper()} data: {len(hospital_prepared)} records")
        print(f"Events: {int(hospital_prepared[event_col].sum())}")

        # C-index
        if len(hospital_prepared) > 20:
            ext_cindex = cph.score(hospital_prepared, scoring_method='concordance_index')
        else:
            ext_cindex = np.nan

        int_cindex = internal_val['validation_c_index']

        print(f"\n--- C-index ---")
        print(f"  SEER Internal:    {int_cindex:.4f}")
        print(f"  Hospital External: {ext_cindex:.4f}")
        print(f"  Drop: {int_cindex - ext_cindex:.4f}")

        # Time-dependent AUC
        print(f"\n--- Time-dependent AUC ---")
        ext_auc_results = {}
        for t, label in zip(TIME_POINTS, TIME_LABELS):
            fpr, tpr, auc_val, n_events = calculate_time_dependent_auc(
                cph, hospital_prepared, time_col, event_col, t
            )
            ext_auc_results[t] = {'fpr': fpr, 'tpr': tpr, 'auc': auc_val, 'n_events': n_events}

            int_auc = internal_val['auc_results'][str(t)]['auc']
            reliability = "⚠️ UNRELIABLE" if n_events < 10 else ""
            print(f"  {label}: Internal={int_auc:.4f}, External={auc_val:.4f} (n_events={n_events}) {reliability}")

            all_results.append({
                'Endpoint': endpoint.upper(),
                'Metric': 'AUC',
                'Time_Point': label,
                'SEER_Internal': int_auc,
                'Hospital_External': auc_val,
                'Drop': int_auc - auc_val if not np.isnan(auc_val) else np.nan,
                'N_Events_External': n_events
            })

        # Store for comparison plot
        internal_results[endpoint] = {
            'c_index': int_cindex,
            'auc': {t: internal_val['auc_results'][str(t)]['auc'] for t in TIME_POINTS}
        }
        external_results[endpoint] = {
            'c_index': ext_cindex,
            'auc': {t: ext_auc_results[t]['auc'] for t in TIME_POINTS}
        }

        # Plot ROC curves
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        colors_list = [COLORS['primary'], COLORS['secondary'], COLORS['tertiary']]

        for i, (t, label) in enumerate(zip(TIME_POINTS, TIME_LABELS)):
            ax = axes[i]
            data = ext_auc_results[t]

            if data['fpr'] is not None:
                ax.plot(data['fpr'], data['tpr'], color=colors_list[i], linewidth=2,
                        label=f'AUC = {data["auc"]:.3f}')
            ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(f'{label} ROC (n_events={data["n_events"]})')
            ax.legend(loc='lower right')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect('equal')

        fig.suptitle(f'{endpoint.upper()} Model: Hospital External Validation ROC', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(ROC_DIR / f'roc_{endpoint}_external.png', dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()

        # Calibration
        calib_results = {}
        for t, label in zip(TIME_POINTS, TIME_LABELS):
            calib = calculate_calibration(cph, hospital_prepared, time_col, event_col, t)
            calib_results[t] = calib

        # Plot calibration
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for i, (t, label) in enumerate(zip(TIME_POINTS, TIME_LABELS)):
            ax = axes[i]
            data = calib_results[t]

            ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Perfect', alpha=0.7)

            if data is not None and len(data) > 0:
                ax.scatter(data['predicted'], data['observed'], s=data['n'] * 3,
                           alpha=0.7, color=COLORS['primary'], edgecolor='black', linewidth=1)
                ax.plot(data['predicted'], data['observed'], color=COLORS['secondary'],
                        linewidth=1.5, alpha=0.8)

            ax.set_xlabel('Predicted Survival')
            ax.set_ylabel('Observed Survival')
            ax.set_title(f'{label} Calibration')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.legend(loc='lower right', fontsize=8)
            ax.set_aspect('equal')

        fig.suptitle(f'{endpoint.upper()} Model: Hospital Calibration', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(CALIBRATION_DIR / f'calibration_{endpoint}_external.png', dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()

        # Add C-index to results
        all_results.append({
            'Endpoint': endpoint.upper(),
            'Metric': 'C-index',
            'Time_Point': 'Overall',
            'SEER_Internal': int_cindex,
            'Hospital_External': ext_cindex,
            'Drop': int_cindex - ext_cindex if not np.isnan(ext_cindex) else np.nan,
            'N_Events_External': int(hospital_prepared[event_col].sum())
        })

    # Plot comparison
    plot_comparison(internal_results, external_results, COMPARISON_DIR / 'internal_vs_external.png')

    # Save summary table
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(TABLES_DIR / 'external_validation_summary.csv', index=False)

    # Save detailed report
    report = {
        'hospital_cohort': {
            'n_total': len(df_harmonized),
            'n_os': len(hospital_prepared) if 'hospital_prepared' in dir() else 0,
            'os_events': int(df_harmonized['event_os'].sum()),
            'css_events': int(df_harmonized['event_css'].dropna().sum()) if 'event_css' in df_harmonized else 0
        },
        'os_model': {
            'seer_c_index': internal_results['os']['c_index'],
            'hospital_c_index': external_results['os']['c_index'],
            'c_index_drop': internal_results['os']['c_index'] - external_results['os']['c_index']
        },
        'css_model': {
            'seer_c_index': internal_results['css']['c_index'],
            'hospital_c_index': external_results['css']['c_index'],
            'c_index_drop': internal_results['css']['c_index'] - external_results['css']['c_index']
        }
    }

    with open(MODELS_DIR / 'external_validation_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n{'=' * 70}")
    print("EXTERNAL VALIDATION SUMMARY")
    print("=" * 70)
    print(results_df.to_string(index=False))

    print(f"\n--- Output Files ---")
    print(f"  ROC curves: {ROC_DIR}")
    print(f"  Calibration: {CALIBRATION_DIR}")
    print(f"  Comparison: {COMPARISON_DIR}")
    print(f"  Tables: {TABLES_DIR}")
    print("\n✓ External validation complete!")


if __name__ == '__main__':
    main()
