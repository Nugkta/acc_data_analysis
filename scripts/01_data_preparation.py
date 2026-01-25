#!/usr/bin/env python3
"""
01_data_preparation.py - Prepare SEER data with separate T, N, M components.

Input: Raw SEER Excel file (ACC数据/r分析seer/SEER纯ACC数据.xlsx)
Output: Train/validation splits saved to outputs/data/
Pos: First step in pipeline - creates clean datasets for modeling

Usage: python scripts/01_data_preparation.py
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import json
import warnings
warnings.filterwarnings("ignore")

from config import (
    RAW_SEER_PATH, DATA_DIR, RANDOM_STATE, TEST_SIZE,
    SEER_COLUMN_MAP, CANDIDATE_VARS
)

def recode_site(site_value):
    """Recode ICD-O-3 site codes to 4 categories."""
    if pd.isna(site_value):
        return np.nan
    site_str = str(site_value)
    code = site_str.split('-')[0] if '-' in site_str else site_str
    code_prefix = code[:3] if len(code) >= 3 else code

    if code_prefix in ['C07', 'C08']:
        return '大唾液腺'
    if code_prefix in ['C32', 'C12', 'C13']:
        return '喉和下咽'
    if code_prefix in ['C30', 'C31', 'C11']:
        return '鼻腔鼻窦副鼻窦鼻咽'
    return '口腔口咽其它'


def clean_tnm_value(val, component):
    """Standardize T, N, M values."""
    if pd.isna(val):
        return f'{component}X'
    val = str(val).strip().upper()
    if val in ['', 'NAN', 'NA']:
        return f'{component}X'
    # Remove component prefix if present for standardization
    if val.startswith(component):
        return val
    return f'{component}{val}' if not val.startswith(component) else val


def main():
    print("=" * 70)
    print("01. DATA PREPARATION WITH SEPARATE T, N, M")
    print("=" * 70)

    # Ensure output directory exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Load raw SEER data
    print(f"\nLoading: {RAW_SEER_PATH}")
    df_raw = pd.read_excel(RAW_SEER_PATH)
    print(f"Raw data: {df_raw.shape}")

    # Select and rename columns
    df = df_raw[[v for v in SEER_COLUMN_MAP.values()]].copy()
    df.columns = list(SEER_COLUMN_MAP.keys())

    # Apply site recoding
    df['site'] = df['site_raw'].apply(recode_site)
    df = df.drop(columns=['site_raw'])

    # Define CSS endpoint
    df['event_css'] = (df['css_classification'] == 'Dead (attributable to this cancer dx)').astype(int)
    df['time_css'] = df['time_os']
    df = df.drop(columns=['css_classification'])

    # Clean T, N, M values
    for col in ['T', 'N', 'M']:
        df[col] = df[col].apply(lambda x: clean_tnm_value(x, col))

    print(f"\n--- T, N, M Distributions (Raw) ---")
    for col in ['T', 'N', 'M']:
        print(f"\n{col}:")
        print(df[col].value_counts().head(10))

    # Convert event_os to numeric
    df['event_os'] = pd.to_numeric(df['event_os'], errors='coerce').astype(int)
    df['time_os'] = pd.to_numeric(df['time_os'], errors='coerce')

    # Apply exclusion criteria
    print(f"\n--- Applying Exclusion Criteria ---")
    print(f"Starting N: {len(df)}")

    # Exclude invalid survival time
    valid_time = (df['time_os'].notna()) & (df['time_os'] > 0)
    n_invalid_time = (~valid_time).sum()
    print(f"Excluded (invalid time): {n_invalid_time}")

    # Exclude missing survival status
    valid_status = df['event_os'].notna()
    n_invalid_status = (~valid_status).sum()
    print(f"Excluded (missing status): {n_invalid_status}")

    df_clean = df[valid_time & valid_status].copy()
    print(f"After exclusions: {len(df_clean)}")

    # Exclude cases where all T, N, M are unknown (TX, NX, MX)
    tnm_known = ~((df_clean['T'] == 'TX') & (df_clean['N'] == 'NX') & (df_clean['M'] == 'MX'))
    n_tnm_unknown = (~tnm_known).sum()
    print(f"Excluded (all T/N/M unknown): {n_tnm_unknown}")

    df_final = df_clean[tnm_known].copy()
    print(f"Final cohort: {len(df_final)}")

    # Convert categorical variables
    for var in CANDIDATE_VARS:
        if var in df_final.columns:
            df_final[var] = df_final[var].astype('category')

    # Create train/validation split
    print(f"\n--- Creating Train/Validation Split ---")
    train_df, val_df = train_test_split(
        df_final,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df_final['event_os']
    )

    # Align categorical levels
    for var in CANDIDATE_VARS:
        if var in train_df.columns:
            train_categories = train_df[var].cat.categories
            val_df[var] = pd.Categorical(val_df[var], categories=train_categories)
            df_final[var] = pd.Categorical(df_final[var], categories=train_categories)

    print(f"Training: {len(train_df)} ({len(train_df)/len(df_final)*100:.1f}%)")
    print(f"Validation: {len(val_df)} ({len(val_df)/len(df_final)*100:.1f}%)")

    # Summary statistics
    print(f"\n--- Cohort Summary ---")
    print(f"OS events: {df_final['event_os'].sum()} ({df_final['event_os'].mean()*100:.1f}%)")
    print(f"CSS events: {df_final['event_css'].sum()} ({df_final['event_css'].mean()*100:.1f}%)")
    print(f"Median follow-up: {df_final['time_os'].median():.1f} months")

    print(f"\n--- T, N, M Distributions (Final) ---")
    for col in ['T', 'N', 'M']:
        print(f"\n{col} (train):")
        print(train_df[col].value_counts())

    # Save datasets
    print(f"\n--- Saving Data ---")
    train_df.to_pickle(DATA_DIR / 'train.pkl')
    val_df.to_pickle(DATA_DIR / 'validation.pkl')
    df_final.to_pickle(DATA_DIR / 'full_cohort.pkl')

    train_df.to_csv(DATA_DIR / 'train.csv', index=False)
    val_df.to_csv(DATA_DIR / 'validation.csv', index=False)

    # Save data dictionary
    data_dict = {
        'created': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'source': str(RAW_SEER_PATH),
        'total_records': len(df_final),
        'train_n': len(train_df),
        'val_n': len(val_df),
        'os_events': int(df_final['event_os'].sum()),
        'css_events': int(df_final['event_css'].sum()),
        'candidate_variables': CANDIDATE_VARS,
        'staging_approach': 'Separate T, N, M components',
        'exclusions': {
            'invalid_time': int(n_invalid_time),
            'missing_status': int(n_invalid_status),
            'all_tnm_unknown': int(n_tnm_unknown)
        }
    }

    with open(DATA_DIR / 'data_dictionary.json', 'w', encoding='utf-8') as f:
        json.dump(data_dict, f, indent=2, ensure_ascii=False)

    print(f"Saved to: {DATA_DIR}")
    print("\n✓ Data preparation complete!")

    return train_df, val_df


if __name__ == '__main__':
    main()
