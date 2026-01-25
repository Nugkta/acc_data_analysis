"""
SEER ACC Data Preprocessing Script
Replicates the R preprocessing approach - KEEPS all records including those with missing TNM
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

try:
    from sklearn.model_selection import train_test_split
except ImportError:
    print("ERROR: sklearn not installed. Please run: pip install scikit-learn")
    exit(1)

# Set random seed for reproducibility
np.random.seed(42)

def recode_site(site_value):
    """Recode site codes to Chinese categories"""
    if pd.isna(site_value):
        return site_value
    site_str = str(site_value)

    if 'C07' in site_str or 'C08' in site_str or 'Parotid' in site_str or 'Submandibular' in site_str or 'Sublingual' in site_str:
        return '大唾液腺'
    elif 'C00' in site_str or 'Lip' in site_str:
        return '唇'
    elif 'C01' in site_str or 'C02' in site_str or 'Tongue' in site_str:
        return '舌'
    elif 'C03' in site_str or 'C04' in site_str or 'C05' in site_str or 'C06' in site_str or 'Gum' in site_str or 'Floor of mouth' in site_str or 'Palate' in site_str or 'mouth' in site_str:
        return '口腔'
    elif 'C09' in site_str or 'C10' in site_str or 'Tonsil' in site_str or 'Oropharynx' in site_str:
        return '口咽'
    elif 'C11' in site_str or 'Nasopharynx' in site_str:
        return '鼻咽'
    elif 'C12' in site_str or 'C13' in site_str or 'Hypopharynx' in site_str or 'Pyriform' in site_str:
        return '下咽'
    elif 'C32' in site_str or 'Larynx' in site_str:
        return '喉和下咽'
    elif 'C30' in site_str or 'C31' in site_str or 'Nasal' in site_str or 'Sinus' in site_str:
        return '鼻腔鼻窦'
    else:
        return '其他'

def main():
    print("=" * 70)
    print("SEER ACC Data Preprocessing - Following R Approach")
    print("=" * 70)

    # Load raw SEER data
    print("\n1. Loading raw SEER data...")
    raw_path = 'ACC数据/r分析seer/SEER纯ACC数据.xlsx'
    df_raw = pd.read_excel(raw_path)
    print(f"   Raw data shape: {df_raw.shape}")

    # Column mapping
    column_mapping = {
        'age': '年龄',
        'sex': '性别',
        'site': '原发部位',
        'grade': '分化级别(thru 2017)',
        'radiotherapy': '放疗',
        'chemotherapy': '化疗',
        'tumor_number': '肿瘤数量',
        'race': '种族',
        'marital_status': '婚姻',
        'urban_rural': '城乡',
        'time': '存活月数',
        'status': '生存（截止至研究日期）',
        'TNMstage': 'TNM',
        'ID': '编号'
    }

    # Select and rename columns
    print("\n2. Selecting and renaming columns...")
    df_clean = df_raw[[column_mapping[k] for k in column_mapping.keys()]].copy()
    df_clean.columns = list(column_mapping.keys())
    print(f"   Selected {len(df_clean.columns)} core variables")

    # Recode site variable
    print("\n3. Recoding 'site' variable...")
    df_clean['site'] = df_clean['site'].apply(recode_site)
    print(f"   Site recoded. Unique values: {df_clean['site'].nunique()}")

    # Convert categorical variables
    print("\n4. Converting categorical variables...")
    categorical_vars = ['age', 'sex', 'site', 'grade', 'radiotherapy', 'chemotherapy',
                       'tumor_number', 'race', 'marital_status', 'urban_rural', 'TNMstage']

    for var in categorical_vars:
        df_clean[var] = df_clean[var].astype('category')
    print(f"   ✓ {len(categorical_vars)} variables converted to category")

    # Ensure numeric types
    df_clean['status'] = pd.to_numeric(df_clean['status'], errors='coerce')
    df_clean['time'] = pd.to_numeric(df_clean['time'], errors='coerce')

    # Check for missing values - but DON'T drop them!
    print("\n5. Missing values summary...")
    print(f"   Total records: {len(df_clean)}")

    missing_summary = df_clean.isnull().sum()
    missing_summary = missing_summary[missing_summary > 0]
    if len(missing_summary) > 0:
        print(f"   Missing values by column:")
        for col, count in missing_summary.items():
            print(f"     {col}: {count} ({count/len(df_clean)*100:.1f}%)")

    print(f"\n   IMPORTANT: Keeping all {len(df_clean)} records (including those with missing TNMstage)")
    print(f"   Cox model will automatically handle missing values during fitting")

    # Check event rate
    event_rate = (df_clean['status'] == 1).sum() / len(df_clean) * 100
    print(f"\n   Event rate (deaths): {event_rate:.1f}%")
    print(f"   Deaths: {(df_clean['status'] == 1).sum()}")
    print(f"   Alive: {(df_clean['status'] == 0).sum()}")

    # Split into training and validation (keeping all records)
    print("\n6. Splitting into training and validation sets...")
    train_data, val_data = train_test_split(
        df_clean,
        test_size=0.33,
        random_state=42,
        stratify=df_clean['status']
    )

    print(f"   Training: {len(train_data)} ({len(train_data)/len(df_clean)*100:.1f}%)")
    print(f"   Validation: {len(val_data)} ({len(val_data)/len(df_clean)*100:.1f}%)")

    train_events = (train_data['status'] == 1).sum()
    val_events = (val_data['status'] == 1).sum()
    print(f"   Training events: {train_events} ({train_events/len(train_data)*100:.1f}%)")
    print(f"   Validation events: {val_events} ({val_events/len(val_data)*100:.1f}%)")

    # Show TNM availability
    print(f"\n   TNMstage availability:")
    print(f"   Training with TNM: {train_data['TNMstage'].notna().sum()} / {len(train_data)} ({train_data['TNMstage'].notna().sum()/len(train_data)*100:.1f}%)")
    print(f"   Validation with TNM: {val_data['TNMstage'].notna().sum()} / {len(val_data)} ({val_data['TNMstage'].notna().sum()/len(val_data)*100:.1f}%)")

    # Align categorical levels
    print("\n7. Aligning categorical levels between train/val...")
    for var in categorical_vars:
        train_categories = train_data[var].cat.categories
        val_data[var] = pd.Categorical(
            val_data[var],
            categories=train_categories,
            ordered=train_data[var].cat.ordered
        )
    print("   ✓ Categorical levels aligned")

    # Save processed datasets
    print("\n8. Saving processed datasets...")
    output_dir = Path('data/processed')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save as CSV
    train_data.to_csv(output_dir / 'seer_train.csv', index=False)
    val_data.to_csv(output_dir / 'seer_validation.csv', index=False)
    df_clean.to_csv(output_dir / 'seer_full.csv', index=False)

    # Save as pickle (preserves categorical dtypes)
    train_data.to_pickle(output_dir / 'seer_train.pkl')
    val_data.to_pickle(output_dir / 'seer_validation.pkl')
    df_clean.to_pickle(output_dir / 'seer_full.pkl')

    print(f"   ✓ CSV files saved to {output_dir}")
    print(f"   ✓ Pickle files saved to {output_dir}")

    # Save data dictionary
    data_dict = {
        'source_file': raw_path,
        'preprocessing_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_records': len(df_raw),
        'records_after_preprocessing': len(df_clean),
        'training_records': len(train_data),
        'validation_records': len(val_data),
        'column_mapping': column_mapping,
        'categorical_variables': categorical_vars,
        'model_variables': ['age', 'sex', 'site', 'grade', 'chemotherapy', 'marital_status', 'TNMstage'],
        'random_seed': 42,
        'missing_data_handling': 'Kept all records (R Cox model handles missing values automatically)',
        'tnm_missing_pct': float(missing_summary.get('TNMstage', 0) / len(df_clean) * 100)
    }

    with open(output_dir / 'data_dictionary.json', 'w', encoding='utf-8') as f:
        json.dump(data_dict, f, indent=2, ensure_ascii=False)

    print(f"   ✓ Data dictionary saved")

    # Display summary
    print("\n" + "=" * 70)
    print("PREPROCESSING COMPLETE!")
    print("=" * 70)
    print(f"\nFinal Model Variables (7):")
    for var in data_dict['model_variables']:
        print(f"  - {var}")

    print(f"\n✓ Training set: data/processed/seer_train.pkl ({len(train_data)} records)")
    print(f"✓ Validation set: data/processed/seer_validation.pkl ({len(val_data)} records)")
    print(f"✓ Full dataset: data/processed/seer_full.pkl ({len(df_clean)} records)")

    print(f"\nNote: {missing_summary.get('TNMstage', 0)} records ({missing_summary.get('TNMstage', 0)/len(df_clean)*100:.1f}%) have missing TNMstage")
    print(f"These records will be automatically excluded during Cox model fitting")
    print(f"But are retained for other analyses and flexibility")

    # Show variable distributions
    print(f"\n" + "=" * 70)
    print("Variable Distributions (Training Set)")
    print("=" * 70)
    for var in ['age', 'sex', 'site', 'grade', 'chemotherapy', 'marital_status', 'TNMstage']:
        print(f"\n{var.upper()}:")
        counts = train_data[var].value_counts(dropna=False).head(10)
        for value, count in counts.items():
            if pd.isna(value):
                print(f"  Missing: {count} ({count/len(train_data)*100:.1f}%)")
            else:
                print(f"  {value}: {count} ({count/len(train_data)*100:.1f}%)")

if __name__ == '__main__':
    main()
