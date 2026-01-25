#!/usr/bin/env python3
"""
02_univariate_cox.py - Univariate Cox regression screening.

Input: Training data from outputs/data/train.pkl
Output: Univariate results table, selected variables for multivariate
Pos: Second step - screens candidate variables (p < 0.05)

Usage: python scripts/02_univariate_cox.py
"""

import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
import json
import warnings
warnings.filterwarnings("ignore")

from config import (
    DATA_DIR, TABLES_DIR, MODELS_DIR,
    CANDIDATE_VARS, UNIVARIATE_P_THRESHOLD, COX_PENALIZER
)


def run_univariate_cox(train_df, var, time_col, event_col):
    """Run univariate Cox regression for a single variable."""
    # Create dummy variables
    model_df = train_df[[time_col, event_col]].copy()
    dummies = pd.get_dummies(train_df[var].astype(str), prefix=var, drop_first=True, dtype=float)

    for col in dummies.columns:
        model_df[col] = dummies[col].values

    model_df = model_df.dropna()

    if len(model_df) < 50:
        return None

    try:
        cph = CoxPHFitter(penalizer=COX_PENALIZER)
        cph.fit(model_df, duration_col=time_col, event_col=event_col)

        # Get summary statistics
        summary = cph.summary.copy()
        summary['variable'] = var

        # Overall p-value (likelihood ratio test)
        lr_stat = cph.log_likelihood_ratio_test()
        overall_p = lr_stat.p_value

        return {
            'variable': var,
            'n_categories': len(dummies.columns) + 1,
            'n_events': int(model_df[event_col].sum()),
            'overall_p': overall_p,
            'c_index': cph.concordance_index_,
            'coefficients': summary[['coef', 'exp(coef)', 'se(coef)', 'p']].to_dict('index')
        }
    except Exception as e:
        print(f"  Warning: {var} failed - {e}")
        return None


def main():
    print("=" * 70)
    print("02. UNIVARIATE COX SCREENING")
    print("=" * 70)

    # Ensure output directories exist
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Load training data
    print(f"\nLoading training data...")
    train_df = pd.read_pickle(DATA_DIR / 'train.pkl')
    print(f"Training set: {len(train_df)} records")

    # Run univariate analysis for OS and CSS
    for endpoint in ['os', 'css']:
        time_col = f'time_{endpoint}'
        event_col = f'event_{endpoint}'

        print(f"\n{'=' * 70}")
        print(f"UNIVARIATE COX SCREENING - {endpoint.upper()}")
        print(f"{'=' * 70}")

        results = []
        for var in CANDIDATE_VARS:
            if var not in train_df.columns:
                print(f"  Skipping {var} (not in data)")
                continue

            result = run_univariate_cox(train_df, var, time_col, event_col)
            if result:
                results.append(result)
                status = "✓" if result['overall_p'] < UNIVARIATE_P_THRESHOLD else "✗"
                print(f"  {status} {var}: p={result['overall_p']:.4f}, C-index={result['c_index']:.3f}")

        # Create summary table
        summary_df = pd.DataFrame([{
            'Variable': r['variable'],
            'N_Categories': r['n_categories'],
            'N_Events': r['n_events'],
            'P_Value': r['overall_p'],
            'C_Index': r['c_index'],
            'Selected': r['overall_p'] < UNIVARIATE_P_THRESHOLD
        } for r in results])

        summary_df = summary_df.sort_values('P_Value')

        # Save summary table
        summary_df.to_csv(TABLES_DIR / f'univariate_{endpoint}.csv', index=False)

        # Get selected variables
        selected_vars = summary_df[summary_df['Selected']]['Variable'].tolist()

        print(f"\n--- Selected Variables (p < {UNIVARIATE_P_THRESHOLD}) ---")
        for var in selected_vars:
            row = summary_df[summary_df['Variable'] == var].iloc[0]
            print(f"  {var}: p={row['P_Value']:.4f}")

        # Save detailed results
        detailed_results = {
            'endpoint': endpoint,
            'n_train': len(train_df),
            'n_events': int(train_df[event_col].sum()),
            'p_threshold': UNIVARIATE_P_THRESHOLD,
            'selected_variables': selected_vars,
            'all_results': results
        }

        with open(MODELS_DIR / f'univariate_{endpoint}_results.json', 'w') as f:
            json.dump(detailed_results, f, indent=2, default=str)

        # Create detailed coefficient table
        coef_rows = []
        for r in results:
            for coef_name, coef_data in r['coefficients'].items():
                coef_rows.append({
                    'Variable': r['variable'],
                    'Category': coef_name.replace(f"{r['variable']}_", ""),
                    'Coefficient': coef_data['coef'],
                    'HR': coef_data['exp(coef)'],
                    'SE': coef_data['se(coef)'],
                    'P_Value': coef_data['p'],
                    'Overall_P': r['overall_p']
                })

        coef_df = pd.DataFrame(coef_rows)
        coef_df.to_csv(TABLES_DIR / f'univariate_{endpoint}_coefficients.csv', index=False)

    print(f"\n--- Output Files ---")
    print(f"  Tables: {TABLES_DIR}")
    print(f"  Models: {MODELS_DIR}")
    print("\n✓ Univariate screening complete!")


if __name__ == '__main__':
    main()
