#!/usr/bin/env python3
"""
03_multivariate_cox.py - Forward stepwise multivariate Cox regression.

Input: Training data and univariate screening results
Output: Final multivariate models for OS and CSS
Pos: Third step - builds final prediction models

Usage: python scripts/03_multivariate_cox.py
"""

import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
from scipy import stats
import json
import warnings
warnings.filterwarnings("ignore")

from config import (
    DATA_DIR, TABLES_DIR, MODELS_DIR, COX_PENALIZER
)


def build_model_df(data, variables, time_col, event_col):
    """Build model dataframe with dummy variables."""
    model_df = data[[time_col, event_col]].copy()
    feature_names = []

    for var in variables:
        if var not in data.columns:
            continue
        dummies = pd.get_dummies(data[var].astype(str), prefix=var, drop_first=True, dtype=float)
        for col in dummies.columns:
            model_df[col] = dummies[col].values
            feature_names.append(col)

    model_df = model_df.dropna()
    return model_df, feature_names


def forward_stepwise_cox(train_df, candidate_vars, time_col, event_col, p_enter=0.05):
    """
    Forward stepwise Cox regression using AIC improvement.
    """
    selected_vars = []
    remaining_vars = candidate_vars.copy()

    print(f"\n  Forward stepwise selection (p_enter={p_enter}):")
    print(f"  Candidates: {candidate_vars}")

    # Fit initial single-variable models and start with the best one
    best_initial_var = None
    best_initial_aic = np.inf

    for var in remaining_vars:
        try:
            test_df, _ = build_model_df(train_df, [var], time_col, event_col)
            cph = CoxPHFitter(penalizer=COX_PENALIZER)
            cph.fit(test_df, duration_col=time_col, event_col=event_col)
            if cph.AIC_partial_partial_ < best_initial_aic:
                best_initial_aic = cph.AIC_partial_
                best_initial_var = var
        except:
            continue

    if best_initial_var:
        selected_vars.append(best_initial_var)
        remaining_vars.remove(best_initial_var)
        print(f"    Step 1: Added {best_initial_var} (AIC={best_initial_aic:.2f})")

    step = 1
    while remaining_vars:
        step += 1
        best_var = None
        best_aic = np.inf
        best_p = 1.0

        # Current model AIC
        if selected_vars:
            current_df, _ = build_model_df(train_df, selected_vars, time_col, event_col)
            current_cph = CoxPHFitter(penalizer=COX_PENALIZER)
            current_cph.fit(current_df, duration_col=time_col, event_col=event_col)
            current_aic = current_cph.AIC_partial_
            current_ll = current_cph.log_likelihood_
        else:
            current_aic = np.inf
            current_ll = 0

        # Try adding each remaining variable
        for var in remaining_vars:
            test_vars = selected_vars + [var]
            try:
                test_df, test_features = build_model_df(train_df, test_vars, time_col, event_col)
                test_cph = CoxPHFitter(penalizer=COX_PENALIZER)
                test_cph.fit(test_df, duration_col=time_col, event_col=event_col)

                # Calculate likelihood ratio test p-value
                if selected_vars:
                    lr_stat = 2 * (test_cph.log_likelihood_ - current_ll)
                    # Count new parameters added
                    n_new = len([f for f in test_features if f.startswith(f'{var}_')])
                    if n_new == 0:
                        n_new = 1
                    p_value = 1 - stats.chi2.cdf(max(0, lr_stat), n_new)
                else:
                    p_value = 0  # First variable already added

                # Use AIC as tiebreaker
                if test_cph.AIC_partial_ < best_aic and p_value < p_enter:
                    best_aic = test_cph.AIC_partial_
                    best_var = var
                    best_p = p_value
            except Exception as e:
                continue

        # Add best variable if it improves model
        if best_var and best_aic < current_aic:
            selected_vars.append(best_var)
            remaining_vars.remove(best_var)
            print(f"    Step {step}: Added {best_var} (p={best_p:.4f}, AIC={best_aic:.2f})")
        else:
            print(f"    Step {step}: No improvement, stopping.")
            break

    return selected_vars


def main():
    print("=" * 70)
    print("03. MULTIVARIATE COX REGRESSION (FORWARD STEPWISE)")
    print("=" * 70)

    # Load training data
    print(f"\nLoading training data...")
    train_df = pd.read_pickle(DATA_DIR / 'train.pkl')
    print(f"Training set: {len(train_df)} records")

    # Run for OS and CSS
    for endpoint in ['os', 'css']:
        time_col = f'time_{endpoint}'
        event_col = f'event_{endpoint}'

        print(f"\n{'=' * 70}")
        print(f"MULTIVARIATE COX - {endpoint.upper()}")
        print(f"{'=' * 70}")

        # Load univariate results
        with open(MODELS_DIR / f'univariate_{endpoint}_results.json', 'r') as f:
            univariate = json.load(f)

        candidate_vars = univariate['selected_variables']
        print(f"\nCandidate variables from univariate: {candidate_vars}")

        # Run forward stepwise selection
        selected_vars = forward_stepwise_cox(
            train_df, candidate_vars, time_col, event_col
        )

        # If no variables selected, use top 3 from univariate
        if not selected_vars:
            print("\n  Warning: No variables selected by stepwise. Using top univariate variables.")
            selected_vars = candidate_vars[:3]

        print(f"\n  Final selected variables: {selected_vars}")

        # Fit final model
        print(f"\n--- Fitting Final Model ---")
        model_df, feature_names = build_model_df(train_df, selected_vars, time_col, event_col)

        cph = CoxPHFitter(penalizer=COX_PENALIZER)
        cph.fit(model_df, duration_col=time_col, event_col=event_col)

        print(f"  N: {len(model_df)}")
        print(f"  Events: {int(model_df[event_col].sum())}")
        print(f"  C-index: {cph.concordance_index_:.4f}")

        # Get summary - handle column names dynamically
        summary = cph.summary.copy()
        summary = summary.reset_index()

        # Rename columns based on what lifelines provides
        col_mapping = {
            'covariate': 'Variable',
            'index': 'Variable',
            'coef': 'Coefficient',
            'exp(coef)': 'HR',
            'se(coef)': 'SE',
            'coef lower 95%': 'Coef_Lower',
            'coef upper 95%': 'Coef_Upper',
            'exp(coef) lower 95%': 'HR_Lower',
            'exp(coef) upper 95%': 'HR_Upper',
            'cmp to': 'Comparison',
            'z': 'Z',
            'p': 'P_Value',
            '-log2(p)': 'Log2P'
        }
        summary.columns = [col_mapping.get(c, c) for c in summary.columns]

        print(f"\n--- Model Coefficients ---")
        for _, row in summary.iterrows():
            var_name = row.get('Variable', row.iloc[0])
            hr = row.get('HR', np.exp(row.get('Coefficient', 0)))
            hr_lower = row.get('HR_Lower', hr * 0.9)
            hr_upper = row.get('HR_Upper', hr * 1.1)
            p_val = row.get('P_Value', 0.05)

            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            print(f"  {str(var_name):30s} HR={hr:.3f} ({hr_lower:.3f}-{hr_upper:.3f}) p={p_val:.4f} {sig}")

        # Save coefficient table
        summary.to_csv(TABLES_DIR / f'multivariate_{endpoint}_coefficients.csv', index=False)

        # Save model specification
        model_spec = {
            'endpoint': endpoint,
            'selected_variables': selected_vars,
            'feature_names': feature_names,
            'n_train': len(model_df),
            'n_events': int(model_df[event_col].sum()),
            'performance': {
                'training_c_index': cph.concordance_index_,
                'log_likelihood': float(cph.log_likelihood_),
                'AIC': float(cph.AIC_partial_)
            },
            'coefficients': {k: float(v) for k, v in cph.params_.to_dict().items()},
        }

        # Handle baseline hazard serialization
        try:
            bh = cph.baseline_hazard_
            model_spec['baseline_hazard'] = {str(k): float(v) for k, v in bh.iloc[:, 0].to_dict().items()}
        except:
            model_spec['baseline_hazard'] = {}

        with open(MODELS_DIR / f'multivariate_{endpoint}_model.json', 'w') as f:
            json.dump(model_spec, f, indent=2)

        # Save fitted model
        import pickle
        with open(MODELS_DIR / f'cox_{endpoint}_model.pkl', 'wb') as f:
            pickle.dump(cph, f)

    print(f"\n--- Output Files ---")
    print(f"  Tables: {TABLES_DIR}")
    print(f"  Models: {MODELS_DIR}")
    print("\n✓ Multivariate modeling complete!")


if __name__ == '__main__':
    main()
