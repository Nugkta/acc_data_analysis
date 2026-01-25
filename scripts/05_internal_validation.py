#!/usr/bin/env python3
"""
05_internal_validation.py - Internal validation on SEER validation set.

Input: Fitted models and SEER validation data
Output: C-index, time-dependent AUC, calibration plots, DCA
Pos: Fifth step - validates models on held-out SEER data

Usage: python scripts/05_internal_validation.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter, KaplanMeierFitter
from sklearn.metrics import roc_curve, auc
import pickle
import json
import warnings
warnings.filterwarnings("ignore")

from config import (
    DATA_DIR, MODELS_DIR, TABLES_DIR, ROC_DIR, CALIBRATION_DIR, COMPARISON_DIR,
    TIME_POINTS, TIME_LABELS, FIGURE_DPI, COLORS
)


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
    """Calculate time-dependent ROC curve and AUC."""
    mask = (data[time_col] >= time_point) | ((data[time_col] < time_point) & (data[event_col] == 1))
    eval_data = data[mask].copy()

    if len(eval_data) < 20:
        return None, None, np.nan, 0

    eval_data['outcome'] = ((eval_data[time_col] <= time_point) & (eval_data[event_col] == 1)).astype(int)
    n_events = eval_data['outcome'].sum()

    if n_events < 5:
        return None, None, np.nan, n_events

    covariates = eval_data.drop([time_col, event_col, 'outcome'], axis=1)
    risk_scores = cph.predict_log_partial_hazard(covariates)

    fpr, tpr, _ = roc_curve(eval_data['outcome'], risk_scores)
    roc_auc = auc(fpr, tpr)

    return fpr, tpr, roc_auc, n_events


def calculate_calibration(cph, data, time_col, event_col, time_point, n_groups=5):
    """Calculate calibration: predicted vs observed survival."""
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
            idx = np.abs(kmf.survival_function_.index - time_point).argmin()
            observed = kmf.survival_function_.iloc[idx].values[0]

        results.append({
            'group': group,
            'n': len(group_data),
            'predicted': mean_predicted,
            'observed': observed
        })

    return pd.DataFrame(results)


def plot_roc_curves(auc_results, time_points, time_labels, title, output_path):
    """Plot ROC curves for multiple time points."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    colors = [COLORS['primary'], COLORS['secondary'], COLORS['tertiary']]

    for i, (t, label) in enumerate(zip(time_points, time_labels)):
        ax = axes[i]
        data = auc_results[t]

        if data['fpr'] is not None:
            ax.plot(data['fpr'], data['tpr'], color=colors[i], linewidth=2,
                    label=f'AUC = {data["auc"]:.3f}')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'{label} ROC (n_events={data["n_events"]})')
        ax.legend(loc='lower right')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')

    fig.suptitle(title, fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()


def plot_calibration_curves(calib_results, time_points, time_labels, title, output_path):
    """Plot calibration curves."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    for i, (t, label) in enumerate(zip(time_points, time_labels)):
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

    fig.suptitle(title, fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()


def main():
    print("=" * 70)
    print("05. INTERNAL VALIDATION (SEER)")
    print("=" * 70)

    # Ensure output directories exist
    for d in [ROC_DIR, CALIBRATION_DIR, COMPARISON_DIR, TABLES_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # Load data
    train_df = pd.read_pickle(DATA_DIR / 'train.pkl')
    val_df = pd.read_pickle(DATA_DIR / 'validation.pkl')

    print(f"\nTraining: {len(train_df)}, Validation: {len(val_df)}")

    all_results = []

    for endpoint in ['os', 'css']:
        time_col = f'time_{endpoint}'
        event_col = f'event_{endpoint}'

        print(f"\n{'=' * 70}")
        print(f"INTERNAL VALIDATION - {endpoint.upper()}")
        print(f"{'=' * 70}")

        # Load model
        with open(MODELS_DIR / f'multivariate_{endpoint}_model.json', 'r') as f:
            model_spec = json.load(f)

        with open(MODELS_DIR / f'cox_{endpoint}_model.pkl', 'rb') as f:
            cph = pickle.load(f)

        feature_names = model_spec['feature_names']
        selected_vars = model_spec['selected_variables']

        # Prepare validation data
        val_prepared = prepare_data_for_model(val_df, selected_vars, time_col, event_col, feature_names)

        # C-index
        train_cindex = model_spec['performance']['training_c_index']
        val_cindex = cph.score(val_prepared, scoring_method='concordance_index')

        print(f"\n--- C-index ---")
        print(f"  Training:   {train_cindex:.4f}")
        print(f"  Validation: {val_cindex:.4f}")
        print(f"  Drop: {train_cindex - val_cindex:.4f}")

        # Time-dependent AUC
        print(f"\n--- Time-dependent AUC ---")
        auc_results = {}
        for t, label in zip(TIME_POINTS, TIME_LABELS):
            fpr, tpr, auc_val, n_events = calculate_time_dependent_auc(
                cph, val_prepared, time_col, event_col, t
            )
            auc_results[t] = {'fpr': fpr, 'tpr': tpr, 'auc': auc_val, 'n_events': n_events}
            print(f"  {label}: AUC={auc_val:.4f} (n_events={n_events})")

            all_results.append({
                'Endpoint': endpoint.upper(),
                'Metric': 'AUC',
                'Time_Point': label,
                'Time_Months': t,
                'Value': auc_val,
                'N_Events': n_events
            })

        # Plot ROC curves
        plot_roc_curves(
            auc_results, TIME_POINTS, TIME_LABELS,
            f'{endpoint.upper()} Model: SEER Internal Validation ROC',
            ROC_DIR / f'roc_{endpoint}_internal.png'
        )

        # Calibration
        print(f"\n--- Calibration ---")
        calib_results = {}
        for t, label in zip(TIME_POINTS, TIME_LABELS):
            calib = calculate_calibration(cph, val_prepared, time_col, event_col, t)
            calib_results[t] = calib
            if calib is not None:
                # Calculate calibration slope
                mean_pred = calib['predicted'].mean()
                mean_obs = calib['observed'].mean()
                print(f"  {label}: mean_pred={mean_pred:.3f}, mean_obs={mean_obs:.3f}")

        # Plot calibration curves
        plot_calibration_curves(
            calib_results, TIME_POINTS, TIME_LABELS,
            f'{endpoint.upper()} Model: SEER Internal Validation Calibration',
            CALIBRATION_DIR / f'calibration_{endpoint}_internal.png'
        )

        # Save detailed results
        val_results = {
            'endpoint': endpoint,
            'n_validation': int(len(val_prepared)),
            'n_events': int(val_prepared[event_col].sum()),
            'training_c_index': float(train_cindex),
            'validation_c_index': float(val_cindex),
            'auc_results': {str(t): {'auc': float(r['auc']) if not np.isnan(r['auc']) else None,
                                     'n_events': int(r['n_events'])}
                           for t, r in auc_results.items()}
        }

        with open(MODELS_DIR / f'internal_validation_{endpoint}.json', 'w') as f:
            json.dump(val_results, f, indent=2)

        # Add C-index to results
        all_results.append({
            'Endpoint': endpoint.upper(),
            'Metric': 'C-index',
            'Time_Point': 'Overall',
            'Time_Months': None,
            'Value': val_cindex,
            'N_Events': int(val_prepared[event_col].sum())
        })

    # Save summary table
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(TABLES_DIR / 'internal_validation_summary.csv', index=False)

    print(f"\n{'=' * 70}")
    print("INTERNAL VALIDATION SUMMARY")
    print("=" * 70)
    print(results_df.to_string(index=False))

    print(f"\n--- Output Files ---")
    print(f"  ROC curves: {ROC_DIR}")
    print(f"  Calibration: {CALIBRATION_DIR}")
    print(f"  Tables: {TABLES_DIR}")
    print("\n✓ Internal validation complete!")


if __name__ == '__main__':
    main()
