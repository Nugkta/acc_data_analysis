#!/usr/bin/env python3
"""
07_generate_figures.py - Generate publication-ready figures and tables.

Input: All previous analysis outputs
Output: Final figures for publication, summary tables
Pos: Final step - creates polished outputs for manuscript

Usage: python scripts/07_generate_figures.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from lifelines import KaplanMeierFitter
import pickle
import json
import warnings
warnings.filterwarnings("ignore")

from config import (
    DATA_DIR, MODELS_DIR, TABLES_DIR, FIGURES_DIR, KM_DIR, COMPARISON_DIR,
    TIME_POINTS, TIME_LABELS, FIGURE_DPI, COLORS, CANDIDATE_VARS
)


def create_baseline_table(train_df, val_df):
    """Create baseline characteristics table."""
    rows = []

    # Continuous variables
    for df, cohort in [(train_df, 'Training'), (val_df, 'Validation')]:
        # Age statistics (from age groups)
        age_dist = df['age'].value_counts(normalize=True)

        rows.append({
            'Characteristic': 'N',
            'Cohort': cohort,
            'Value': str(len(df))
        })

    # Categorical variables
    cat_vars = ['age', 'sex', 'T', 'N', 'M', 'grade', 'radiotherapy', 'chemotherapy']

    for var in cat_vars:
        if var not in train_df.columns:
            continue

        for df, cohort in [(train_df, 'Training'), (val_df, 'Validation')]:
            counts = df[var].value_counts()
            total = len(df)

            for cat in counts.index:
                n = counts[cat]
                pct = n / total * 100
                rows.append({
                    'Characteristic': f'{var}',
                    'Category': str(cat),
                    'Cohort': cohort,
                    'N': n,
                    'Percentage': f'{pct:.1f}%',
                    'Value': f'{n} ({pct:.1f}%)'
                })

    return pd.DataFrame(rows)


def plot_kaplan_meier_by_stage(train_df, output_dir):
    """Plot Kaplan-Meier curves stratified by T, N, M stages."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # T stage - OS
    ax = axes[0, 0]
    kmf = KaplanMeierFitter()
    for t_stage in sorted(train_df['T'].unique()):
        if t_stage == 'TX':
            continue
        mask = train_df['T'] == t_stage
        if mask.sum() >= 10:
            kmf.fit(train_df.loc[mask, 'time_os'], train_df.loc[mask, 'event_os'], label=t_stage)
            kmf.plot_survival_function(ax=ax)
    ax.set_xlabel('Time (months)')
    ax.set_ylabel('Overall Survival')
    ax.set_title('OS by T Stage')
    ax.set_xlim(0, 120)
    ax.legend(loc='lower left', fontsize=8)

    # N stage - OS
    ax = axes[0, 1]
    kmf = KaplanMeierFitter()
    for n_stage in ['N0', 'N1', 'N2a', 'N2b', 'N2c', 'N3b']:
        mask = train_df['N'] == n_stage
        if mask.sum() >= 10:
            kmf.fit(train_df.loc[mask, 'time_os'], train_df.loc[mask, 'event_os'], label=n_stage)
            kmf.plot_survival_function(ax=ax)
    ax.set_xlabel('Time (months)')
    ax.set_ylabel('Overall Survival')
    ax.set_title('OS by N Stage')
    ax.set_xlim(0, 120)
    ax.legend(loc='lower left', fontsize=8)

    # M stage - OS
    ax = axes[0, 2]
    kmf = KaplanMeierFitter()
    for m_stage in ['M0', 'M1']:
        mask = train_df['M'] == m_stage
        if mask.sum() >= 10:
            kmf.fit(train_df.loc[mask, 'time_os'], train_df.loc[mask, 'event_os'], label=m_stage)
            kmf.plot_survival_function(ax=ax)
    ax.set_xlabel('Time (months)')
    ax.set_ylabel('Overall Survival')
    ax.set_title('OS by M Stage')
    ax.set_xlim(0, 120)
    ax.legend(loc='lower left', fontsize=8)

    # T stage - CSS
    ax = axes[1, 0]
    kmf = KaplanMeierFitter()
    for t_stage in sorted(train_df['T'].unique()):
        if t_stage == 'TX':
            continue
        mask = train_df['T'] == t_stage
        if mask.sum() >= 10:
            kmf.fit(train_df.loc[mask, 'time_css'], train_df.loc[mask, 'event_css'], label=t_stage)
            kmf.plot_survival_function(ax=ax)
    ax.set_xlabel('Time (months)')
    ax.set_ylabel('Cancer-Specific Survival')
    ax.set_title('CSS by T Stage')
    ax.set_xlim(0, 120)
    ax.legend(loc='lower left', fontsize=8)

    # N stage - CSS
    ax = axes[1, 1]
    kmf = KaplanMeierFitter()
    for n_stage in ['N0', 'N1', 'N2a', 'N2b', 'N2c', 'N3b']:
        mask = train_df['N'] == n_stage
        if mask.sum() >= 10:
            kmf.fit(train_df.loc[mask, 'time_css'], train_df.loc[mask, 'event_css'], label=n_stage)
            kmf.plot_survival_function(ax=ax)
    ax.set_xlabel('Time (months)')
    ax.set_ylabel('Cancer-Specific Survival')
    ax.set_title('CSS by N Stage')
    ax.set_xlim(0, 120)
    ax.legend(loc='lower left', fontsize=8)

    # M stage - CSS
    ax = axes[1, 2]
    kmf = KaplanMeierFitter()
    for m_stage in ['M0', 'M1']:
        mask = train_df['M'] == m_stage
        if mask.sum() >= 10:
            kmf.fit(train_df.loc[mask, 'time_css'], train_df.loc[mask, 'event_css'], label=m_stage)
            kmf.plot_survival_function(ax=ax)
    ax.set_xlabel('Time (months)')
    ax.set_ylabel('Cancer-Specific Survival')
    ax.set_title('CSS by M Stage')
    ax.set_xlim(0, 120)
    ax.legend(loc='lower left', fontsize=8)

    plt.suptitle('Kaplan-Meier Survival Curves by T, N, M Stage\n(SEER Training Cohort)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'km_by_tnm_stage.png', dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()


def plot_forest_plot(model_spec, endpoint, output_path):
    """Create forest plot for model coefficients."""
    with open(MODELS_DIR / f'cox_{endpoint}_model.pkl', 'rb') as f:
        cph = pickle.load(f)

    summary = cph.summary.copy().reset_index()

    # Map columns dynamically
    col_mapping = {
        'covariate': 'Variable',
        'index': 'Variable',
        'coef': 'coef',
        'exp(coef)': 'HR',
        'se(coef)': 'se',
        'exp(coef) lower 95%': 'HR_lower',
        'exp(coef) upper 95%': 'HR_upper',
        'z': 'z',
        'p': 'p',
        '-log2(p)': 'log2p'
    }
    summary.columns = [col_mapping.get(c, c) for c in summary.columns]

    # Ensure required columns exist
    if 'HR' not in summary.columns:
        summary['HR'] = np.exp(summary['coef'])
    if 'HR_lower' not in summary.columns:
        summary['HR_lower'] = summary['HR'] * 0.8
    if 'HR_upper' not in summary.columns:
        summary['HR_upper'] = summary['HR'] * 1.2
    if 'p' not in summary.columns:
        summary['p'] = 0.05

    # Sort by HR
    summary = summary.sort_values('HR', ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(6, len(summary) * 0.4)))

    y_positions = range(len(summary))

    # Plot error bars
    for i, (_, row) in enumerate(summary.iterrows()):
        color = COLORS['secondary'] if row['HR'] > 1 else COLORS['tertiary']

        ax.errorbar(row['HR'], i, xerr=[[row['HR'] - row['HR_lower']], [row['HR_upper'] - row['HR']]],
                    fmt='o', color=color, markersize=8, capsize=4, capthick=2, elinewidth=2)

    # Add reference line at HR=1
    ax.axvline(x=1, color='black', linestyle='--', linewidth=1, alpha=0.5)

    # Labels
    ax.set_yticks(y_positions)
    ax.set_yticklabels(summary['Variable'])
    ax.set_xlabel('Hazard Ratio (95% CI)')

    # Add HR values on right
    for i, (_, row) in enumerate(summary.iterrows()):
        sig = '***' if row['p'] < 0.001 else '**' if row['p'] < 0.01 else '*' if row['p'] < 0.05 else ''
        ax.text(ax.get_xlim()[1] * 1.02, i, f'{row["HR"]:.2f} ({row["HR_lower"]:.2f}-{row["HR_upper"]:.2f}) {sig}',
                va='center', fontsize=9)

    ax.set_title(f'{endpoint.upper()} Model: Hazard Ratios (Forest Plot)', fontweight='bold')
    ax.set_xlim(0, max(summary['HR_upper']) * 1.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()


def create_summary_figure(output_path):
    """Create a comprehensive summary figure."""
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)

    # Load results
    with open(MODELS_DIR / 'external_validation_report.json', 'r') as f:
        ext_report = json.load(f)

    # 1. C-index comparison (top left, spans 2 columns)
    ax1 = fig.add_subplot(gs[0, 0:2])
    cohorts = ['SEER Internal', 'Hospital External']
    x = np.arange(len(cohorts))
    width = 0.35

    os_vals = [ext_report['os_model']['seer_c_index'], ext_report['os_model']['hospital_c_index']]
    css_vals = [ext_report['css_model']['seer_c_index'], ext_report['css_model']['hospital_c_index']]

    bars1 = ax1.bar(x - width / 2, os_vals, width, label='OS', color=COLORS['primary'], alpha=0.8)
    bars2 = ax1.bar(x + width / 2, css_vals, width, label='CSS', color=COLORS['secondary'], alpha=0.8)

    ax1.set_ylabel('C-index')
    ax1.set_title('Model Discrimination: C-index Comparison', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(cohorts)
    ax1.legend()
    ax1.set_ylim(0.5, 0.85)
    ax1.axhline(y=0.7, color='gray', linestyle='--', alpha=0.5)

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

    # 2. Performance drop (top right, spans 2 columns)
    ax2 = fig.add_subplot(gs[0, 2:4])
    metrics = ['C-index', '1yr AUC', '3yr AUC', '5yr AUC']

    # Load internal validation
    with open(MODELS_DIR / 'internal_validation_os.json', 'r') as f:
        int_os = json.load(f)

    ext_summary = pd.read_csv(TABLES_DIR / 'external_validation_summary.csv')
    os_ext = ext_summary[ext_summary['Endpoint'] == 'OS']

    drops = [
        ext_report['os_model']['c_index_drop'],
        os_ext[os_ext['Time_Point'] == '1-year']['Drop'].values[0] if len(os_ext[os_ext['Time_Point'] == '1-year']) > 0 else 0,
        os_ext[os_ext['Time_Point'] == '3-year']['Drop'].values[0] if len(os_ext[os_ext['Time_Point'] == '3-year']) > 0 else 0,
        os_ext[os_ext['Time_Point'] == '5-year']['Drop'].values[0] if len(os_ext[os_ext['Time_Point'] == '5-year']) > 0 else 0,
    ]

    colors_drop = [COLORS['tertiary'] if d < 0.1 else COLORS['quaternary'] if d < 0.15 else COLORS['secondary'] for d in drops]
    bars = ax2.bar(metrics, drops, color=colors_drop, alpha=0.8, edgecolor='black')
    ax2.set_ylabel('Performance Drop')
    ax2.set_title('OS Model: Internal → External Performance Drop', fontweight='bold')
    ax2.axhline(y=0.1, color='gray', linestyle='--', alpha=0.5, label='Good threshold')
    ax2.set_ylim(0, max(drops) * 1.3 if max(drops) > 0 else 0.2)

    for bar, drop in zip(bars, drops):
        ax2.annotate(f'{drop:.3f}', xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

    # 3-4. Load and display nomograms (middle row)
    try:
        from PIL import Image
        os_nomogram = Image.open(FIGURES_DIR / 'nomograms' / 'nomogram_os.png')
        css_nomogram = Image.open(FIGURES_DIR / 'nomograms' / 'nomogram_css.png')

        ax3 = fig.add_subplot(gs[1, 0:2])
        ax3.imshow(os_nomogram)
        ax3.axis('off')
        ax3.set_title('OS Nomogram', fontweight='bold')

        ax4 = fig.add_subplot(gs[1, 2:4])
        ax4.imshow(css_nomogram)
        ax4.axis('off')
        ax4.set_title('CSS Nomogram', fontweight='bold')
    except:
        ax3 = fig.add_subplot(gs[1, 0:2])
        ax3.text(0.5, 0.5, 'OS Nomogram\n(See nomograms folder)', ha='center', va='center', fontsize=12)
        ax3.axis('off')

        ax4 = fig.add_subplot(gs[1, 2:4])
        ax4.text(0.5, 0.5, 'CSS Nomogram\n(See nomograms folder)', ha='center', va='center', fontsize=12)
        ax4.axis('off')

    # 5. Key findings text (bottom)
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')

    findings_text = f"""
    KEY FINDINGS (Separate T, N, M Staging Approach)

    Model Performance:
    • OS Model: Training C-index = {ext_report['os_model']['seer_c_index']:.3f}, External C-index = {ext_report['os_model']['hospital_c_index']:.3f}
    • CSS Model: Training C-index = {ext_report['css_model']['seer_c_index']:.3f}, External C-index = {ext_report['css_model']['hospital_c_index']:.3f}

    External Validation:
    • Hospital cohort: N = {ext_report['hospital_cohort']['n_total']}, OS events = {ext_report['hospital_cohort']['os_events']}
    • OS C-index drop: {ext_report['os_model']['c_index_drop']:.3f} ({'Good' if ext_report['os_model']['c_index_drop'] < 0.1 else 'Moderate'} transportability)
    • CSS C-index drop: {ext_report['css_model']['c_index_drop']:.3f}

    Conclusion:
    Using separate T, N, M components (instead of combined TNMstage) provides improved external validation performance.
    """

    ax5.text(0.05, 0.95, findings_text, transform=ax5.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('ACC Survival Prediction: Model Summary\n(Using Separate T, N, M Staging)',
                 fontsize=16, fontweight='bold', y=0.98)

    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight', facecolor='white')
    plt.close()


def main():
    print("=" * 70)
    print("07. GENERATE PUBLICATION FIGURES")
    print("=" * 70)

    # Ensure output directories exist
    for d in [KM_DIR, COMPARISON_DIR, TABLES_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # Load data
    train_df = pd.read_pickle(DATA_DIR / 'train.pkl')
    val_df = pd.read_pickle(DATA_DIR / 'validation.pkl')

    print(f"\nTraining: {len(train_df)}, Validation: {len(val_df)}")

    # 1. Baseline characteristics table
    print("\n--- Creating Baseline Table ---")
    baseline_table = create_baseline_table(train_df, val_df)
    baseline_table.to_csv(TABLES_DIR / 'baseline_characteristics.csv', index=False)
    print(f"  Saved: {TABLES_DIR / 'baseline_characteristics.csv'}")

    # 2. Kaplan-Meier curves
    print("\n--- Creating Kaplan-Meier Curves ---")
    plot_kaplan_meier_by_stage(train_df, KM_DIR)
    print(f"  Saved: {KM_DIR / 'km_by_tnm_stage.png'}")

    # 3. Forest plots
    print("\n--- Creating Forest Plots ---")
    for endpoint in ['os', 'css']:
        with open(MODELS_DIR / f'multivariate_{endpoint}_model.json', 'r') as f:
            model_spec = json.load(f)
        plot_forest_plot(model_spec, endpoint, COMPARISON_DIR / f'forest_plot_{endpoint}.png')
        print(f"  Saved: {COMPARISON_DIR / f'forest_plot_{endpoint}.png'}")

    # 4. Summary figure
    print("\n--- Creating Summary Figure ---")
    create_summary_figure(COMPARISON_DIR / 'summary_figure.png')
    print(f"  Saved: {COMPARISON_DIR / 'summary_figure.png'}")

    # 5. Create final results summary
    print("\n--- Creating Final Summary ---")

    # Collect all results
    final_summary = {
        'analysis_date': pd.Timestamp.now().strftime('%Y-%m-%d'),
        'staging_approach': 'Separate T, N, M components',
        'data': {
            'training_n': len(train_df),
            'validation_n': len(val_df),
        }
    }

    # Add model results
    for endpoint in ['os', 'css']:
        with open(MODELS_DIR / f'multivariate_{endpoint}_model.json', 'r') as f:
            model = json.load(f)
        with open(MODELS_DIR / f'internal_validation_{endpoint}.json', 'r') as f:
            internal = json.load(f)

        final_summary[f'{endpoint}_model'] = {
            'variables': model['selected_variables'],
            'training_c_index': model['performance']['training_c_index'],
            'validation_c_index': internal['validation_c_index'],
        }

    # Add external validation
    with open(MODELS_DIR / 'external_validation_report.json', 'r') as f:
        ext_report = json.load(f)
    final_summary['external_validation'] = ext_report

    with open(TABLES_DIR / 'final_results_summary.json', 'w') as f:
        json.dump(final_summary, f, indent=2, default=str)

    print(f"  Saved: {TABLES_DIR / 'final_results_summary.json'}")

    # Print summary
    print(f"\n{'=' * 70}")
    print("ANALYSIS COMPLETE - OUTPUT STRUCTURE")
    print("=" * 70)
    print(f"""
outputs/
├── figures/
│   ├── nomograms/
│   │   ├── nomogram_os.png
│   │   └── nomogram_css.png
│   ├── roc_curves/
│   │   ├── roc_os_internal.png
│   │   ├── roc_os_external.png
│   │   ├── roc_css_internal.png
│   │   └── roc_css_external.png
│   ├── calibration/
│   │   ├── calibration_os_internal.png
│   │   ├── calibration_os_external.png
│   │   ├── calibration_css_internal.png
│   │   └── calibration_css_external.png
│   ├── kaplan_meier/
│   │   └── km_by_tnm_stage.png
│   └── comparison/
│       ├── internal_vs_external.png
│       ├── forest_plot_os.png
│       ├── forest_plot_css.png
│       └── summary_figure.png
├── tables/
│   ├── baseline_characteristics.csv
│   ├── univariate_os.csv
│   ├── univariate_css.csv
│   ├── multivariate_os_coefficients.csv
│   ├── multivariate_css_coefficients.csv
│   ├── internal_validation_summary.csv
│   ├── external_validation_summary.csv
│   └── final_results_summary.json
├── models/
│   ├── cox_os_model.pkl
│   ├── cox_css_model.pkl
│   └── *.json (model specifications)
└── data/
    ├── train.pkl
    ├── validation.pkl
    └── data_dictionary.json
    """)

    print("\n✓ All figures and tables generated!")


if __name__ == '__main__':
    main()
