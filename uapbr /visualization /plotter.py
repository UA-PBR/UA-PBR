"""Visualization utilities for UA-PBR results."""

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

def generate_figure(all_metrics, config, seed_results, p_val, save_path=None):
    """Generate 12-panel publication-quality figure.
    
    Args:
        all_metrics: Dictionary of all metrics
        config: Configuration object
        seed_results: List of per-seed results
        p_val: p-value from statistical test
        save_path: Path to save figure (optional)
    """
    plt.style.use('seaborn-v0_8-paper')
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Risk Comparison Heatmap
    ax1 = plt.subplot(3, 4, 1)
    risk_matrix = np.zeros((len(config.corruption_types), len(config.severities)))
    for i, ct in enumerate(config.corruption_types):
        for j, sev in enumerate(config.severities):
            key = f"{ct}_{sev}"
            risk_matrix[i, j] = np.mean(all_metrics[key]['uapbr_risk'])
    
    im = ax1.imshow(risk_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=0.3)
    ax1.set_xticks(range(len(config.severities)))
    ax1.set_xticklabels([f'{s}' for s in config.severities])
    ax1.set_yticks(range(len(config.corruption_types)))
    ax1.set_yticklabels([ct.replace('_', '\n') for ct in config.corruption_types])
    ax1.set_xlabel('Severity')
    ax1.set_ylabel('Corruption Type')
    ax1.set_title('UA-PBR Risk Heatmap')
    plt.colorbar(im, ax=ax1)
    
    # 2. Risk Reduction Bar Chart
    ax2 = plt.subplot(3, 4, 2)
    conditions = ['Clean'] + [f'{ct[:4]}\n{s}' for ct in config.corruption_types 
                              for s in [0.3, 0.5, 0.7]]
    uapbr_means = [np.mean(all_metrics['clean']['uapbr_risk'])]
    std_means = [np.mean(all_metrics['clean']['std_risk'])]
    
    for ct in config.corruption_types:
        for sev in [0.3, 0.5, 0.7]:
            key = f"{ct}_{sev}"
            uapbr_means.append(np.mean(all_metrics[key]['uapbr_risk']))
            std_means.append(np.mean(all_metrics[key]['std_risk']))
    
    x = np.arange(len(conditions))
    width = 0.35
    
    ax2.bar(x - width/2, uapbr_means[:len(x)], width, label='UA-PBR', color='#2E86AB', alpha=0.8)
    ax2.bar(x + width/2, std_means[:len(x)], width, label='Standard CNN', color='#A23B72', alpha=0.8)
    
    ax2.set_xlabel('Condition')
    ax2.set_ylabel('Risk')
    ax2.set_title('Risk Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(conditions, rotation=45, ha='right')
    ax2.axhline(y=config.lambda_cost, color='red', linestyle='--', linewidth=2, label=f'λ={config.lambda_cost}')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # 3. Risk Reduction Percentage
    ax3 = plt.subplot(3, 4, 3)
    risk_reduction = []
    labels = []
    for ct in config.corruption_types:
        for sev in config.severities:
            key = f"{ct}_{sev}"
            uapbr = np.mean(all_metrics[key]['uapbr_risk'])
            std = np.mean(all_metrics[key]['std_risk'])
            reduction = (std - uapbr) / std * 100 if std > 0 else 0
            risk_reduction.append(reduction)
            labels.append(f'{ct[:4]}\n{sev}')
    
    colors = ['green' if r > 0 else 'red' for r in risk_reduction]
    ax3.bar(range(len(risk_reduction)), risk_reduction, color=colors, alpha=0.7)
    ax3.set_xlabel('Condition')
    ax3.set_ylabel('Risk Reduction (%)')
    ax3.set_title('Risk Reduction vs Standard CNN')
    ax3.set_xticks(range(len(labels)))
    ax3.set_xticklabels(labels, rotation=45, ha='right')
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax3.axhline(y=50, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax3.grid(True, alpha=0.3)
    
    # 4. Acceptance Rate vs Accuracy
    ax4 = plt.subplot(3, 4, 4)
    colors_plot = plt.cm.viridis(np.linspace(0, 1, len(config.corruption_types)))
    for i, ct in enumerate(config.corruption_types):
        accept_rates = []
        acc_accepted = []
        for sev in config.severities:
            key = f"{ct}_{sev}"
            accept_rates.append(np.mean(all_metrics[key]['accept_rate']))
            acc_accepted.append(np.mean(all_metrics[key]['acc_accepted']))
        ax4.scatter(accept_rates, acc_accepted, color=colors_plot[i], s=100, 
                   label=ct, alpha=0.6, edgecolors='black')
    
    ax4.set_xlabel('Acceptance Rate')
    ax4.set_ylabel('Accuracy on Accepted')
    ax4.set_title('Acceptance-Accuracy Tradeoff')
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0.95, 1.01)
    ax4.legend(loc='lower left')
    ax4.grid(True, alpha=0.3)
    
    # 5. Statistical Significance Heatmap
    ax5 = plt.subplot(3, 4, 5)
    p_matrix = np.zeros((len(config.corruption_types), len(config.severities)))
    for i, ct in enumerate(config.corruption_types):
        for j, sev in enumerate(config.severities):
            key = f"{ct}_{sev}"
            t_stat, p_val = stats.ttest_rel(
                all_metrics[key]['uapbr_risk'],
                all_metrics[key]['std_risk']
            )
            p_matrix[i, j] = -np.log10(p_val + 1e-10)
    
    im = ax5.imshow(p_matrix, cmap='viridis', aspect='auto')
    ax5.set_xticks(range(len(config.severities)))
    ax5.set_xticklabels([f'{s}' for s in config.severities])
    ax5.set_yticks(range(len(config.corruption_types)))
    ax5.set_yticklabels([ct.replace('_', '\n') for ct in config.corruption_types])
    ax5.set_xlabel('Severity')
    ax5.set_ylabel('Corruption Type')
    ax5.set_title('Statistical Significance (-log10(p))')
    plt.colorbar(im, ax=ax5)
    
    # 6. Acceptance Rate by Corruption
    ax6 = plt.subplot(3, 4, 6)
    for i, ct in enumerate(config.corruption_types):
        accept_rates = []
        sev_list = []
        for sev in config.severities:
            key = f"{ct}_{sev}"
            accept_rates.append(np.mean(all_metrics[key]['accept_rate']))
            sev_list.append(sev)
        ax6.plot(sev_list, accept_rates, 'o-', label=ct, color=colors_plot[i], linewidth=2, markersize=8)
    
    ax6.set_xlabel('Severity')
    ax6.set_ylabel('Acceptance Rate')
    ax6.set_title('Acceptance Rate vs Severity')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 7. F1 Score by Corruption
    ax7 = plt.subplot(3, 4, 7)
    for i, ct in enumerate(config.corruption_types):
        f1_scores = []
        sev_list = []
        for sev in config.severities:
            key = f"{ct}_{sev}"
            f1_scores.append(np.mean(all_metrics[key]['f1_accepted']))
            sev_list.append(sev)
        ax7.plot(sev_list, f1_scores, 'o-', label=ct, color=colors_plot[i], linewidth=2, markersize=8)
    
    ax7.set_xlabel('Severity')
    ax7.set_ylabel('F1 Score')
    ax7.set_title('F1 Score vs Severity')
    ax7.legend()
    ax7.set_ylim(0.99, 1.01)
    ax7.grid(True, alpha=0.3)
    
    # 8. Threshold Distribution
    ax8 = plt.subplot(3, 4, 8)
    tau_phys = all_metrics['clean']['tau_phy']
    tau_uncs = all_metrics['clean']['tau_unc']
    
    ax8.scatter(tau_phys, tau_uncs, s=100, alpha=0.6, c='purple', edgecolors='black')
    ax8.set_xlabel('τ_phy')
    ax8.set_ylabel('τ_unc')
    ax8.set_title(f'Threshold Distribution (n={len(tau_phys)})')
    ax8.grid(True, alpha=0.3)
    ax8.scatter(np.mean(tau_phys), np.mean(tau_uncs), s=200, c='red', marker='*',
               label=f'Mean: ({np.mean(tau_phys):.3f}, {np.mean(tau_uncs):.3f})')
    ax8.legend()
    
    # 9. Lipschitz Constant Distribution
    ax9 = plt.subplot(3, 4, 9)
    L_values = [r['L'] for r in seed_results]
    ax9.hist(L_values, bins=10, alpha=0.7, edgecolor='black', color='teal')
    ax9.axvline(np.mean(L_values), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(L_values):.3f}')
    ax9.set_xlabel('Lipschitz Constant L')
    ax9.set_ylabel('Frequency')
    ax9.set_title('Lipschitz Constant Distribution')
    ax9.legend()
    ax9.grid(True, alpha=0.3)
    
    # 10. Rejection Components
    ax10 = plt.subplot(3, 4, 10)
    rej_phy = np.mean(all_metrics['clean']['rejection_phy'])
    rej_unc = np.mean(all_metrics['clean']['rejection_unc'])
    accept = 1 - rej_phy - rej_unc
    
    ax10.pie([accept, rej_phy, rej_unc], 
             labels=['Accept', 'Reject (Physics)', 'Reject (Uncertainty)'],
             colors=['green', 'red', 'orange'],
             autopct='%1.1f%%',
             explode=[0, 0.05, 0.05])
    ax10.set_title('Clean Test: Rejection Breakdown')
    
    # 11. CNN Training Progress
    ax11 = plt.subplot(3, 4, 11)
    accuracies = [r['cnn_best_acc'] for r in seed_results]
    ax11.bar(range(len(accuracies)), accuracies, color='steelblue', alpha=0.7)
    ax11.set_xlabel('Seed')
    ax11.set_ylabel('Best Accuracy')
    ax11.set_title('Bayesian CNN Best Accuracy')
    ax11.set_ylim(0.9, 1.0)
    ax11.set_xticks(range(len(accuracies)))
    ax11.grid(True, alpha=0.3)
    
    # 12. Summary Text
    ax12 = plt.subplot(3, 4, 12)
    ax12.axis('off')
    summary_text = f"""
PRODUCTION RESULTS SUMMARY
==========================
Seeds: {config.n_seeds}
Samples: {config.n_samples}

CLEAN TEST:
UA-PBR Risk: {np.mean(all_metrics['clean']['uapbr_risk']):.4f}±{np.std(all_metrics['clean']['uapbr_risk']):.4f}
Std CNN Risk: {np.mean(all_metrics['clean']['std_risk']):.4f}±{np.std(all_metrics['clean']['std_risk']):.4f}
Accept Rate: {np.mean(all_metrics['clean']['accept_rate']):.2f}±{np.std(all_metrics['clean']['accept_rate']):.2f}
Acc Accepted: {np.mean(all_metrics['clean']['acc_accepted']):.4f}±{np.std(all_metrics['clean']['acc_accepted']):.4f}

BEST IMPROVEMENTS:
Gaussian (0.9): {(np.mean(all_metrics['gaussian_0.9']['std_risk']) - np.mean(all_metrics['gaussian_0.9']['uapbr_risk']))/np.mean(all_metrics['gaussian_0.9']['std_risk'])*100:.1f}%
Physics Violating (0.9): {(np.mean(all_metrics['physics_violating_0.9']['std_risk']) - np.mean(all_metrics['physics_violating_0.9']['uapbr_risk']))/np.mean(all_metrics['physics_violating_0.9']['std_risk'])*100:.1f}%

p-value: {p_val:.4f}
"""
    ax12.text(0.05, 0.95, summary_text, fontsize=9, va='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.savefig(save_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    
    plt.show()
