# Results Directory

This directory stores experiment results.

## File Structure

- `config.json` - Experiment configuration
- `results_summary.csv` - Summary results table
- `all_metrics.pkl` - Raw metrics for all seeds
- `checkpoint_seed_*.pt` - Model checkpoints

## Results Format

Results are saved in CSV format with columns:
- `Condition`: Test condition (clean or corruption type_severity)
- `UA-PBR_Risk`: UA-PBR empirical risk (mean±std)
- `Std_Risk`: Standard CNN risk (mean±std)
- `Accept_Rate`: Acceptance rate
- `Acc_Accepted`: Accuracy on accepted samples
- `F1_Accepted`: F1 score on accepted samples

## Reproducibility

All experiments use fixed random seeds (42 + seed_index) for complete reproducibility.
