# main.py
# Main script to run the entire LEO-CVAE experiment.
# It orchestrates the process by calling functions from the specialized modules.

import torch
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import random
import os 

# Import project modules 
import config as cfg
from  src.leo_cvae import apply_leo_cvae 
from src.mlp_classifier import MLPClassifier, train_mlp_classifier, evaluate_mlp_classifier

def set_seeds(seed=42):
    """Sets random seeds for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    """Main function to run the entire experiment."""
    set_seeds(seed=42)
    device = torch.device(cfg.DEVICE)
    
    if not os.path.exists('results'):
        os.makedirs('results')
        print("Created 'results' directory.")

    print(f"Loading data from {cfg.EXCEL_PATH}...")
    df = pd.read_csv(cfg.EXCEL_PATH)
    
    # --- 1. Feature Selection and Preprocessing ---
    # Define the initial set of features to consider
    end_col_index = df.columns.get_loc(cfg.FEATURE_END_COL) if cfg.FEATURE_END_COL else None
    initial_feature_cols = df.loc[:, cfg.FEATURE_START_COL:end_col_index].columns.tolist()
    if cfg.TARGET_COL in initial_feature_cols:
        initial_feature_cols.remove(cfg.TARGET_COL)

    #  Preprocess Categorical Features
    if cfg.CATEGORICAL_FEATURES:
        print(f"Applying one-hot encoding to: {cfg.CATEGORICAL_FEATURES}")
        # Use pandas get_dummies for robust one-hot encoding
        df_processed = pd.get_dummies(df, columns=cfg.CATEGORICAL_FEATURES, prefix=cfg.CATEGORICAL_FEATURES, drop_first=False)
        
        # Identify numerical columns (those not in the categorical list)
        numerical_cols = [col for col in initial_feature_cols if col not in cfg.CATEGORICAL_FEATURES]
        
        # Get the names of the new one-hot encoded columns
        one_hot_cols = [c for c in df_processed.columns if any(c.startswith(f"{cat_col}_") for cat_col in cfg.CATEGORICAL_FEATURES)]
        
        # Combine numerical and new one-hot encoded columns for the final feature set
        final_feature_cols = numerical_cols + one_hot_cols
    else:
        # If no categorical features are specified, proceed as before
        df_processed = df
        final_feature_cols = initial_feature_cols

    print(f"Found {len(final_feature_cols)} features after processing.")
    
    # Create the final feature matrix X and target vector y
    X = df_processed[final_feature_cols].values
    y = LabelEncoder().fit_transform(df_processed[cfg.TARGET_COL])
    input_dim = X.shape[1]
    
    # --- 2. Cross-Validation Loop ---
    skf = StratifiedKFold(n_splits=cfg.N_SPLITS, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_index, val_index) in enumerate(skf.split(X, y)):
        print(f"\n{'='*25} FOLD {fold+1}/{cfg.N_SPLITS} {'='*25}")
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        # Apply LEO-CVAE Oversampling
        X_train_resampled, y_train_resampled = apply_leo_cvae(
            X_train, y_train, cfg, fold_idx=fold+1, plot_history=True
        )

        # Scale data and create DataLoaders
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_resampled)
        X_val_scaled = scaler.transform(X_val)
        
        train_dataset = TensorDataset(torch.FloatTensor(X_train_scaled), torch.LongTensor(y_train_resampled))
        val_dataset = TensorDataset(torch.FloatTensor(X_val_scaled), torch.LongTensor(y_val))
        train_loader = DataLoader(train_dataset, batch_size=cfg.MLP_BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=cfg.MLP_BATCH_SIZE, shuffle=False)

        # Train and Evaluate the Classifier
        model = MLPClassifier(input_dim=input_dim, hidden_dims=cfg.MLP_HIDDEN_DIMS, dropout_rate=cfg.MLP_DROPOUT, num_classes=cfg.NUM_CLASSES)
        print(f"\nStarting MLP classifier training for fold {fold+1}...")
        trained_model = train_mlp_classifier(model, train_loader, val_loader, cfg)

        print(f"Evaluating fold {fold+1}...")
        results = evaluate_mlp_classifier(trained_model, val_loader, device, num_classes=cfg.NUM_CLASSES)
        fold_results.append(results)
        
        metric_key = 'auc' if cfg.NUM_CLASSES == 2 else 'f1_weighted'
        print(f"Fold {fold+1} {metric_key.upper()}: {results.get(metric_key, 0):.4f}")

# --- Final Summary ---
    print(f"\n{'='*20} {cfg.N_SPLITS}-FOLD CROSS-VALIDATION SUMMARY {'='*20}")
    results_df = pd.DataFrame(fold_results)
    mean_metrics = results_df.mean()
    std_metrics = results_df.std()

    # Print summary to console
    print(f"{'Metric':<22} | {'Mean':<10} | {'Std Dev':<10}")
    print(f"{'-'*22}-+-{'-'*10}-+-{'-'*10}")
    for metric in mean_metrics.index:
        mean_val = mean_metrics[metric]
        std_val = std_metrics[metric]
        print(f"{metric:<22} | {mean_val:<10.4f} | {std_val:<10.4f}")

    # --- Save summary results to an Excel file ---
    summary_df = pd.DataFrame({'Mean': mean_metrics, 'Std Dev': std_metrics})
    output_filename = f'results/final_leo_cvae_summary_{cfg.TARGET_COL}.xlsx'
    summary_df.to_excel(output_filename, index=True, engine='openpyxl')
    print(f"\nSummary results saved to '{output_filename}'")

if __name__ == "__main__":
    main()


    