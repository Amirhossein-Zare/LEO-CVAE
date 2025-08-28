# mlp_classifier.py
# This module contains the MLP classifier architecture and its associated
# training and evaluation functions. To adapt for multiclass classification,
# changes would be focused here.

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import (f1_score, roc_auc_score, precision_recall_curve,
                             auc, average_precision_score)

# ==============================================================================
# 1. MLP MODEL DEFINITION AND UTILITIES
# ==============================================================================
class MLPClassifier(nn.Module):
    """A standard Multi-Layer Perceptron for classification."""
    def __init__(self, input_dim, hidden_dims=[64, 32], dropout_rate=0.2, num_classes=2):
        super(MLPClassifier, self).__init__()
        self.num_classes = num_classes
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        # Output layer size depends on the number of classes
        output_dim = 1 if num_classes == 2 else num_classes
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class ClassifierEarlyStopping:
    """Early stopping handler for classifier training."""
    def __init__(self, patience=10, min_delta=0.001, mode='max'):
        self.patience, self.min_delta, self.mode = patience, min_delta, mode
        self.counter, self.best_score = 0, float('-inf') if mode == 'max' else float('inf')
    def __call__(self, score):
        improved = (score > self.best_score + self.min_delta) if self.mode == 'max' else (score < self.best_score - self.min_delta)
        if improved: self.best_score, self.counter = score, 0
        else: self.counter += 1
        return self.counter >= self.patience

# ==============================================================================
# 2. MLP TRAINING AND EVALUATION LOGIC
# ==============================================================================
def train_mlp_classifier(model, train_loader, val_loader, cfg):
    """Training loop for the MLP classifier."""
    device = torch.device(cfg.DEVICE); model.to(device)
    # Choose loss function based on number of classes
    criterion = nn.BCEWithLogitsLoss() if cfg.NUM_CLASSES == 2 else nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.MLP_LEARNING_RATE, weight_decay=cfg.MLP_WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.7)
    early_stopping = ClassifierEarlyStopping(patience=cfg.MLP_PATIENCE, mode='max')
    best_val_metric, best_model_state = 0.0, None

    for epoch in range(cfg.MLP_EPOCHS):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad(); outputs = model(batch_x)
            # Adapt target shape for loss function
            if cfg.NUM_CLASSES == 2: loss = criterion(outputs, batch_y.float().unsqueeze(1))
            else: loss = criterion(outputs, batch_y.long())
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5); optimizer.step()

        model.eval()
        val_probs, val_labels = [], []; all_outputs = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                if cfg.NUM_CLASSES == 2: val_probs.extend(torch.sigmoid(outputs).detach().cpu().numpy())
                else: all_outputs.extend(torch.softmax(outputs, dim=1).detach().cpu().numpy())
                val_labels.extend(batch_y.cpu().numpy())

        if cfg.NUM_CLASSES == 2:
            # For binary classification, standard AUC is used.
            current_metric = roc_auc_score(val_labels, val_probs)
        else:
            # Use AUC micro for multi-class early stopping, which requires probabilities.
            try:
                current_metric = roc_auc_score(val_labels, all_outputs, multi_class='ovr', average='micro')
            except ValueError:
                # Fallback for rare cases where a validation batch might lack class diversity
                current_metric = 0.5
        
        scheduler.step(current_metric)
        if current_metric > best_val_metric: best_val_metric, best_model_state = current_metric, model.state_dict().copy()
        if early_stopping(current_metric): break

    if best_model_state: model.load_state_dict(best_model_state)
    return model

def evaluate_mlp_classifier(model, test_loader, device, num_classes=2):
    """
    Evaluates the trained MLP classifier and returns performance metrics
    as specified in the paper.
    """
    model.to(device).eval()
    all_predictions, all_probabilities, all_labels = [], [], []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            outputs = model(batch_x)
            
            if num_classes == 2:
                probs = torch.sigmoid(outputs).cpu().numpy()
                preds = (probs > 0.5).astype(int)
                all_probabilities.extend(probs.flatten())
            else: # Multiclass
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)
                all_probabilities.extend(probs)
                
            all_predictions.extend(preds.flatten())
            all_labels.extend(batch_y.cpu().numpy())

    if num_classes == 2:
        # Calculate recall and precision for the AUPRC
        precision, recall, _ = precision_recall_curve(all_labels, all_probabilities)
        
        return {
            'f1_score': f1_score(all_labels, all_predictions, zero_division=0),
            'auc': roc_auc_score(all_labels, all_probabilities),
            'auprc': auc(recall, precision),
        }
    else: # Multiclass metrics
        all_probabilities_arr = np.array(all_probabilities)
        
        # Calculate AUPRC for both micro and macro
        auprc_micro = average_precision_score(all_labels, all_probabilities_arr, average="micro")
        auprc_macro = average_precision_score(all_labels, all_probabilities_arr, average="macro")

        return {
            'f1_micro': f1_score(all_labels, all_predictions, average='micro', zero_division=0),
            'auc_micro': roc_auc_score(all_labels, all_probabilities_arr, multi_class='ovr', average='micro'),
            'auprc_micro': auprc_micro,
            'f1_macro': f1_score(all_labels, all_predictions, average='macro', zero_division=0),
            'auc_macro': roc_auc_score(all_labels, all_probabilities_arr, multi_class='ovr', average='macro'),
            'auprc_macro': auprc_macro,
        }