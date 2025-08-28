# LEO_CVAE.py
# Implements the LEO-CVAE framework for uncertainty-aware generative oversampling.
# This module contains the CVAE architecture, the Local Entropy Score (LES) calculation,
# the Local Entropy-Weighted Loss (LEWL), the entropy-guided sampling strategy,
# and the main function to apply the oversampling pipeline.

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# 1. LEO-CVAE MODEL DEFINITION
# ==============================================================================
class LEO_CVAE_Model(nn.Module):
    """
    The Conditional Variational Autoencoder (CVAE) architecture for the LEO-CVAE framework.
    This model is trained using an uncertainty-aware loss function to learn a robust
    representation of the data, particularly in high-entropy regions.
    """
    def __init__(self, input_dim, num_classes, latent_dim=16, hidden_dim=32):
        super(LEO_CVAE_Model, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.latent_dim = latent_dim

        # Encoder: Maps a sample and its class condition to a latent distribution
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + num_classes, hidden_dim),
            nn.LeakyReLU(0.2), nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2), nn.Dropout(0.1)
        )
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)      # Predicts the mean (mu)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim) # Predicts the log variance (logvar)

        # Decoder: Reconstructs a sample from the latent space, conditioned on its class
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + num_classes, hidden_dim // 2),
            nn.LeakyReLU(0.2), nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LeakyReLU(0.2), nn.Dropout(0.1),
            nn.Linear(hidden_dim, input_dim) # Output layer reconstructs the original sample
        )

    def reparameterize(self, mu, logvar):
        """Performs the reparameterization trick for differentiable sampling from the latent space."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, y):
        """Defines the forward pass of the CVAE."""
        y_onehot = torch.nn.functional.one_hot(y, num_classes=self.num_classes).float()
        combined_input = torch.cat([x, y_onehot], dim=1)
        h = self.encoder(combined_input)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        combined_latent = torch.cat([z, y_onehot], dim=1)
        recon_x = self.decoder(combined_latent)
        return recon_x, mu, logvar

    def loss_function(self, recon_x, x, mu, logvar, H, class_weights, beta=1.0, gamma=1.0, min_kld=0.05):
        """
        Calculates the Local Entropy-Weighted Loss (LEWL), the core of the LEO-CVAE framework.
        This loss function synergistically combines a weighted reconstruction error with a regularized
        KL divergence to focus the learning process on uncertain, high-entropy regions.
        """
        # 1. Local Entropy-Weighted Reconstruction Loss (LW-Recon)
        recon_loss_per_sample = nn.functional.mse_loss(recon_x, x, reduction='none').sum(dim=1)
        entropy_weights = torch.pow(1.0 + H, gamma)
        final_weights = class_weights * entropy_weights
        weighted_recon_loss = (final_weights * recon_loss_per_sample).mean()

        # 2. KL Divergence with Posterior Collapse Prevention
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
        # Apply a minimum threshold to prevent posterior collapse
        kld_loss = torch.max(kld_loss, torch.tensor(min_kld, device=kld_loss.device))

        # 3. Final Composite Loss
        total_loss = weighted_recon_loss + beta * kld_loss
        return total_loss, weighted_recon_loss, kld_loss

# ==============================================================================
# 2. UTILITY CLASSES AND FUNCTIONS FOR LEO-CVAE
# ==============================================================================
class CVAEEarlyStopping:
    """Implements an early stopping mechanism to prevent overfitting during model training by monitoring validation loss."""
    def __init__(self, patience=10, min_delta=1e-4, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = float('inf') if mode == 'min' else float('-inf')
        self.early_stop = False

    def __call__(self, val_loss):
        """Checks if the validation loss metric has improved."""
        improved = (val_loss < self.best_score - self.min_delta) if self.mode == 'min' else (val_loss > self.best_score + self.min_delta)
        if improved:
            self.best_score = val_loss
            self.counter = 0
        else:
            self.counter += 1
        if self.counter >= self.patience:
            self.early_stop = True
        return self.early_stop

def calculate_reconstruction_metrics(original, reconstructed):
    """Calculates reconstruction quality metrics to assess how well the CVAE is learning the data distribution."""
    mse = np.mean((original - reconstructed) ** 2)
    mae = np.mean(np.abs(original - reconstructed))
    orig_flat, recon_flat = original.flatten(), reconstructed.flatten()
    if np.std(orig_flat) == 0 or np.std(recon_flat) == 0:
        correlation = 0.0
    else:
        correlation = np.corrcoef(orig_flat, recon_flat)[0, 1]
    return {'mse': mse, 'mae': mae, 'correlation': correlation}

def plot_cvae_training_history(history, fold_idx):
    """Visualizes the training and validation history of the LEO-CVAE model, including total loss, KLD, and reconstruction components."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'LEO-CVAE Training History - Fold {fold_idx}', fontsize=16)

    # Plot 1: Total Loss
    axes[0, 0].plot(history['train_losses'], label='Train Loss')
    axes[0, 0].plot(history['val_losses'], label='Val Loss', linestyle='--')
    axes[0, 0].set_title('Total Loss'); axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_yscale('log')

    # Plot 2: KLD Loss
    axes[0, 1].plot(history['train_kld_losses'], label='Train KLD')
    axes[0, 1].plot(history['val_kld_losses'], label='Val KLD', linestyle='--')
    axes[0, 1].axhline(y=0.05, color='gray', linestyle=':', label='Collapse Threshold')
    axes[0, 1].set_title('KLD Loss'); axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_yscale('log')

    # Plot 3: Reconstruction Loss
    axes[1, 0].plot(history['train_recon_losses'], label='Train Recon Loss')
    axes[1, 0].plot(history['val_recon_losses'], label='Val Recon Loss', linestyle='--')
    axes[1, 0].set_title('Reconstruction Loss'); axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_yscale('log')

    # Plot 4: Reconstruction Correlation
    corr = [m['correlation'] for m in history['reconstruction_metrics']]
    axes[1, 1].plot(corr, label='Recon Correlation')
    axes[1, 1].set_title('Reconstruction Correlation'); axes[1, 1].legend(); axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    output_path = f'results/cvae_history_fold_{fold_idx}.png'
    plt.savefig(output_path, dpi=300)
    print(f"Saved CVAE training plot to {output_path}")

    plt.show()

# ==============================================================================
# 3. LEO-CVAE TRAINING AND OVERSAMPLING LOGIC
# ==============================================================================
def calculate_local_entropy_scores(X, labels, num_classes, k):
    """
    Implements the Local Entropy Score (LES) calculation. LES serves as a quantitative proxy
    for sample uncertainty by measuring the degree of class overlap in a sample's local
    neighborhood using Shannon entropy.
    """
    print(f"Quantifying sample uncertainty using Local Entropy Score (LES) with k={k}...")
    knn = NearestNeighbors(n_neighbors=k + 1)
    knn.fit(X)
    _, indices = knn.kneighbors(X)
    entropy_scores = np.zeros(len(X))

    for i in range(len(X)):
        neighbor_indices = indices[i, 1:]
        neighbor_labels = labels[neighbor_indices]
        class_counts = Counter(neighbor_labels)
        local_entropy = 0.0
        for class_idx in range(num_classes):
            count = class_counts.get(class_idx, 0)
            if count > 0:
                probability = count / k
                local_entropy -= probability * np.log2(probability)
        entropy_scores[i] = local_entropy
    return torch.tensor(entropy_scores, dtype=torch.float32)

def _train_cvae_model(cvae, train_loader, val_loader, optimizer, epochs, beta, gamma, class_weight_map, device, patience=15, min_delta=1e-4, min_kld=0.05):
    """Orchestrates the training and validation loop for the LEO-CVAE model using the LEWL objective."""
    class_weight_map = class_weight_map.to(device)
    early_stopping = CVAEEarlyStopping(patience=patience, min_delta=min_delta)
    history = {
        'train_losses': [], 'val_losses': [], 'train_recon_losses': [],
        'train_kld_losses': [], 'val_recon_losses': [], 'val_kld_losses': [],
        'reconstruction_metrics': []
    }
    best_val_loss, best_model_state, kld_collapse_warnings = float('inf'), None, 0

    print(f"\n--- LEO-CVAE Training ---")
    print(f"Epochs: {epochs}, Patience: {patience}, Gamma: {gamma:.2f}, Beta: {beta:.2f}, Min KLD: {min_kld:.3f}")

    for epoch in range(epochs):
        # --- Training Phase ---
        cvae.train()
        epoch_train_loss, epoch_train_recon, epoch_train_kld, num_batches = 0.0, 0.0, 0.0, 0
        epoch_recon_metrics = {'mse': [], 'mae': [], 'correlation': []}
        
        for batch_x, batch_y, batch_h in train_loader:
            batch_x, batch_y, batch_h = batch_x.to(device), batch_y.to(device), batch_h.to(device)
            optimizer.zero_grad()
            recon_x, mu, logvar = cvae(batch_x, batch_y)
            batch_class_weights = class_weight_map[batch_y]
            
            total_loss, recon_loss, kld_loss = cvae.loss_function(
                recon_x, batch_x, mu, logvar, batch_h, batch_class_weights, beta, gamma, min_kld=min_kld
            )
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(cvae.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_train_loss += total_loss.item()
            epoch_train_recon += recon_loss.item()
            epoch_train_kld += kld_loss.item()
            num_batches += 1
            
            if num_batches == 1:
                with torch.no_grad():
                    metrics = calculate_reconstruction_metrics(batch_x.cpu().numpy()[:32], recon_x.cpu().numpy()[:32])
                    for key, value in metrics.items():
                        epoch_recon_metrics[key].append(value)

        avg_train_loss = epoch_train_loss / num_batches
        avg_train_kld = epoch_train_kld / num_batches
        if avg_train_kld < min_kld * 1.5:
            kld_collapse_warnings += 1

        # --- Validation Phase ---
        cvae.eval()
        epoch_val_loss, epoch_val_recon, epoch_val_kld, val_num_batches = 0.0, 0.0, 0.0, 0
        with torch.no_grad():
            for batch_x, batch_y, batch_h in val_loader:
                batch_x, batch_y, batch_h = batch_x.to(device), batch_y.to(device), batch_h.to(device)
                recon_x, mu, logvar = cvae(batch_x, batch_y)
                batch_class_weights = class_weight_map[batch_y]
                total_loss, recon_loss, kld_loss = cvae.loss_function(
                    recon_x, batch_x, mu, logvar, batch_h, batch_class_weights, beta, gamma, min_kld=min_kld
                )
                epoch_val_loss += total_loss.item()
                epoch_val_recon += recon_loss.item()
                epoch_val_kld += kld_loss.item()
                val_num_batches += 1
        
        avg_val_loss = epoch_val_loss / val_num_batches
        avg_val_recon = epoch_val_recon / val_num_batches
        avg_val_kld = epoch_val_kld / val_num_batches

        # --- Store metrics and print progress ---
        history['train_losses'].append(avg_train_loss)
        history['val_losses'].append(avg_val_loss)
        history['train_recon_losses'].append(epoch_train_recon / num_batches)
        history['val_recon_losses'].append(avg_val_recon)
        history['train_kld_losses'].append(avg_train_kld)
        history['val_kld_losses'].append(avg_val_kld)
        avg_recon_metrics = {k: np.mean(v) if v else 0.0 for k, v in epoch_recon_metrics.items()}
        history['reconstruction_metrics'].append(avg_recon_metrics)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = cvae.state_dict().copy()

        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            corr = avg_recon_metrics.get('correlation', 0)
            print(f"Epoch {epoch+1:3d}/{epochs}: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} (R: {avg_val_recon:.4f}, K: {avg_val_kld:.4f}) | Corr: {corr:.3f}")

        if early_stopping(avg_val_loss):
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    if best_model_state:
        cvae.load_state_dict(best_model_state)
        print(f"Loaded best model with validation loss: {best_val_loss:.4f}")

    if kld_collapse_warnings > epochs * 0.3:
        print(f"⚠️ KLD collapse may have occurred in {kld_collapse_warnings}/{len(history['train_losses'])} epochs.")
        
    return history

def apply_leo_cvae(X, y, cfg, fold_idx=0, plot_history=True):
    """
    The main entry point for the LEO-CVAE framework. This function orchestrates the entire
    uncertainty-aware oversampling pipeline, from calculating LES to training the model
    and generating synthetic samples.
    """
    device = torch.device(cfg.DEVICE)
    print("Original class distribution:", Counter(y))
    
    # Calculate class weights for the loss function
    class_counts_orig = Counter(y)
    total_samples_orig = len(y)
    cvae_class_weights = torch.tensor([
        total_samples_orig / (cfg.NUM_CLASSES * class_counts_orig.get(i, 1)) 
        for i in sorted(class_counts_orig.keys())
    ], dtype=torch.float32)
    print(f"CVAE Class Weights: {[f'{w:.2f}' for w in cvae_class_weights.numpy()]}")
    
    # Calculate entropy scores
    H_scores = calculate_local_entropy_scores(X, y, num_classes=cfg.NUM_CLASSES, k=cfg.LEO_KNN_K)
    print(f"Entropy Score Dist: Min={H_scores.min():.4f}, Max={H_scores.max():.4f}, Mean={H_scores.mean():.4f}")
    
    # Prepare data for CVAE training
    X_train, X_val, y_train, y_val, H_train, H_val = train_test_split(X, y, H_scores.numpy(), test_size=0.2, random_state=42, stratify=y)
    
    cvae = LEO_CVAE_Model(input_dim=X.shape[1], num_classes=cfg.NUM_CLASSES, latent_dim=cfg.LEO_LATENT_DIM, hidden_dim=cfg.LEO_HIDDEN_DIM).to(device)
    optimizer = optim.Adam(cvae.parameters(), lr=cfg.LEO_LEARNING_RATE, weight_decay=cfg.LEO_WEIGHT_DECAY)
    
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train), torch.FloatTensor(H_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val), torch.FloatTensor(H_val))
    train_loader = DataLoader(train_dataset, batch_size=cfg.LEO_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.LEO_BATCH_SIZE, shuffle=False)
    
    # Train the CVAE model
    history = _train_cvae_model(cvae, train_loader, val_loader, optimizer, cfg.LEO_EPOCHS, cfg.LEO_BETA, cfg.LEO_GAMMA, cvae_class_weights, device, patience=cfg.LEO_PATIENCE, min_kld=cfg.LEO_MIN_KLD)
    
    if plot_history:
        plot_cvae_training_history(history, fold_idx)
    
    # --- Entropy-Guided Sample Generation ---
    print("\nPerforming entropy-guided generation of synthetic samples...")
    cvae.eval()
    with torch.no_grad():
        class_counts = Counter(y_train)
        if len(class_counts) < 2:
            return X, y
            
        majority_class_count = max(class_counts.values())
        X_resampled_list, y_resampled_list = [X], [y]

        for class_idx, count in class_counts.items():
            if count < majority_class_count:
                # 1. Determine number of samples to generate
                num_to_generate = majority_class_count - count
                minority_indices = np.where(y_train == class_idx)[0]
                minority_H_scores = torch.FloatTensor(H_train)[minority_indices]
                
                # 2. Establish entropy-guided sampling distribution for seed selection
                focused_H = torch.pow(1.0 + minority_H_scores, cfg.LEO_GAMMA)
                H_sum = torch.sum(focused_H)
                probs = focused_H / H_sum if H_sum > 0 else torch.ones(len(minority_indices)) / len(minority_indices)
                
                # 3. Sample seed instances, prioritizing those in high-entropy regions
                seed_indices = np.random.choice(minority_indices, size=num_to_generate, p=probs.numpy())
                seed_samples = torch.FloatTensor(X_train[seed_indices]).to(device)
                seed_labels = torch.LongTensor(y_train[seed_indices]).to(device)
                
                # 4. Generate new synthetic samples from the seeds using the trained CVAE
                y_onehot_seeds = torch.nn.functional.one_hot(seed_labels, num_classes=cfg.NUM_CLASSES).float()
                h_seeds = cvae.encoder(torch.cat([seed_samples, y_onehot_seeds], dim=1))
                mu_seeds, logvar_seeds = cvae.fc_mu(h_seeds), cvae.fc_logvar(h_seeds)
                z_new = cvae.reparameterize(mu_seeds, logvar_seeds)
                synthetic_samples = cvae.decoder(torch.cat([z_new, y_onehot_seeds], dim=1)).cpu().numpy()
                
                # 5. Report quality and append new data to the training set
                quality_metrics = calculate_reconstruction_metrics(X_train[seed_indices], synthetic_samples)
                print(f"Class {class_idx}: Generated {num_to_generate} samples, Quality (corr): {quality_metrics['correlation']:.3f}")
                X_resampled_list.append(synthetic_samples)
                y_resampled_list.append(np.full(num_to_generate, class_idx))

    # Combine original and synthetic data
    X_resampled = np.concatenate(X_resampled_list)
    y_resampled = np.concatenate(y_resampled_list)
    
    print("Resampled class distribution:", Counter(y_resampled))
    return X_resampled, y_resampled