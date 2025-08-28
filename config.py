# config.py

# ==============================================================================
# CONFIGURATION FILE
# ==============================================================================

# --- Data and Execution Parameters ---
EXCEL_PATH = "corrected_selected.csv"
TARGET_COL = 'DIAGNOSIS'
N_SPLITS = 5
NUM_CLASSES = 3
DEVICE = 'cpu' # Use 'cuda' if GPU is available

# --- Feature Selection ---
# Define the range of columns to be used as features.
# Set FEATURE_END_COL to None to select all columns from start to the end.
FEATURE_START_COL = '11715989_a_at_JTB'
FEATURE_END_COL = None 


# List the names of any columns that should be treated as categorical.
# These will be one-hot encoded.
CATEGORICAL_FEATURES = [] 

# --- LEO-CVAE Hyperparameters ---
LEO_LATENT_DIM = 16
LEO_HIDDEN_DIM = 64
LEO_EPOCHS = 500
LEO_LEARNING_RATE = 1e-3
LEO_WEIGHT_DECAY = 1e-5
LEO_BATCH_SIZE = 32
LEO_PATIENCE = 25
LEO_GAMMA = 2.5  # Controls focus on high-entropy samples
LEO_KNN_K = 15     # Neighbors for local entropy calculation
LEO_BETA = 4    # Weight for the KLD loss term
LEO_MIN_KLD = 0.1 # Floor for KLD to prevent collapse

# --- MLP Classifier Hyperparameters ---
MLP_HIDDEN_DIMS = [32]
MLP_DROPOUT = 0.5
MLP_EPOCHS = 200
MLP_LEARNING_RATE = 1e-4
MLP_WEIGHT_DECAY = 1e-3
MLP_BATCH_SIZE = 32
MLP_PATIENCE = 20
MLP_LABEL_SMOOTHING = 0.1 