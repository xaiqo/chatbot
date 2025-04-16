# Model configuration parameters
USE_GPU = True
DEVICE = 'cuda' if USE_GPU else 'cpu'

# Model parameters
EMBED_DIM = 512
NUM_BLOCKS = 6
HEADS = 8
HIDDEN_SIZE = 2048

# Regularization parameters
DROPOUT_RATE = 0.1
WEIGHT_DECAY = 0.01

# BPE tokenization
BPE_VOCAB_SIZE = 32768
BPE_MIN_FREQUENCY = 2

# Adam optimizer
LEARNING_RATE = 3e-4
BETA1 = 0.9
BETA2 = 0.999
EPSILON = 1e-8

# Training parameters
BATCH_SIZE = 32
EPOCHS = 10
MAX_SEQ_LENGTH = 512

# Inference parameters
MAX_RESPONSE_LENGTH = 256
TEMPERATURE = 0.7
TOP_K = 40
TOP_P = 0.9

# Paths
PDF_DIR = "../pdf_documents"
PROCESSED_DATA_DIR = "C:\\data\\processed_data"
MODEL_SAVE_PATH = "trained_model.npy"
TOKENIZER_SAVE_PATH = "tokenizer.json"