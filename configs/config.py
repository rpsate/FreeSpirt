# train parameter
TRAIN_DATASET_SIZE = 48000
VALID_DATASET_SIZE = 6000
TEST_DATASET_SIZE = 6000
MAX_NODE_SIZE = 450
BATCH_SIZE = 4096
NUM_EPOCHS = 1
LEARNING_RATE = 0.005
WEIGHT_DECAY = 0.01
THRESHOLD = 0.5
EARLY_STOP = 4
MODEL_SAVE_PATH = 'model_path/siamese.pth'

# ASTEncoder parameter
VOCAB_SIZE = 350
EMBEDDING_DIM = 16
HIDDEN_DIM = 64
OUTPUT_DIM = 768
NUM_LAYERS = 16
NUM_HEADS = 2
DROPOUT = 0.1
TRAVERSAL_METHOD = 'BFS'  # BFS: 广度优先遍历, DFS: 深度优先遍历

FASTTEXT_MODEL_PATH = 'model_bin/cc.zh.300.bin'
FASTTEXT_EMBEDDING_DIM = 256

# SiameseNetwork parameter
SIAMESE_EMBEDDING_DIM = OUTPUT_DIM  # embedding_dim + 特征向量长度

# 提取特征
'''
['ast', 'block', 'string', 'callee']
ast vector dim: 3
block vector dim: 6
string vector dim: FASTTEXT_EMBEDDING_DIM
callee vector dim: FASTTEXT_EMBEDDING_DIM
'''
FEATURES = []
ARCHIMEDES_SPIRAL_DIM = 2
INIT_BLOCK_FEATURES = {
        'IF_STMT': 0,
        'FOR_STMT': 0,
        'WHILE_STMT': 0,
        'SWITCH_STMT': 0,
        'DECL_STMT': 0,
        'COMPOUND_STMT': 0
}

