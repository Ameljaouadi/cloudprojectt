

# -------------------------------------------------------------------------
#                               Configurations
# -------------------------------------------------------------------------
MODEL_LOC = './pneumonia_detection_cnn_model.h5'
DATA_DIR = '../data/chest_xray/'
TRAINING_DATA_DIR = DATA_DIR + '/train/'
TEST_DATA_DIR = DATA_DIR + '/test/'
VAL_DATA_DIR = DATA_DIR + '/val/'
DETECTION_CLASSES = ('NORMAL', 'PNEUMONIA')
BATCH_SIZE = 32
EPOCHS = 10


