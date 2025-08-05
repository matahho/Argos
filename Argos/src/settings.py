import os

# Server Settings
NUMBER_OF_CLIENTS = 10
CHECKPOINT_PATH = "/kaggle/working/checkpoints"

# Clients' Settings
CLIENT_BATCH_SIZE = 2
CLIENT_LEARNING_RATE = 0.001
TRAIN_PERCENTAGE = 0.9
TEST_PERCENTAGE = 0.1


# Dataset
DATASET_PATH = "/kaggle/input/traffic-signs-dataset-mapillary-and-dfg"
CLASSES_JSON_FILE = os.path.join(DATASET_PATH, "classes.json")
DATASET_SIZE = 19_000