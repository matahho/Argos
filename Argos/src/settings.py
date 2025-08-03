import os

# Server Settings
NUMBER_OF_CLIENTS = 2

# Clients' Settings
CLIENT_BATCH_SIZE = 1
CLIENT_LEARNING_RATE = 0.001
TRAIN_PERCENTAGE = 0.8
TEST_PERCENTAGE = 0.2


# Dataset
DATASET_PATH = "./data"
CLASSES_JSON_FILE = os.path.join(DATASET_PATH, "classes.json")
CLIENT_PARTITION_DIR = os.path.join(DATASET_PATH, "partitions")
DATASET_SIZE = 100