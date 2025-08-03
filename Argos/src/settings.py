import os

from src.utils import device_allocation

# Server Settings
NUMBER_OF_CLIENTS = 2

# Clients' Settings
CLIENT_BATCH_SIZE = 32
CLIENT_LEARNING_RATE = 0.001
DEVICE = device_allocation()
TRAIN_PERCENTAGE = 0.8
VAL_PERCENTAGE = 0.1
TEST_PERCENTAGE = 0.1


# Dataset
DATASET_PATH = "./data"
CLASSES_JSON_FILE = os.path.join(DATASET_PATH, "classes.json")
CLIENT_PARTITION_DIR = os.path.join(DATASET_PATH, "partitions")


