from Argos.utils import device_allocation
import os

# Server Settings
NUMBER_OF_CLIENTS = 2

# Clients' Settings
CLIENT_BATCH_SIZE = 32
CLIENT_LEARNING_RATE = 0.001
DEVICE = device_allocation()


# Dataset
DATASET_PATH = "../data/"
CLASSES_JSON_FILE = os.path.join(DATASET_PATH, "classes.json")


