import os

# ############### ROOT DIR ###############
# paths_config.py is in: <root>/config/paths_config.py
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACT_DIR = os.path.join(ROOT_DIR, "artifacts")

########################### DATA INGESTION #########################

RAW_DIR = os.path.join(ARTIFACT_DIR, "raw")
RAW_FILE_PATH = os.path.join(RAW_DIR, "raw.csv")
TRAIN_FILE_PATH = os.path.join(RAW_DIR, "train.csv")
TEST_FILE_PATH = os.path.join(RAW_DIR, "test.csv")

CONFIG_PATH = os.path.join(ROOT_DIR, "config", "config.yaml")

######################## DATA PROCESSING ########################

PROCESSED_DIR = os.path.join(ARTIFACT_DIR, "processed")
PROCESSED_TRAIN_DATA_PATH = os.path.join(PROCESSED_DIR, "processed_train.csv")
PROCESSED_TEST_DATA_PATH = os.path.join(PROCESSED_DIR, "processed_test.csv")

####################### MODEL TRAINING #################

MODEL_OUTPUT_PATH = os.path.join(ARTIFACT_DIR, "models", "lgbm_model.pkl")