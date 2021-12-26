################################################################
##################### Input folders/ files #####################
################################################################
INPUT_FOLDER = "/media/push44/PUSH/feedback-prize-competition/feedback-prize-2021"
TRAIN_ESSAY_FOLDER = f"{INPUT_FOLDER}/train"
TEST_ESSAY_FOLDER = f"{INPUT_FOLDER}/test"

TRAIN_ANNOT_FILE = f"{INPUT_FOLDER}/train.csv"
SUBMISSION_FILE = f"{INPUT_FOLDER}/sample_submission.csv"

TRAIN_IOB_FILE = f"{INPUT_FOLDER}/train_iob.csv"
META_DATA_FILE = f"../models/meta.bin"

from transformers import BertTokenizer

TRAIN_CHUNK_SIZE = 50
TEST_CHUNK_SIZE = 50

TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 8
EPOCHS = 3

MAX_LEN = 256
MAX_WAITING = 3

MODEL_FILE = "../models/model.bin"

TOKENIZER = BertTokenizer.from_pretrained(
    "bert-base-uncased",
    do_lower=True
)