import config
import torch
import joblib

import pandas as pd
import numpy as np

from dataset import Dataset
from model import Model
from engine import *

from sklearn import preprocessing
from sklearn import model_selection
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

def get_optimizer(model):

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    optimizer_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.001,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

    return AdamW(optimizer_parameters, lr=3e-5)

def get_scheduler(num_samples, optimizer):

    num_train_steps = int(num_samples / config.EPOCHS * config.TRAIN_BATCH_SIZE)

    return get_linear_schedule_with_warmup(
        optimizer = optimizer,
        num_warmup_steps = 0,
        num_training_steps = num_train_steps
    ) 

def train(df):

    ##################### STAGE 1 #####################
    #iob tag label encoding
    label_enc = preprocessing.LabelEncoder()
    df["label"] = label_enc.fit_transform(df["tag"].values)
    meta_data = {"label_enc": label_enc}

    #save label encoder object for test prediction
    joblib.dump(meta_data, config.META_DATA_FILE)

    ##################### STAGE 2 #####################

    sentences = []
    labels = []
    for start_ind in range(0,df.shape[0], config.TRAIN_CHUNK_SIZE):
        sentences.append(list(df.iloc[start_ind:start_ind+50]["word"].values))
        labels.append(list(df.iloc[start_ind:start_ind+50]["label"].values))

    ##################### STAGE 3 #####################

    train_sentences, valid_sentences, train_labels, valid_labels = model_selection.train_test_split(sentences, labels)

    ##################### STAGE 4 #####################

    train_dataset = Dataset(train_sentences, train_labels)
    valid_dataset = Dataset(valid_sentences, valid_labels)

    train_data_loader = torch.utils.data.DataLoader(
        dataset = train_dataset,
        batch_size = config.TRAIN_BATCH_SIZE
    )

    valid_data_loader = torch.utils.data.DataLoader(
        dataset = valid_dataset,
        batch_size = config.VALID_BATCH_SIZE
    )

    ##################### STAGE 5 #####################
    num_tags = len(label_enc.classes_)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    entity_model = Model(num_tags).to(device)
    optimizer = get_optimizer(entity_model)
    scheduler = get_scheduler(len(train_dataset), optimizer)

    ##################### STAGE 6 #####################
    best_loss = np.inf
    waiting = 0
    
    training_loss = []
    validation_loss = []
    for epoch in range(config.EPOCHS):
        print(f"Epoch# {epoch+1}...")

        train_loss = train_one_epoch(entity_model, train_data_loader, optimizer, device, scheduler)
        training_loss.append(train_loss)

        valid_loss = validate_one_epoch(entity_model, valid_data_loader, device)
        validation_loss.append(valid_loss)
        print(f"Validation loss: {round(valid_loss, 5)}")

        if valid_loss<best_loss:
            best_loss = valid_loss
            torch.save(entity_model.state_dict(), config.MODEL_FILE)
            waiting=0

        else:
            waiting+=1
            if waiting==config.MAX_WAITING:
                print("======"*20)
                print(f"Best loss: {round(best_loss, 5)}")
                print("======"*20)
                break


if __name__ == "__main__":
    train(
        df = pd.read_csv(config.TRAIN_IOB_FILE, na_filter=False)
    )