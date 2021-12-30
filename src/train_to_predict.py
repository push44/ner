#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import joblib
import torch
import transformers

import pandas as pd
import numpy as np
import torch.nn as nn

from tqdm.notebook import tqdm

from sklearn import preprocessing
from sklearn import model_selection
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup


# ### 1. config.py

# In[2]:


#################### Input/ output folders ####################
INPUT_FOLDER = "../input/feedback-prize-2021"
TRAIN_ESSAY_FOLDER = f"{INPUT_FOLDER}/train"
TEST_ESSAY_FOLDER = f"{INPUT_FOLDER}/test"

TRAIN_ANNOT_FILE = f"{INPUT_FOLDER}/train.csv"
SUBMISSION_FILE = f"{INPUT_FOLDER}/sample_submission.csv"

MODEL_FILE = "/kaggle/working/model.bin"
META_DATA_FILE = "/kaggle/working/meta.bin"

#TRAIN_IOB_FILE = f"{INPUT_FOLDER}/train_iob.csv"
#META_DATA_FILE = f"../models/meta.bin"

TRAIN_CHUNK_SIZE = 50
TEST_CHUNK_SIZE = 50

TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 8
EPOCHS = 2

MAX_LEN = 256
MAX_WAITING = 3

BERT_PATH = "../input/bert-base-uncased/"

TOKENIZER = transformers.BertTokenizer.from_pretrained(
    BERT_PATH,
    do_lower=True
)


# ### 2. create_iob_tag.py

# In[3]:


#return file text in a tokenized list using python .split()
def return_text_list(file_id, train_essay_folder):
    with open(f"{train_essay_folder}/{file_id}.txt", "r") as f:
        text_list = f.read().split()
    return text_list

def create_iob_df(train_annot_df, train_essay_folder):
    #convert predictionstring to list
    train_annot_df["predictionlist"] = train_annot_df["predictionstring"].apply(lambda string: list(map(int, string.split())))
    #get unique file ids
    file_ids = train_annot_df["id"].unique().tolist()[:100]

    #create iob tag dataframe (one row for one word of a file)
    iob_dfs = []
    for file_id in tqdm(file_ids):
        #get dataframe for file_id
        sample_df = train_annot_df[train_annot_df["id"]==file_id]
        #create iob_df for file_id
        sample_iob_df = pd.DataFrame({
            "id":file_id,
            "word":return_text_list(file_id, train_essay_folder),
            "tag":"O"
        })
        #for every row in train_annot_df (multiple rows for one file_id)
        for i in range(sample_df.shape[0]):
            #get info for i'th row of a file_id
            prediction_list = sample_df["predictionlist"].iloc[i]
            discourse_type = sample_df["discourse_type"].iloc[i].lower()
            
            #starting of discourse tag is marked with B (begining)
            sample_iob_df["tag"].iloc[prediction_list[0]] = f"B-{discourse_type}"
            if len(prediction_list)>1:
                #rest of discourse tag is marked with I (inside)
                sample_iob_df["tag"].iloc[prediction_list[1]:prediction_list[-1]+1] = f"I-{discourse_type}"
        
        #append all sample_iob_df representing each file_id
        iob_dfs.append(sample_iob_df)
        
    #return single iob_df
    return pd.concat(iob_dfs).reset_index(drop=True)


# ### 3. dataset.py

# In[4]:


class Dataset:
    def __init__(self, text_corpus, tag_corpus, tokenizer):
        self.text_corpus = text_corpus
        self.tag_corpus = tag_corpus
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.text_corpus)

    def __getitem__(self, item):
        text = self.text_corpus[item]
        tags = self.tag_corpus[item]

        input_ids = []
        target_tags = []

        for ind, word in enumerate(text):
            inputs = self.tokenizer.encode(
                  word,
                  add_special_tokens = False
              )

            input_len = len(inputs)

            input_ids.extend(inputs)
            target_tags.extend([tags[ind]]*input_len)

        input_ids = input_ids[:MAX_LEN - 2]
        target_tags = target_tags[:MAX_LEN - 2]

        input_ids = [101] + input_ids + [102]
        target_tags = [0] + target_tags + [0]

        token_type_ids = [0]*len(input_ids)
        attention_mask = [1]*len(input_ids)

        padding_len = MAX_LEN - len(input_ids)

        input_ids = input_ids + ([0] * padding_len)
        token_type_ids = token_type_ids + ([0] * padding_len)
        attention_mask = attention_mask + ([0] * padding_len)
        target_tags = target_tags + ([0] * padding_len)

        #print(TOKENIZER.convert_ids_to_tokens(input_ids))

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "target": torch.tensor(target_tags, dtype=torch.long)
        }


# ### 4. model.py

# In[5]:


def loss_fn(output, target, mask, num_labels):
    lfn = nn.CrossEntropyLoss()
    active_loss = mask.view(-1) == 1
    active_logits = output.view(-1, num_labels)
    active_labels = torch.where(
        active_loss,
        target.view(-1),
        torch.tensor(lfn.ignore_index).type_as(target)
    )
    loss = lfn(active_logits, active_labels)
    return loss

class Model(nn.Module):
    def __init__(self, num_tag, bert_path):
        super(Model, self).__init__()
        self.num_tag = num_tag
        self.bert = transformers.BertModel.from_pretrained(
            bert_path,
            return_dict = False
        )
        self.drop = nn.Dropout(0.3)
        self.linear = nn.Linear(768, self.num_tag)

    def forward(self, input_ids, token_type_ids, attention_mask, target):
        last_hidden_state, pooler_output = self.bert(
            input_ids = input_ids,
            token_type_ids = token_type_ids,
            attention_mask = attention_mask
        )

        last_hidden_state = self.drop(last_hidden_state)
        linear_output = self.linear(last_hidden_state)

        loss = loss_fn(linear_output, target, attention_mask, self.num_tag)

        return linear_output, loss


# ### 5. engine.py

# In[6]:


#################### TRAIN ####################
def train_one_step(model, data, optimizer, device):
    optimizer.zero_grad()
    
    for k, v in data.items():
        data[k] = v.to(device)
        
    output, loss = model(**data)
    output = output.cpu().detach().numpy().tolist()

    loss.backward()
    optimizer.step()
    
    return output, float(loss)

def train_one_epoch(model, data_loader, optimizer, device, scheduler):
    model.train()
    
    total_loss = 0
    for data in tqdm(data_loader, total=len(data_loader)):
        output, loss = train_one_step(model, data, optimizer, device)
        total_loss += loss
        
    scheduler.step()
        
    return total_loss

#################### VALIDATE ####################
def validate_one_step(model, data, device):
    
    for k, v in data.items():
        data[k] = v.to(device)
        
    output, loss = model(**data)
    output = output.cpu().detach().numpy().tolist()

    return output, float(loss)

def validate_one_epoch(model, data_loader, device):
    model.eval()
    
    total_loss = 0
    for data in tqdm(data_loader, total=len(data_loader)):
        with torch.no_grad():
            output, loss = validate_one_step(model, data, device)
        total_loss += loss
        
    return total_loss/len(data_loader)


# ### 6. train.py

# In[7]:


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

def get_scheduler(num_samples, optimizer, epochs, train_batch_size):

    num_train_steps = int(num_samples / epochs * train_batch_size)

    return get_linear_schedule_with_warmup(
        optimizer = optimizer,
        num_warmup_steps = 0,
        num_training_steps = num_train_steps
    )

def train(df, meta_data_file, train_chunk_size, tokenizer, train_batch_size, valid_batch_size, bert_path, epochs, model_file, max_waiting):

    ##################### STAGE 1 #####################
    #iob tag label encoding
    label_enc = preprocessing.LabelEncoder()
    df["label"] = label_enc.fit_transform(df["tag"].values)
    meta_data = {"label_enc": label_enc}

    #save label encoder object for test prediction
    joblib.dump(meta_data, meta_data_file)

    ##################### STAGE 2 #####################

    sentences = []
    labels = []
    for start_ind in range(0,df.shape[0], train_chunk_size):
        sentences.append(list(df.iloc[start_ind:start_ind+50]["word"].values))
        labels.append(list(df.iloc[start_ind:start_ind+50]["label"].values))

    ##################### STAGE 3 #####################

    train_sentences, valid_sentences, train_labels, valid_labels = model_selection.train_test_split(sentences, labels)

    ##################### STAGE 4 #####################

    train_dataset = Dataset(train_sentences, train_labels, tokenizer)
    valid_dataset = Dataset(valid_sentences, valid_labels, tokenizer)

    train_data_loader = torch.utils.data.DataLoader(
        dataset = train_dataset,
        batch_size = train_batch_size
    )

    valid_data_loader = torch.utils.data.DataLoader(
        dataset = valid_dataset,
        batch_size = valid_batch_size
    )

    ##################### STAGE 5 #####################
    num_tags = len(label_enc.classes_)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Model(num_tags, bert_path).to(device)
    optimizer = get_optimizer(model)
    scheduler = get_scheduler(len(train_dataset), optimizer, epochs, train_batch_size)

    ##################### STAGE 6 #####################
    best_loss = np.inf
    waiting = 0
    
    training_loss = []
    validation_loss = []
    for epoch in range(epochs):
        print(f"Epoch# {epoch+1}...")

        train_loss = train_one_epoch(model, train_data_loader, optimizer, device, scheduler)
        training_loss.append(train_loss)

        valid_loss = validate_one_epoch(model, valid_data_loader, device)
        validation_loss.append(valid_loss)
        print(f"Validation loss: {round(valid_loss, 5)}")

        if valid_loss<best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), model_file)
            waiting=0

        else:
            waiting+=1
            if waiting==max_waiting:
                print("======"*20)
                print(f"Best loss: {round(best_loss, 5)}")
                print("======"*20)
                break


# #### run training

# In[8]:


iob_df = create_iob_df(
    train_annot_df = pd.read_csv(
        TRAIN_ANNOT_FILE
    ),
    train_essay_folder = TRAIN_ESSAY_FOLDER
)
                
train(
    df=iob_df,
    meta_data_file=META_DATA_FILE,
    train_chunk_size=TRAIN_CHUNK_SIZE,
    tokenizer=TOKENIZER,
    train_batch_size=TRAIN_BATCH_SIZE,
    valid_batch_size=VALID_BATCH_SIZE,
    bert_path=BERT_PATH,
    epochs=EPOCHS,
    model_file=MODEL_FILE,
    max_waiting=MAX_WAITING
)


# ### 7. predict.py

# In[9]:


def first_capital(string):
    string = string.lower()
    string_list = []
    for ind, val in enumerate(string):
        if ind==0:
            string_list.append(val.upper())
        else:
            string_list.append(val)
    return "".join(string_list)

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield " ".join(lst[i:i + n])

        
def convert_submission_format_df(df, file_id):
    df = df[df["label"]!="O"]
    df_gr = df.groupby("label")["ind"].apply(list).reset_index()

    label_ind_dict = {}
    for ind in range(df_gr.shape[0]):
        label = df_gr.iloc[ind]["label"]
        ind_list = df_gr.iloc[ind]["ind"]
        label_ind_dict[label] = sorted(ind_list)

    pred_dfs = []
    label_wise_predictionstring = {}
    for label in label_ind_dict.keys():
        prediction_indices = label_ind_dict[label]
        predlist = []
        for ind, val in enumerate(prediction_indices):
            if ind==0:
                pred = [str(val)]
            else:
                if val - int(pred[-1])==1:
                    pred.append(str(val))
                else:
                    predlist.append(" ".join(pred))
                    pred = [str(val)]
                    
            if ind==len(prediction_indices)-1:
                predlist.append(" ".join(pred))
        label_wise_predictionstring[label] = predlist
        pred_df = pd.DataFrame({"predictionstring": label_wise_predictionstring[label]})
        pred_df["class"] = label
        pred_dfs.append(pred_df)
    pred_df = pd.concat(pred_dfs)
    pred_df["id"] = file_id
    return pred_df

def predict(test_sentence, label_enc, model, device, tokenizer):

    tokenized_sentence = tokenizer.encode(test_sentence)

    test_sentence = test_sentence.split()

    test_dataset = Dataset(
        text_corpus = [test_sentence],
        tag_corpus = [[0] * len(test_sentence)],
        tokenizer = tokenizer
    )

    with torch.no_grad():
        data = test_dataset[0]
        for k, v in data.items():
            data[k] = v.to(device).unsqueeze(0)
        predictions, _ = model(**data)
        predictions = predictions.cpu().detach().numpy()

    label_indices = label_enc.inverse_transform(
        predictions.argmax(2).reshape(-1)
    )

    tokens = tokenizer.convert_ids_to_tokens(test_dataset[0]["input_ids"].to("cpu").numpy())

    new_tokens, new_labels = [], []
    for token, label_idx in zip(tokens, label_indices):
        if token.startswith("##"):
            new_tokens[-1] = new_tokens[-1] + token[2:]
        else:
            new_labels.append(label_idx)
            new_tokens.append(token)

    return (new_tokens, new_labels)


def main(meta_data_file, bert_path, model_file, test_essay_folder, test_chunk_size, tokenizer):
    meta_data = joblib.load(meta_data_file)
    label_encoder = meta_data["label_enc"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_tags = len(label_encoder.classes_)
    model = Model(num_tags, bert_path).to(device)
    model.load_state_dict(torch.load(model_file, map_location=torch.device(device)))

    output_dfs = []
    for test_file_name in tqdm(os.listdir(test_essay_folder)):
        with open(f"{test_essay_folder}/{test_file_name}", "r") as file:
            sentence_tokens = file.read().split()

        tokens, labels = [], []
        for sentence_batch in chunks(sentence_tokens, test_chunk_size):
            batch_tokens, batch_labels = predict(
                test_sentence=sentence_batch,
                label_enc=label_encoder,
                model=model,
                device=device,
                tokenizer=tokenizer
            )
            for token, label in zip(batch_tokens, batch_labels):
                if not token in ["[PAD]", "[CLS]", "[SEP]"]:
                    tokens.append(token)
                    labels.append(label)

        labels = list(map(lambda val: first_capital(val.split("-")[1]) if val!="O" else val , labels))
        output_df = pd.DataFrame({"label":labels, "ind":range(len(labels))})
        output_df = convert_submission_format_df(
            df = output_df,
            file_id = test_file_name.split(".")[0]
        )
        output_dfs.append(output_df)
    return pd.concat(output_dfs)


# In[10]:


final_output_df = main(
    META_DATA_FILE,
    BERT_PATH,
    MODEL_FILE,
    TEST_ESSAY_FOLDER,
    TEST_CHUNK_SIZE,
    TOKENIZER
)

final_output_df.to_csv("submission.csv", index=False)

