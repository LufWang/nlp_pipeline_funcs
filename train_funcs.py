"""
Text Classification training functions and pipelines 
based on pytorch & transformer libraries
Author: @wang.lufei@mayo.edu
"""

# Libraries
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup, BertForSequenceClassification
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict
from textwrap import wrap
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from tqdm.auto import tqdm
from sklearn.metrics import precision_score, recall_score, confusion_matrix, f1_score
from sklearn.utils import class_weight
import sys
import re
import os 
import pickle
from datetime import datetime
import json
import shortuuid
from nlp_pipeline_funcs.config import pretrained_path

def get_config(params):
    # random pick a set of params
    config = {}
    
    for name in params:
        choice = np.random.choice(params[name])
        config[name] = choice
        
    
    return config
        
    

class build_torch_dataset(Dataset):
    """
    Build torch dataset
    
    Input:
        texts: 
        labels
        tokenizer
        max_len
        
    output:
        dictionary: used to construct torch dataloader
    """
    
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]
        
        encoding = self.tokenizer.encode_plus(
                                          text,
                                          add_special_tokens=True,
                                          truncation = True,
                                          max_length=self.max_len,
                                          return_token_type_ids=False,
                                          padding='max_length',
                                          return_attention_mask=True,
                                          return_tensors='pt',
                                                )
        return {
                  'text': text,
                  'input_ids': encoding['input_ids'].flatten(),
                  'attention_mask': encoding['attention_mask'].flatten(),
                  'labels': torch.tensor(label, dtype=torch.int64)
                }


def create_data_loader(df, text_col, label_col, tokenizer, max_len, batch_size):
    """
    Construct data loader, on a torch dataset
    
    input:
        dataframe
        text column name
        label column name
        tokenizer
        max_len
        batch_size
        
    output:
        torch dataloader object
    """
    ds = build_torch_dataset(
                            texts=df[text_col].to_numpy(),
                            labels=df[label_col].to_numpy(),
                            tokenizer=tokenizer,
                            max_len=max_len
                              )
    return DataLoader(
                        ds,
                        batch_size=batch_size,
                        num_workers=0,
                        shuffle = False,
                        drop_last = False
                      )


def split_data_binary(df, 
                  label_col, 
                  RANDOM_SEED, 
                  stratify_col,
                  test_size = 0.1,
                  ratio = None):
    """
    pass in original dataframe, train, val, test split under/over sample
    
    Input:
        dataframe
        label column name
        random seed
        stratify col or None
        test size
        neg/pos class ratio (1 pos:neg 1:1, 2 pos:neg 1:2)
    
    output:
        df_train, df_test
    """

    stratify = df[stratify_col] if stratify_col else None
    
    
    df_train, df_test = train_test_split(
                                            df,
                                            test_size=test_size,
                                            random_state=RANDOM_SEED,
                                            stratify=stratify
                                            )
    
    if ratio: # if under sample train data
        df_train_pos = df_train[df_train[label_col] == 1]
        df_train_neg = df_train[df_train[label_col] == 0]
        down_ratio = df_train_pos.shape[0] / df_train_neg.shape[0]
        neg_size = down_ratio * ratio
        if neg_size > df_train_neg.shape[0]:
            print('Ratio picked greater than data available.. returning all neg data.')

        _, downsampled = train_test_split(df_train_neg, 
                                          test_size = down_ratio * ratio)
        
        df_train = pd.concat([df_train_pos, downsampled])
    
    
    # check data distribution
    print('-' * 20)
    print('train label distribution:')
    train_vc = df_train[label_col].value_counts()
    print('Pos: {}  Neg: {}'.format(train_vc.values[1], train_vc.values[0]))
    print('-' * 20)
    print('val/test label distribution:')
    test_vc = df_test[label_col].value_counts()
    print('Pos: {}  Neg: {}'.format(test_vc.values[1], test_vc.values[0]))
    print('-' * 20)
     
       
    return df_train.sample(frac=1, random_state = RANDOM_SEED), df_test


def split_data_multi(df, 
                     label_col, 
                     RANDOM_SEED, 
                     stratify_col,
                     test_size = 0.1,
                     sample = None):
    """
    pass in original dataframe - train, val, test split - adjust sample numbers by tree in training data
    
    Input:
        dataframe
        label column name
        random seed
        test size 
        sample: dictionary default None - 
     
    
    output:
        df_train, df_test
        
    """
    
    stratify = df[stratify_col] if stratify_col else None

    
    # train val test split - stratified on label
    df_train, df_val = train_test_split(df,
                                          test_size=test_size,
                                          random_state=RANDOM_SEED,
                                          stratify = stratify)

    
    
    if sample: # if choose number of samples by categories 
        df_list = []
        for label in sample:
            num_samples = sample[label]
            df_subset = df_train[df_train[label_col] == label]        
            if num_samples >df_subset.shape[0]:
                num_samples = df_subset.shape[0]
            
            df_list.append(df_subset.sample(num_samples))
       
        df_train = pd.concat(df_list)
      
    
    
    # data distribution check
    print('train: {}, val: {}'.format(df_train.shape[0], df_val.shape[0]))
    print('-' * 20)
    print('Lowest {} Counts'.format(label_col))
    train_min = df_train[label_col].value_counts().min()
    val_min = df_val[label_col].value_counts().min()
    print('train: {}. val: {}.'.format(train_min, val_min))
    
       
    return df_train.sample(frac=1, random_state = RANDOM_SEED), df_val

# Model Structure
class CustomBertBinaryClassifier(nn.Module):
    def __init__(self, pretrained_path, n_classes, device):
        super(CustomBertBinaryClassifier, self).__init__()
        self.device = device
        self.bert = BertModel.from_pretrained(pretrained_path)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes).to(self.device)
        
    def forward(self, input_ids, attention_mask):
        outputs  = self.bert(input_ids = input_ids, 
                                      attention_mask = attention_mask)
        outputs = self.drop(outputs[1]).to(self.device)
    
        outputs = self.out(outputs)
        
        return outputs
    
class CustomBertMultiClassifier(nn.Module):
    """
    Neural Network Structure
    
    """
    
    def __init__(self, pretrained_path, n_classes, device):
        super(CustomBertMultiClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_path)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes).to(device)
        self.n_classes = n_classes
        self.device = device
        
    def forward(self, input_ids, attention_mask):
        outputs  = self.bert(input_ids = input_ids, 
                                      attention_mask = attention_mask)
        outputs = self.drop(outputs[1]).to(self.device)
        
        
        outputs = self.out(outputs)
        
        return outputs


def get_loss_pred(outputs, labels, loss_fn, threshold, binary):
    """
    get loss and prediction from output of NN
    
    Args:
        outputs: output from model()
        binary: True or False
        loss_fn: loss function
        threshold: threshold to give a positive prediction (only for binary now)
    
    Returns:
        loss: pytorch loss
        preds: list of predicted labels
        
    """
    m = nn.Softmax(dim=1)
    
    if binary: # if doing binary classification
        outputs = outputs.squeeze()
        loss = loss_fn(outputs, labels)
        preds_proba = np.array(torch.sigmoid(outputs).tolist()) # add sigmoid since no sigmoid in NN
        preds = np.where(preds_proba > threshold, 1, 0)
        
        return loss, preds, preds_proba, [1-preds_proba, preds_proba]
            
    else: # if doing multiclass 
        m = nn.Softmax(dim=1)
        loss = loss_fn(outputs, labels.long())
        # _, preds = torch.max(outputs, dim=1)
        preds_proba, preds = torch.max(m(outputs), dim=1)
    
    
        return loss, preds, preds_proba, m(outputs)

def train_epoch(model, train_data_loader, loss_fn, optimizer, device, scheduler, threshold = 0.5, binary = True):
    
    """
    Function that train 1 epoch
    """
    
    print('training...')
    
    losses = []
    preds_l = []
    true_labels_l = []

    
    
    for d in tqdm(train_data_loader):
        
        model.train()
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        labels = d["labels"].to(device) 
        
        # getting output on current weights
        outputs = model(
                          input_ids=input_ids,
                          attention_mask=attention_mask
                     )
   
        
        # getting loss and preds for the current batch
        loss, preds, preds_proba, preds_probas_all = get_loss_pred(outputs, labels, loss_fn, threshold, binary)
        
        # backprogogate and update weights/biases
        losses.append(loss.item())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        
        preds_l.extend(preds.tolist())
        true_labels_l.extend(labels.tolist())
        
  
    # convert lists to arrays
    preds_l = np.array(preds_l)
    true_labels_l = np.array(true_labels_l)
    
        
    return preds_l, true_labels_l, losses


def eval_model(model, data_loader, loss_fn, device, threshold = 0.5, binary = True):

    model.eval()
    losses = []
    preds_l = []
    preds_probas_l = []
    true_labels_l = []
    
    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["labels"].to(device)
            # labels = torch.nn.functional.one_hot(labels.to(torch.int64)).to(device).to(float) # one hot label to the right shape

            outputs = model(
                              input_ids=input_ids,
                              attention_mask=attention_mask
                            )

            
            loss, preds, preds_probas, preds_probas_all = get_loss_pred(outputs, labels, loss_fn, threshold, binary)
            


            preds_l.extend(preds.tolist())
            true_labels_l.extend(labels.tolist())
            preds_probas_l.extend(preds_probas.tolist())
            
            losses.append(loss.item())
            
    preds_l = np.array(preds_l)
    true_labels_l = np.array(true_labels_l)
    preds_probas_l = np.array(preds_probas_l)
    
    
    
    return preds_l, preds_probas_l, true_labels_l, losses



def eval_model_detailed(model, data_loader, loss_fn, device, threshold = 0.5, binary = True):
    model.eval()
    print('generating detailed evaluation..')
    
    texts_l = []
    preds_l = []
    preds_probas_l = []
    true_labels_l = []
    preds_probas_all_l = []
    
    with torch.no_grad():
        for d in tqdm(data_loader):
            texts = d['text']
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["labels"].to(device)

            outputs = model(
                              input_ids=input_ids,
                              attention_mask=attention_mask
                            )
 
            loss, preds, preds_probas, preds_probas_all = get_loss_pred(outputs, labels, loss_fn, threshold, binary)
            
            texts_l.extend(texts)
            preds_l.extend(preds.tolist())
            preds_probas_l.extend(preds_probas.tolist())
            true_labels_l.extend(labels.tolist())
            
            if not binary:
                preds_probas_all_l.extend(preds_probas_all.cpu().tolist())
            
    
    return texts_l, preds_l, preds_probas_l, true_labels_l, preds_probas_all_l

def save_model(model, save_path, config, model_info, model_name,labels_to_indexes, indexes_to_labels):
                    
    print('Saving Model...')
    
    # generate ID
    model_id = shortuuid.ShortUUID().random(length=12)
                    
    if not os.path.isdir(save_path):
        os.mkdir(save_path) # create directory if not exist

    # change here 
    save_path_final = os.path.join(save_path, model_id + '|' + model_name)

    if not os.path.isdir(save_path_final):
        os.mkdir(save_path_final) # create directory for model if not exist

    torch.save(model.state_dict(), os.path.join(save_path_final, model_id + '|' + 'model.bin'))  # save model

    with open(os.path.join(save_path_final, model_id + '|' + 'labels_to_indexes.pickle'), 'wb') as handle:
        pickle.dump(labels_to_indexes, handle, protocol=pickle.HIGHEST_PROTOCOL) # save label to index map

    with open(os.path.join(save_path_final, model_id + '|' + 'indexes_to_labels.pickle'), 'wb') as handle:
        pickle.dump(indexes_to_labels, handle, protocol=pickle.HIGHEST_PROTOCOL) # save index to label map
                        
    with open(os.path.join(save_path_final, model_id + '|' + 'config.pickle'), 'wb') as f:
        pickle.dump(config, f, protocol=pickle.HIGHEST_PROTOCOL)
                    
    with open(os.path.join(save_path_final, model_id + '|' + 'model_info.json'), 'w', encoding='utf-8') as f:
        json.dump(model_info, f, ensure_ascii=False, indent=4)

    print('Model Files Saved.')
    print()
    
def save_model_v2(model, model_name, save_path, files):
    """
    model - model
    save_path - path
    files - dict of files with key to be names and values to be files to be saved in the same directory
    """
                    
    print('Saving Model...')
    
    # generate ID
    model_id = shortuuid.ShortUUID().random(length=12)
                    
    if not os.path.isdir(save_path):
        os.mkdir(save_path) # create directory if not exist

    # change here 
    save_path_final = os.path.join(save_path, model_id + '|' + model_name)

    if not os.path.isdir(save_path_final):
        os.mkdir(save_path_final) # create directory for model if not exist
    
    # save torch model
    torch.save(model.state_dict(), os.path.join(save_path_final, model_id + '|' + 'model.bin'))  # save model
    
    # save file in the files
    for file_name in files:
        if file_name == 'model_info.json':
            with open(os.path.join(save_path_final, model_id + '|' + file_name), 'w', encoding='utf-8') as f:
                json.dump(files[file_name], f, ensure_ascii=False, indent=4)
                
        else:
            with open(os.path.join(save_path_final, model_id + '|' + file_name), 'wb') as f:
                pickle.dump(files[file_name], f, protocol=pickle.HIGHEST_PROTOCOL)


        

    print('Model Files Saved.')
    print()

    
def train_binary(df_train, 
                 df_val, 
                 label_name, 
                 text_col, 
                 config, 
                 threshold,
                 best_val_f1_global,
                 device, 
                 eval_every,
                 early_stopping,
                 save_path):
    
    """
    Function that wraps training and evaluating together using bce loss
    # saving model based on val loss each eval step
    
    Input:
        df_train
        df_val
        label col name
        text col
        config dictionary
        decision threshold
        best val f1 global
        device
        seed
    

    """
    
    # getting config
    lr = config['lr']
    EPOCHS = config['epoch']
    boost = config['boost']
    MAX_LEN = config['MAX_LEN']
    BATCH_SIZE = config['BATCH_SIZE']
    weight_decay = config['weight_decay']
    warmup_steps = config['warmup_steps']
    
    # create tokenizer
    tokenizer = BertTokenizer.from_pretrained(pretrained_path)
    
    
    # getting ratio for pos_weight
    RATIO = df_train[df_train[label_name] == 0].shape[0] / df_train[df_train[label_name] == 1].shape[0]
    
    # create data loaders on datasets
    train_data_loader = create_data_loader(df_train, text_col, label_name, tokenizer, int(MAX_LEN), int(BATCH_SIZE))
    val_data_loader = create_data_loader(df_val, text_col, label_name,tokenizer, int(MAX_LEN), int(BATCH_SIZE))
    
    # initialize model
    model = CustomBertBinaryClassifier(pretrained_path, 1, device)
    model = model.to(device)


    ## training prereqs
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay = weight_decay) # optimizer to update weights
    total_steps = len(train_data_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
                                              optimizer,
                                              num_warmup_steps=warmup_steps,
                                              num_training_steps=total_steps
    )

    # loss function for binary classification
    loss_fn = nn.BCEWithLogitsLoss(
                                    pos_weight = torch.tensor(RATIO * boost)
                                    ).to(device)
    
    
    # get list eval steps 
    total_steps = len(train_data_loader) * EPOCHS
    print(f'Total Training Steps: {total_steps}')
    eval_steps = [x * eval_every for x in range(1, int(total_steps / eval_every) + 1)]
    if eval_steps[-1] != total_steps:
        eval_steps.append(total_steps)
    print('Evaluation Steps:', eval_steps)
    eval_ind = 0
    
    # initialize scores and name
    best_val_f1 = 0


    binary = True  # doing binary classificaiton
    
    global_step = 0
    best_val_f1= best_val_f1_global # initialize best val f1
    val_losses_list = [] # record val loss every eval step -> for early stopping
    train_losses_list = []
    running_train_loss = 0
    patience_count = 0
    val_f1_list = []
    
    
    # start training
    for epoch in range(EPOCHS):

        print(f'Epoch {epoch + 1}/{EPOCHS}')
        print('-' * 10)
        print('Training...')

        losses = []
        preds_l = []
        true_labels_l = []
        
        # training through the train_data_loader
        for d in tqdm(train_data_loader):
            
            global_step += 1

            model.train()
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["labels"].to(device) 

            # getting output on current weights
            outputs = model(
                              input_ids=input_ids,
                              attention_mask=attention_mask
                         )


            # getting loss and preds for the current batch
            loss, preds, preds_proba, preds_proba_all = get_loss_pred(outputs, labels, loss_fn, threshold, binary)

            # backprogogate and update weights/biases
            losses.append(loss.item())
            running_train_loss += loss.item() # update running train loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()


            preds_l.extend(preds.tolist())
            true_labels_l.extend(labels.tolist())
            
            # evaluating based on step
            if global_step == eval_steps[eval_ind]:
                print()
                eval_ind += 1
                
                
                val_preds, val_preds_probas, val_trues, val_losses = eval_model(
                                                                                model,
                                                                                val_data_loader,
                                                                                loss_fn,
                                                                                device,
                                                                                threshold,
                                                                                binary
                                                                                )
                val_f1 = f1_score(val_trues, val_preds)
                val_precision = precision_score(val_trues, val_preds, zero_division = 0)
                val_recall = recall_score(val_trues, val_preds, zero_division = 0)
                val_f1_list.append(val_f1)
                
                val_loss = np.mean(val_losses) # getting average val loss
                val_losses_list.append(val_loss)
                train_loss = running_train_loss / eval_every # getting average train loss
                train_losses_list.append(train_loss)
                
                running_train_loss = 0 # reset training loss               

                
                print(f'Train loss {train_loss} Val loss {val_loss} Val precision: {val_precision} recall: {val_recall}  f1: {val_f1}')

                
                # check if needed to be early stopped:
                if early_stopping:
                    if patience_count > early_stopping:
                        if val_f1_list[-1] > val_f1_list[-(early_stopping + 1)]:
                            print('Early Stopping..')
                            return val_f1_list

                    patience_count += 1
                
                
                # if new best validation f1 save model
                if val_f1 > best_val_f1:             
                    
                    if not os.path.isdir(save_path):
                        os.mkdir(save_path) # create directory for models if not existt
                        
                    files = {
                        'config.pickle': config ,
                        'model_info.json': {
                            'val_precision': round(val_precision, 3),
                            'val_recall': round (val_recall, 3),
                            'val_f1': round(val_f1, 3),
                            'val_loss': round(val_loss, 5),
                            'DTREE-Binary': label_name
                        }
                    }
                    
                    
                    save_model_v2(model, label_name, save_path, files)
                    tokenizer.save_pretrained(save_path)

                    # updating scores
                    best_val_f1 = val_f1

                    patience_count = 0 # reset early stopping patience count if a model saved
                
                
        
        
            
                
    return val_f1_list
    



def train_multi(df_train, 
                df_val, 
                label_col,
                text_col,
                config,
                best_val_f1_global,
                device, 
                labels_to_indexes,
                indexes_to_labels,
                focused_indexes = None, # label index to focus the performance on
                save_path = None,
                save_name_affix = '',
                pretrained_path = '/home/jupyter/gen4-dev/storage/gen4-models/pretrained-models'
               ):
    
    """
    Function that wraps training and evaluating together for multiclassification
    Evaluate at the end of every epoch
    
    Input:
        train: dataframe
        val: dataframe
        label_name: str -name of label column
        text column
        config
    
    Output:
        best validation f1
        best model name
    """
    
    # getting config
    lr = config['lr']
    EPOCHS = config['epoch']
    MAX_LEN = config['MAX_LEN']
    BATCH_SIZE = config['BATCH_SIZE']
    warmup_steps = config['warmup_steps']
    weight_decay = config['weight_decay']
    model_name = config['model_name']
    
    
    # initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained(os.path.join(pretrained_path, model_name))
    
    # create data loaders on datasets
    train_data_loader = create_data_loader(df_train, text_col, label_col, tokenizer, int(MAX_LEN), int(BATCH_SIZE))
    val_data_loader = create_data_loader(df_val, text_col, label_col, tokenizer, int(MAX_LEN), int(BATCH_SIZE))
    
    # initialize model
    model=CustomBertMultiClassifier(model_name, pretrained_path, len(indexes_to_labels), device)
    model = model.to(device)

    ## training params
    class_weight = []
    sample = df_train[label_col].value_counts().to_dict() # full composition

    for label in indexes_to_labels:
        class_weight.append(max(sample.values()) / sample[label])
    
    optimizer = AdamW(model.parameters(), lr=lr, correct_bias=False, weight_decay=weight_decay)
    total_steps = len(train_data_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
                                              optimizer,
                                              num_warmup_steps=warmup_steps,
                                              num_training_steps=total_steps
    )

    # boost of minority class weight
    loss_fn = nn.CrossEntropyLoss(
                                    weight = torch.tensor(class_weight).to(device)
                                    ).to(device)
    best_val_f1 = best_val_f1_global
    best_model_name = None
    binary = False
    threshold = 0
    
    for epoch in range(EPOCHS):
        print(f'Epoch {epoch + 1}/{EPOCHS}')
        print('-' * 10)
        

        preds, trues, train_loss = train_epoch(
                                                model,
                                                train_data_loader,
                                                loss_fn,
                                                optimizer,
                                                device,
                                                scheduler,
                                                threshold,
                                                binary
                                                )
        
        
        if focused_indexes:
            train_f1_all = f1_score(trues, preds, average = None)
            print(train_f1_all[focused_indexes])
            train_f1 = np.mean(train_f1_all[focused_indexes])
        else:
            train_f1 = f1_score(trues, preds, average = 'macro')

        print(f'F1 Score{train_f1}  Avg loss {np.mean(train_loss)}  ')
        print()
        preds, preds_probas, trues, val_loss = eval_model(
                                                model,
                                                val_data_loader,
                                                loss_fn,
                                                device,
                                                threshold,
                                                binary
                                                )
        
        
        if focused_indexes:
            val_f1_all = f1_score(trues, preds, average = None)
            for index in focused_indexes:
                print(f'{indexes_to_labels[index]}: F1 {val_f1_all[index]} ')
            val_f1 = np.mean(val_f1_all[focused_indexes])
            
        else:
            val_f1 = f1_score(trues, preds, average = 'macro')
        
        print('-'*30)
        print(f'F1 Socre{val_f1}  Avg loss {np.mean(val_loss)} ')
        print()
        
        if save_path:
            if val_f1 > best_val_f1: # if f1 score better. save model checkpoint
                
                model_info = {
                    'val_f1': val_f1,
                    'val_loss': round(np.mean(val_loss), 4),
                    'time_generated': datetime.now() 
                }
                
                save_model(model, save_path, config, model_info)  
                
                
                best_val_f1 = val_f1 # update best f1 score
        
         

    return best_val_f1, best_model_name




def train_multi_w_eval_steps(df_train, 
                            df_val, 
                            label_col,
                            text_col,
                            config,
                            best_val_f1_global,
                            device, 
                            save_model_name,
                            labels_to_indexes,
                            indexes_to_labels, 
                            eval_every = 500,
                            early_stopping = 10,
                            focused_indexes = None,
                            save_path = None,
                            save_name_affix = '',
                            pretrained_path = '/home/jupyter/gen4-dev/storage/gen4-models/pretrained-models'
                           ):
    
    """
    Function that streamline training, evaluating, and saving model for multiclassification
    Evaluate every [eval_every] steps
    
    Input:
        df_train: pd dataframe
        df_val: pd dataframe
        label_col: str - column name to train on, need to be in int
        text_col: str - column name for text input 
        config: dict
        best_val_f1_global: float - val_f1 threshold to save the model
        device: str - torch device: gpu/cpu
        save_model_name: str
        labels_to_indexes: dict - map labels to indexes
        indexes_to_labels: dict - map indexes to labels
        eval_every: int - eval step9
        early_stopping: int - how many evals to wait before terminate (if val metric does not improve)
        focused_indexes: list - indexes of labels to focuse on (if passed in will save based on these focused labels only)
        save_path: path (will create if not exist)
        save_name_affix: str - append anything passed in to the end of model name
        pretrained_path: path - path to the pretrained model to use 
    
    Output:
        best validation f1
        best model name
    """
    
    # getting config
    lr = config['lr']
    EPOCHS = config['epoch']
    MAX_LEN = config['MAX_LEN']
    BATCH_SIZE = config['BATCH_SIZE']
    warmup_steps = config['warmup_steps']
    weight_decay = config['weight_decay']
    model_name = config['model_name']
    boost = config['boost']
    
    # initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained(os.path.join(pretrained_path, model_name))
    
    # create data loaders on datasets
    train_data_loader = create_data_loader(df_train, text_col, label_col, tokenizer, int(MAX_LEN), int(BATCH_SIZE))
    val_data_loader = create_data_loader(df_val, text_col, label_col, tokenizer, int(MAX_LEN), int(BATCH_SIZE))
    
    # initialize model
    model=CustomBertMultiClassifier(model_name, pretrained_path, len(indexes_to_labels), device)
    model = model.to(device)
    
    # get list eval steps 
    total_steps = len(train_data_loader) * EPOCHS
    print(f'Total Training Steps: {total_steps}')
    eval_steps = [x * eval_every for x in range(1, int(total_steps / eval_every) + 1)]
    if eval_steps[-1] != total_steps:
        eval_steps.append(total_steps)
    print('Evaluation Steps:', eval_steps)
    

    ## assigning weights to each label class to account for imbalance
    class_weight = []
    sample = df_train[label_col].value_counts().to_dict()

    for label in indexes_to_labels:
        class_weight.append(max(sample.values()) / sample[label])
    if focused_indexes: # if focused index boost them
        for index in focused_indexes:
            class_weight[index] = class_weight[index] * float(boost)
 
    
    optimizer = AdamW(model.parameters(), lr=lr, correct_bias=False, weight_decay=weight_decay)
    total_steps = len(train_data_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
                                              optimizer,
                                              num_warmup_steps=warmup_steps,
                                              num_training_steps=total_steps
    )

    # boost of minority class weight
    loss_fn = nn.CrossEntropyLoss(
                                    weight = torch.tensor(class_weight).to(device)
                                    ).to(device)
    best_val_f1 = best_val_f1_global
    best_model_name = None
    binary = False
    threshold = 0
    
    global_step = 0
    eval_ind = 0
    val_losses_list = [] # record val loss every eval step -> for early stopping
    train_losses_list = []
    running_train_loss = 0
    patience_count = 0
    val_f1_list = []
    
    for epoch in range(EPOCHS):
        print()
        print()
        print(f'Epoch {epoch + 1}/{EPOCHS}')
        print('-' * 10)
        
        losses = []
        train_preds_l = []
        train_true_labels_l = []
        
        
        # training through the train_data_loader
        for d in tqdm(train_data_loader):
            
            global_step += 1

            model.train()
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["labels"].to(device) 

            # getting output on current weights
            outputs = model(
                              input_ids=input_ids,
                              attention_mask=attention_mask
                         )


            # getting loss and preds for the current batch
            loss, preds, preds_proba, preds_probas_all = get_loss_pred(outputs, labels, loss_fn, threshold, binary)

            # backprogogate and update weights/biases
            losses.append(loss.item())
            running_train_loss += loss.item() # update running train loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            # add preds and true labels to list
            train_preds_l.extend(preds.tolist())
            train_true_labels_l.extend(labels.tolist())
            
            # evaluating based on step
            if global_step == eval_steps[eval_ind]:
                print()
                eval_ind += 1
                
                print()
                print(f'Evaluateing at Step {global_step}....')
                preds, preds_probas, trues, val_losses = eval_model(
                                                                     model,
                                                                     val_data_loader,
                                                                     loss_fn,
                                                                     device,
                                                                     threshold,
                                                                     binary
                                                                                    )
                val_f1_by_tree = {}
                if focused_indexes:
                    val_f1_all = f1_score(trues, preds, average = None)
                    val_precision_all = precision_score(trues, preds, average = None, zero_division=0)
                    val_recall_all = recall_score(trues, preds, average = None)
                    
                    
                    for index in focused_indexes:
                        print('#'*30)
                        print(f'{indexes_to_labels[index]}: F1 {val_f1_all[index]} ')
                        print(f'{indexes_to_labels[index]}: Precsion {val_precision_all[index]} ')
                        print(f'{indexes_to_labels[index]}: Recall {val_recall_all[index]} ')
                        
                        val_f1_by_tree[indexes_to_labels[index]] = val_f1_all[index]
                        
                        
                    val_f1 = np.mean(val_f1_all[focused_indexes])

                else:
                    val_f1 = f1_score(trues, preds, average = 'macro')
                
                print('-'*30)
                print(f'F1 Socre: {val_f1}  Avg loss: {np.mean(val_losses)} ')
                print()
                
                if save_path:  # if a save path is provided, save model
            
                    if val_f1 > best_val_f1: # if f1 score better. save model checkpoint
                        if not os.path.isdir(save_path):
                            os.mkdir(save_path) # create directory if not exist
                        
                        
                        model_info = {
                                'val_f1': val_f1,
                                'val_loss': float(np.round(np.mean(val_losses), 4)),
                                'time_generated': datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
                                'focused_DTREES':  [indexes_to_labels[x]for x in focused_indexes] if focused_indexes  else [],
                                'val_f1_by_dtree': val_f1_by_tree
                            }
                
                        # save_model(model, save_path, config, model_info, save_model_name, labels_to_indexes, indexes_to_labels)  
                        files = {
                            'config.pickle': config,
                            'model_info.json': model_info,
                            'labels_to_indexes.pickle': labels_to_indexes,
                            'indexes_to_labels.pickle': indexes_to_labels
                            
                        }
                        
                        save_model_v2(model, save_model_name, save_path, files )
                        tokenizer.save_pretrained(save_path)

                        best_val_f1 = val_f1 # update best f1 score

                
                val_f1_list.append(val_f1)          
                val_loss = np.mean(val_losses) # getting average val loss
                val_losses_list.append(val_loss)
                
                train_loss = running_train_loss / eval_every # getting average train loss
                train_losses_list.append(train_loss)
                
                running_train_loss = 0 # reset training loss               
     
                # print(f'Running Train loss {train_loss} Val loss {val_loss}')

                
                # check if needed to be early stopped:
                if early_stopping:
                    if patience_count > early_stopping:
                        if val_f1_list[-1] > val_f1_list[-(early_stopping + 1)]:
                            print('Early Stopping..')
                            print('Val F1 List: ', val_f1_list)
                            return None, None
                
                    patience_count += 1
        
        
        # evaluate train after every epoch
        if focused_indexes:
            train_f1_all = f1_score(train_true_labels_l, train_preds_l, average = None)
            print(train_f1_all[focused_indexes])
            train_f1 = np.mean(train_f1_all[focused_indexes])
        else:
            train_f1 = f1_score(trues, preds, average = 'macro')
        
        print('-'*50)
        print(f'End of Epoch: Train F1 Score{train_f1}  Train Avg loss {np.mean(losses)}  ')
        print('-'*50)
        print()

         

    return best_val_f1, best_model_name





"""
Functions for loading models and predicting
"""


def predict_one(text, model, tokenizer, MAX_LEN, device):
    
    # tokenize and encode text
    encoded_review = tokenizer.encode_plus(
                                          text,
                                          add_special_tokens=True,
                                          truncation = True,
                                          max_length= MAX_LEN,
                                          return_token_type_ids=False,
                                          padding='max_length',
                                          return_attention_mask=True,
                                          return_tensors='pt',
                                                )
    
    input_ids = encoded_review['input_ids'].to(device)
    attention_mask = encoded_review['attention_mask'].to(device)
    output = model(input_ids, attention_mask)
    
    torch.cuda.empty_cache()
    
    return output



def predict_binary_cohort(df, text_col, label_col, model, tokenizer, MAX_LEN, device, batch_size):
    
    # text_col = 'cleaned_text'
    # label_col = 'GENERAL INTERNAL MEDICINE'
    
    ds = build_torch_dataset(
                            texts=df[text_col].to_numpy(),
                            labels=df[label_col].to_numpy(),
                            tokenizer=tokenizer,
                            max_len=MAX_LEN
                              )
    data_loader=  DataLoader(
                                ds,
                                batch_size=batch_size,
                                num_workers=0,
                                shuffle = False,
                                drop_last = False
                              )
    
    loss_fn = nn.BCEWithLogitsLoss().to(device)
    texts, preds, preds_proba, true_labels, preds_proba_all = eval_model_detailed(model, data_loader, loss_fn, device, threshold = 0.5, binary = True)
    
    
    return texts, preds, preds_proba



def predict_multi_cohort(df, text_col, label_col, model, tokenizer, MAX_LEN, device, batch_size):
    ds = build_torch_dataset(
                            texts=df[text_col].to_numpy(),
                            labels=df[label_col].to_numpy(),
                            tokenizer=tokenizer,
                            max_len=MAX_LEN
                              )
    data_loader=  DataLoader(
                                ds,
                                batch_size=batch_size,
                                num_workers=0,
                                shuffle = False,
                                drop_last = False
                              )
    
    loss_fn = nn.CrossEntropyLoss().to(device)
    texts, preds, preds_proba, true_labels, preds_probas_all = eval_model_detailed(model, data_loader, loss_fn, device, threshold = 0.5, binary = False)
    
    
    
    return texts, preds, preds_proba, true_labels, preds_probas_all



