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
from tabulate import tabulate
import shutil




def get_config(params):
    # random pick a set of params
    config = {}
    
    for name in params:
        choice = np.random.choice(params[name])

        if type(choice) == np.int64:
            choice = int(choice)
        elif type(choice) == np.float64:
            choice= float(choice)
        elif type(choice) == np.str_:
            choice = str(choice)

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
                  'labels': torch.tensor(label, dtype=torch.float)
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

    

def save_model_v2(model, tokenizer, model_name, save_path, files):
    """
    model - model
    tokenizer - tokenizer intialized using transformer
    model_name - str: name of the saved directory
    save_path - path
    files - dict of files with key to be names and values to be files to be saved in the same directory, have to be all json files
    """
                    
    print('Saving Model...')
    
    # generate ID
    model_id = shortuuid.ShortUUID().random(length=12)
                    
    if not os.path.isdir(save_path):
        os.mkdir(save_path) # create directory if not exist

    # change here 
    save_path_final = os.path.join(save_path, model_id + '-' + model_name)

    if not os.path.isdir(save_path_final):
        os.mkdir(save_path_final) # create directory for model if not exist
    
    # save torch model
    torch.save(model.state_dict(), os.path.join(save_path_final, model_id + '-' + 'model.bin'))  # save model

    # save tokenizer
    tokenizer.save_pretrained(save_path_final)
    
    # save file in the files
    for file_name in files:
        with open(os.path.join(save_path_final, model_id + '-' + file_name), 'w', encoding='utf-8') as f:
            json.dump(files[file_name], f, ensure_ascii=False, indent=4)


    print('Model Files Saved.')
    print()


def evaluate_by_metrics(y_true, y_pred, metrics_list, average = 'binary', verbose = True):

    """
    Helper function that prints out and save a list of metrics defined by the user
    """

    results = {}
    output_str = ''

    for metric_name in metrics_list:
        metric_func = metrics_list[metric_name]

        if metric_name == 'confusion_matrix':
            score = metric_func(y_true, y_pred)
            score = score.tolist()
            output_str += f'| {metric_name}: {score} '

            results[metric_name] = score

        else:
            score = metric_func(y_true, y_pred, average=average, zero_division=0)
            results[metric_name] = score
            
            if type(score) == float:
                score = round(score, 4)

            output_str += f'| {metric_name}: {score} '


    
    if verbose:
        print(output_str)

    return results

def train_binary(model,
                df_train, 
                df_val, 
                tokenizer,
                label_col,
                text_col,
                config,
                device, 
                threshold,
                eval_func = f1_score

               ):
    
    """
    Function that wraps training and evaluating together for multiclassification
    mainly used for cross validation
    does not checkpoint model

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
    # specialty = config['specialty']
    RANDOM_SEED = config['RANDOM_SEED']


    # initialize tokenizer
    # tokenizer = BertTokenizer.from_pretrained(os.path.join(pretrained_path, model_name))

    # create data loaders on datasets
    train_data_loader = create_data_loader(df_train, text_col, label_col, tokenizer, int(MAX_LEN), int(BATCH_SIZE))
    val_data_loader = create_data_loader(df_val, text_col, label_col, tokenizer, int(MAX_LEN), int(BATCH_SIZE))

    ## training prereqs
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay = weight_decay) # optimizer to update weights
    total_steps = len(train_data_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
                                              optimizer,
                                              num_warmup_steps=warmup_steps,
                                              num_training_steps=total_steps
    )

    # loss function for binary classification
    # getting ratio for pos_weight
    RATIO = df_train[df_train[label_col] == 0].shape[0] / df_train[df_train[label_col] == 1].shape[0]
    loss_fn = nn.BCEWithLogitsLoss(
                                    pos_weight = torch.tensor(RATIO)
                                    ).to(device)

    binary = True

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
        
    
    # after training evaluate
    train_score = eval_func(trues, preds, zero_division=0)

    print(f'Train Score: {round(train_score, 4)}  Avg loss: {round(np.mean(train_loss), 4)}  ')

    
    preds, preds_probas, trues, val_loss = eval_model(
                                            model,
                                            val_data_loader,
                                            loss_fn,
                                            device,
                                            threshold,
                                            binary
                                            )
    
    val_score = eval_func(trues, preds, zero_division=0)
    

    print(f'Val Socre{round(val_score, 4)}  Avg loss {round(np.mean(val_loss), 4)} ')
    print()
    

        
            

    return val_score


def train_binary_w_eval_steps(
                                model,
                                df_train, 
                                df_val, 
                                label_name, 
                                text_col, 
                                config, 
                                threshold,
                                best_val_score_global,
                                device, 
                                eval_freq, # evaluation frequency per epoch
                                early_stopping,
                                save_path,
                                save_metric = f1_score,
                                metrics_list = {
                                    "f1": f1_score,
                                    "precision": precision_score,
                                    "recall": recall_score,
                                    "confusion_matrix": confusion_matrix
                                }):
    
    """
    Function that wraps training and evaluating together using bce loss
    # saving model based on val loss each eval step
    
    Input:
        model
        df_train
        df_val
        label col name
        text col
        config dictionary
        decision threshold
        best val score global
        device
        evaluate frequency
        early stopping
        save_path
        save metric
        metrics_list
    

    """
    
    # getting config
    lr = config['lr']
    EPOCHS = config['epoch']
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
    
    ## training prereqs
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay = weight_decay) # optimizer to update weights
    total_steps = len(train_data_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
                                              optimizer,
                                              num_warmup_steps=warmup_steps,
                                              num_training_steps=total_steps
    )

    # loss function for binary classification
    loss_fn = nn.BCEWithLogitsLoss(
                                    pos_weight = torch.tensor(RATIO )
                                    ).to(device)
    
    
    # get list eval steps 
    epoch_steps = len(train_data_loader)
    total_steps = epoch_steps * EPOCHS
    total_evaluations = eval_freq * EPOCHS
    print(f'Total Training Steps: {total_steps}')
    eval_steps = [int(total_steps/total_evaluations) * i for i in range(1, total_evaluations)]
    eval_steps.append(total_steps)
    print('Evaluation Steps:', eval_steps)
    eval_ind = 0
    
    binary = True  # doing binary classificaiton
    
    global_step = 1
    best_val_score= best_val_score_global # initialize best score ## needs to be the score returned by function in save_metric
    val_losses_list = [] # record val loss every eval step -> for early stopping
    running_train_loss = 0
    patience_count = 0
    val_scores_list = []
    
    
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
                
                val_preds, val_preds_probas, val_trues, val_losses = eval_model(
                                                                                model,
                                                                                val_data_loader,
                                                                                loss_fn,
                                                                                device,
                                                                                threshold,
                                                                                binary
                                                                                )
                
                val_loss = np.mean(val_losses) # getting average val loss
                val_losses_list.append(val_loss)

                # print out scores
                print(f'Evaluation at Step {global_step}....')
                print(f'Val loss {round(val_loss, 3)}')
                eval_results = evaluate_by_metrics(val_trues, val_preds, metrics_list)
                val_score = save_metric(val_trues, val_preds, zero_division=0)

                val_scores_list.append(val_score)
                print()
                
                eval_ind += 1
                
                # check if needed to be early stopped:
                if early_stopping:
                    if patience_count > early_stopping:
                        if val_scores_list[-1] > val_scores_list[-(early_stopping + 1)]:
                            print('Early Stopping..')
                            return val_scores_list

                    patience_count += 1
                

                # if new best validation f1 save model
                if val_score > best_val_score:             
                    
                    if not os.path.isdir(save_path):
                        os.mkdir(save_path) # create directory for models if not existt

                    eval_results['val_loss'] = round(val_loss, 5)
                    eval_results['label_name'] = label_name

                        
                    files = {
                        'config.json': config ,
                        'model_info.json': eval_results
                    }
                    
                    
                    save_model_v2(model, tokenizer, label_name, save_path, files)

                    # updating scores
                    best_val_score = val_score
                    patience_count = 0 # reset early stopping patience count if a model saved

            global_step += 1    
                
        
        
            
                
    return val_scores_list, best_val_score
    


def train_multi(model,
                df_train, 
                df_val, 
                tokenizer,
                label_col,
                text_col,
                config,
                device, 
                indexes_to_labels,
                focused_indexes = None,
                eval_func = f1_score

               ):
    
    """
    Function that wraps training and evaluating together for multiclassification
    mainly used for cross validation
    does not checkpoint model

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
    # specialty = config['specialty']
    RANDOM_SEED = config['RANDOM_SEED']


    # initialize tokenizer
    # tokenizer = BertTokenizer.from_pretrained(os.path.join(pretrained_path, model_name))

    # create data loaders on datasets
    train_data_loader = create_data_loader(df_train, text_col, label_col, tokenizer, int(MAX_LEN), int(BATCH_SIZE))
    val_data_loader = create_data_loader(df_val, text_col, label_col, tokenizer, int(MAX_LEN), int(BATCH_SIZE))

    ## training params
    class_weight = []
    sample = df_train[label_col].value_counts().to_dict() # full composition

    for label in indexes_to_labels:
        class_weight.append(max(sample.values()) / sample[label])

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
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
        
    
    # after training evaluate
    if focused_indexes:
        train_score_all = eval_func(trues, preds, average = None)
        train_score = np.mean(train_score_all[focused_indexes])
    else:
        train_score = eval_func(trues, preds, average = 'macro')

    print(f'Train Score: {round(train_score, 4)}  Avg loss: {round(np.mean(train_loss), 4)}  ')

    
    preds, preds_probas, trues, val_loss = eval_model(
                                            model,
                                            val_data_loader,
                                            loss_fn,
                                            device,
                                            threshold,
                                            binary
                                            )
    
    
    if focused_indexes:
        val_score_all = eval_func(trues, preds, average = None)
        for index in focused_indexes:
            print(f'{indexes_to_labels[index]}: Val Score {val_score_all[index]} ')
        val_score = np.mean(val_score_all[focused_indexes])
        
    else:
        val_score = eval_func(trues, preds, average = 'macro')

    print(f'Val Score: {round(val_score, 4)}  Avg loss: {round(np.mean(val_loss), 4)} ')
    print()
    

        
            

    return val_score



def train_multi_w_eval_steps(
                            model,
                            df_train, 
                            df_val, 
                            label_col,
                            text_col,
                            config,
                            best_val_score_global,
                            device, 
                            save_model_name,
                            labels_to_indexes,
                            indexes_to_labels, 
                            eval_freq,
                            early_stopping = 10,
                            focused_indexes = None,
                            save_path = None,
                            save_metric = f1_score,
                            metrics_list = {
                                        "f1": f1_score
                                     }
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
    
    print(pretrained_path)
    # initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained(pretrained_path)
    
    # create data loaders on datasets
    train_data_loader = create_data_loader(df_train, text_col, label_col, tokenizer, int(MAX_LEN), int(BATCH_SIZE))
    val_data_loader = create_data_loader(df_val, text_col, label_col, tokenizer, int(MAX_LEN), int(BATCH_SIZE))
    
    
    # get list eval steps 
    epoch_steps = len(train_data_loader)
    total_steps = epoch_steps * EPOCHS
    total_evaluations = eval_freq * EPOCHS
    print(f'Total Training Steps: {total_steps}')
    eval_steps = [int(total_steps/total_evaluations) * i for i in range(1, total_evaluations)]
    eval_steps.append(total_steps)
    print(f'Eval at steps: {eval_steps}')
    

    ## assigning weights to each label class to account for imbalance
    class_weight = []
    sample = df_train[label_col].value_counts().to_dict()

    for label in indexes_to_labels:
        class_weight.append(max(sample.values()) / sample[label])
    if focused_indexes: # if focused index boost their weights 
        for index in focused_indexes:
            class_weight[index] = class_weight[index] * float(boost)
 
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
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
    best_val_score = best_val_score_global
    best_model_name = None
    binary = False
    threshold = 0
    
    global_step = 1
    eval_ind = 0
    val_losses_list = [] # record val loss every eval step -> for early stopping
    running_train_loss = 0
    patience_count = 0
    val_score_list = []
    
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
                val_preds, val_preds_probas, val_trues, val_losses = eval_model(
                                                                     model,
                                                                     val_data_loader,
                                                                     loss_fn,
                                                                     device,
                                                                     threshold,
                                                                     binary
                                                                                    )
                val_score_by_label = {}
                
                if focused_indexes:

                    eval_results = evaluate_by_metrics(val_trues, val_preds, metrics_list, average = None, verbose=False)
                    val_score_all = save_metric(val_trues, val_preds, average=None, zero_division=0)
                    
                    # print out scores to console
                    data_all = []
                    for index in focused_indexes:
                        data = []
                        label_name = indexes_to_labels[index]
                        data.append(label_name)
                        for metric_name in eval_results:
                            score = round(eval_results[metric_name][index], 3)
                            data.append(score)
                        
                        data_all.append(data)
                        val_score_by_label[indexes_to_labels[index]] = round(val_score_all[index], 3)
                    
                    print()
                    print(tabulate(data_all, headers=['Label'] + list(eval_results.keys())))
                    print()
                        
                        
                    val_score = np.mean(val_score_all[focused_indexes])

                else:
                    eval_results = evaluate_by_metrics(val_trues, val_preds, metrics_list, average = 'macro', verbose=True)
                    val_score = save_metric(val_trues, val_preds, average = 'macro')
                
                print(f'Overall Score: {val_score}')
                print()


                if save_path:  # if a save path is provided, save model
            
                    if val_score > best_val_score: # if f1 score better. save model checkpoint
                        if not os.path.isdir(save_path):
                            os.mkdir(save_path) # create directory if not exist
                        
                        
                        model_info = {
                                'val_score': val_score,
                                'val_loss': float(np.round(np.mean(val_losses), 4)),
                                'time_generated': datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
                                'focused_labels':  [indexes_to_labels[x]for x in focused_indexes] if focused_indexes  else [],
                                'val_f1_by_focused_label': val_score_by_label
                            }
                
                        files = {
                            'config.json': config,
                            'model_info.json': model_info,
                            'labels_to_indexes.json': labels_to_indexes,
                            'indexes_to_labels.json': indexes_to_labels
                            
                        }
                        
                        save_model_v2(model, tokenizer, save_model_name, save_path, files  )

                        best_val_score = val_score # update best f1 score

                
                val_score_list.append(val_score)          
                val_loss = np.mean(val_losses) # getting average val loss
                val_losses_list.append(val_loss)
                
                
                # check if needed to be early stopped:
                if early_stopping:
                    if patience_count > early_stopping:
                        if val_score_list[-1] > val_score_list[-(early_stopping + 1)]:
                            print('Early Stopping..')
                            print('Val F1 List: ', val_score_list)
                            return None, None
                
                    patience_count += 1
       
            
            global_step += 1
        
  

    return best_val_score, best_model_name



def train_multi_and_checkpoint(
                            model,
                            df_train, 
                            df_val, 
                            label_col,
                            text_col,
                            config,
                            best_val_score_global,
                            device, 
                            save_model_name,
                            labels_to_indexes,
                            indexes_to_labels, 
                            focused_indexes = None,
                            save_path = None,
                            metrics_list = {
                                        "f1": f1_score
                                     }
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
    
    print(pretrained_path)
    # initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained(pretrained_path)
    
    # create data loaders on datasets
    train_data_loader = create_data_loader(df_train, text_col, label_col, tokenizer, int(MAX_LEN), int(BATCH_SIZE))
    val_data_loader = create_data_loader(df_val, text_col, label_col, tokenizer, int(MAX_LEN), int(BATCH_SIZE))
    

    ## assigning weights to each label class to account for imbalance
    class_weight = []
    sample = df_train[label_col].value_counts().to_dict()

    for label in indexes_to_labels:
        class_weight.append(max(sample.values()) / sample[label])
    if focused_indexes: # if focused index boost their weights 
        for index in focused_indexes:
            class_weight[index] = class_weight[index] * float(boost)
 
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
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
    best_val_score = best_val_score_global
    best_model_name = None
    binary = False
    threshold = 0
    

    val_losses_list = [] # record val loss every eval step -> for early stopping
    running_train_loss = 0
 
    
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
            
            
    # after training finishes evaluate
    val_preds, val_preds_probas, val_trues, val_losses = eval_model(
                                                            model,
                                                            val_data_loader,
                                                            loss_fn,
                                                            device,
                                                            threshold,
                                                            binary
                                                                        )
    
    if focused_indexes:

        eval_results = evaluate_by_metrics(val_trues, val_preds, metrics_list, average = None, verbose=False)
        
        # print out scores to console
        data_all = []
        for index in focused_indexes:
            data = []
            label_name = indexes_to_labels[index]
            data.append(label_name)
            for metric_name in eval_results:
                score = round(eval_results[metric_name][index], 3)
                data.append(score)
            
            data_all.append(data)
        
        print()
        print(tabulate(data_all, headers=['Label'] + list(eval_results.keys())))
        print()
            
            
    else:
        eval_results = evaluate_by_metrics(val_trues, val_preds, metrics_list, average = 'macro', verbose=True)
    
    print()


    if save_path:  # if a save path is provided, save model

        if not os.path.isdir(save_path):
            os.mkdir(save_path) # create directory if not exist
        
        
        model_info = {
                'val_scores': eval_results,
                'val_loss': float(np.round(np.mean(val_losses), 4)),
                'time_generated': datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
                'focused_labels':  [indexes_to_labels[x]for x in focused_indexes] if focused_indexes  else []
            }

        files = {
            'config.json': config,
            'model_info.json': model_info,
            'labels_to_indexes.json': labels_to_indexes,
            'indexes_to_labels.json': indexes_to_labels
            
        }
        
        save_model_v2(model, tokenizer, save_model_name, save_path, files )


    
    val_loss = np.mean(val_losses) # getting average val loss
    val_losses_list.append(val_loss)
    
        
        
  

    return best_val_score, best_model_name

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

def evaluate_single_spec_w_model_files(model_name, model_dir, df, text_col, label_col, pretrained_path):
    """
    Evaluate a single specialty model for DFD
    """
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model_id, specialty = model_name.split('-')
    model_path = os.path.join(model_dir, model_name)
    print(f'Evaluating Single Specialty Mutli Dtree Model: {specialty}...')
    
    
    # read in label map
    with open(os.path.join(model_path, f'{model_id}-indexes_to_labels.json'), 'rb') as handle:
        indexes_to_labels = json.load(handle)
    with open(os.path.join(model_path, f'{model_id}-labels_to_indexes.json'), 'rb') as handle:
        labels_to_indexes = json.load(handle)

    
        
    with open(os.path.join(model_path, f'{model_id}-config.json'), 'rb') as handle:
        config = json.load(handle)
        
    MAX_LEN = config['MAX_LEN']
    BATCH_SIZE = config['BATCH_SIZE']
    
    # load model from local
    model = CustomBertMultiClassifier(pretrained_path, len(labels_to_indexes), device)
    model.load_state_dict(torch.load(os.path.join(model_path, f'{model_id}-model.bin')))
    model = model.to(device)
    model.eval()
    
    tokenizer = BertTokenizer.from_pretrained(model_path)
    
    # prepare test data
    df['label'] = df[label_col].apply(lambda x: labels_to_indexes.get(x, labels_to_indexes['OTHERS']))
         
    texts, preds, preds_proba, trues, preds_proba_all = predict_multi_cohort(df, text_col, 'label', model, tokenizer, MAX_LEN, device, BATCH_SIZE)
    
    df['pred'] = preds
    
    precisions = precision_score(df.label, df.pred, average = None)
    recalls = recall_score(df.label, df.pred, average = None)
    f1s = f1_score(df.label, df.pred, average = None)
    
    for label in labels_to_indexes:
        ind = labels_to_indexes[label]
        print(f'{label}: Precision {precisions[ind]} Recall {recalls[ind]} F1 {f1s[ind]}')
        print()
    
    
    return df, (precisions, recalls, f1s), indexes_to_labels




####
# Helper functions
#####
def model_selection(path):
    model_names = os.listdir(path)

    for name in model_names:
        model_id = name.split('-')[0]
        model_info = json.load(open(f'{path}/{name}/{model_id}-model_info.json'))
        print(name)
        print(model_info)
        print()


def move_model_files(model_dir, model_name, dest_dir):
    """
    Move a directory of model files to another directory, dir will be created if not exists
    """

    
    src = os.path.join(model_dir, model_name)
    
    
    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)

    dst = os.path.join(dest_dir, model_name)
    
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    
    
    print('Files Copied Successfully.')


