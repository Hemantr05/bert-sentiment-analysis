import torch
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from transformers import BertTokenizer
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, RandomSampler
from transformers import AdamW, get_linear_schedule_with_warmup

tokenizer = BertTokenizer.from_pretrained(
    'bert-base-uncased',
    do_lower_case=True
)

def f1_score_func(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, preds_flat, average = 'weighted')

def accuracy_per_class(preds, labels):
    label_dict_inverse = {v: k for k, v in enumerate(df.target.unique())}
    
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    
    for label in np.unique(labels_flat):
        y_preds = preds_flat[labels_flat==label]
        y_true = labels_flat[labels_flat==label]
        print(f'Class: {label_dict_inverse[label]}')
        print(f'Accuracy:{len(y_preds[y_preds==label])}/{len(y_true)}\n')

def load_csv(path, columns=['id', 'text', 'target']):
    df = pd.read_csv(path, usecols=columns)
    df.set_index('id', inplace=True)
    return df

def get_labels(df):
    possible_labels = df.target.unique()
    label_dict = {}
    for index, possible_label in enumerate(possible_labels):
        label_dict[possible_label] = index
    return label_dict

def train_test_split(data):
    label_dict = get_labels(data)

    X_train, X_val, y_train, y_val = train_test_split(data.index.values, 
                                                    data.target.values, 
                                                    test_size=0.15,
                                                    random_state=42,
                                                    stratify=data.target.values)

    data['data_type'] = ['not_set']*data.shape[0]
    # print(data.head())

    data.loc[X_train, 'data_type'] = 'train'
    data.loc[X_val, 'data_type'] = 'val'

    data.groupby(['target', 'data_type']).count()

    return label_dict, data


def train_embeddings(df):
    encoded_data_train = tokenizer.batch_encode_plus(
        df[df.data_type=='train'].text.values,
        add_special_tokens=True,
        return_attention_mask=True,
        pad_to_max_length=True,
        max_length=256,
        return_tensors='pt'
    )

    input_ids_train = encoded_data_train['input_ids']
    attention_masks_train = encoded_data_train['attention_mask']
    labels_train = torch.tensor(df[df.data_type=='train'].target.values)

    return input_ids_train, attention_masks_train, labels_train

def val_embeddings(df):
    encoded_data_val = tokenizer.batch_encode_plus(
        df[df.data_type=='val'].text.values,
        add_special_tokens=True,
        return_attention_mask=True,
        pad_to_max_length=True,
        max_length=256,
        return_tensors='pt'
    )

    input_ids_val = encoded_data_val['input_ids']
    attention_masks_val = encoded_data_val['attention_mask']
    labels_val = torch.tensor(df[df.data_type=='val'].target.values)

    return input_ids_val, attention_masks_val, labels_val


def train_dataloader(input_ids_train,
                    attention_masks_train,
                    labels_train,
                    batch_size=4
                ):

    dataset_train = TensorDataset(
        input_ids_train, 
        attention_masks_train,
        labels_train
    )

    dataloader_train = DataLoader(
        dataset_train,
        sampler=RandomSampler(dataset_train),
        batch_size=batch_size
    )

    return dataloader_train

def val_dataloader(input_ids_val,
                    attention_masks_val,
                    labels_val,
                    batch_size=32
                ):
    dataset_val = TensorDataset(
        input_ids_val, 
        attention_masks_val,
        labels_val
    )

    dataloader_val = DataLoader(
        dataset_val,
        sampler=RandomSampler(dataset_val),
        batch_size=batch_size
    )

    return dataloader_val


def hyperparameter_init(dataloader_train,
                        model,
                        epochs,
                        lr,
                        eps,
                        num_warmup_steps):
    optimizer = AdamW(
        model.parameters(),
        lr = lr,
        eps = eps
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps = len(dataloader_train)*epochs
    )

    return optimizer, scheduler