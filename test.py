import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from src.trainer import model
from src.utils import tokenizer, load_csv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

def test_embeddings(df_test):
    encoded_data_test = tokenizer.batch_encode_plus(
        df_test.text.values,
        add_special_tokens=True,
        return_attention_mask=True,
        pad_to_max_length=True,
        max_length=256,
        return_tensors='pt'
    )

    input_ids_test = encoded_data_test['input_ids']
    attention_masks_test = encoded_data_test['attention_mask']
    return input_ids_test, attention_masks_test

def test_dataloader(input_ids_test, 
                    attention_masks_test,
                    batch_size):
    dataset_test = TensorDataset(input_ids_test, 
                                attention_masks_test)

    dataloader_test = DataLoader(
        dataset_test,
        batch_size=batch_size
    )
    return dataloader_test

def evaluate(dataloader_test):

    model.eval()
    
    loss_test_total = 0
    predictions, true_vals = [], []
    
    for batch in tqdm(dataloader_test):
        
        batch = tuple(b.to(device) for b in batch)

        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                 }

        with torch.no_grad():        
            outputs = model(**inputs)
        
        logits = outputs[0]

        logits = logits.detach().cpu().numpy()
        predictions.append(logits)
    
    
    predictions = np.concatenate(predictions, axis=0)
    preds_flat = np.argmax(predictions, axis=1).flatten()
        
    return preds_flat


if __name__ == "__main__":
    parse = argparse.ArgumentParser
    parse.add_argument("--test_set", default="data/test (1).csv")
    parse.add_argument("--batch_size", default=32)
    args = parse.parse_args()

    data = load_csv(args.test_set)
    input_ids_test, attention_masks_test = test_embeddings(data)

    dataloader_test = test_dataloader(input_ids_test, attention_masks_test, batch_size=32)

    predictions = evaluate(dataloader_test)

    submission = pd.read_csv("sample_submission.csv")
    submission["target"] = predictions
    submission.set_index('id')
    submission.to_csv("bert-sentiment-analysis.csv")