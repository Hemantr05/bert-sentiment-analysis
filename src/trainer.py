import torch
import random
from tqdm.notebook import tqdm
from transformers import BertForSequenceClassification

from .utils import *

seed_val = 17
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

model = BertForSequenceClassification.from_pretrained(
                                      'bert-base-uncased', 
                                      num_labels = 2,
                                      output_attentions = False,
                                      output_hidden_states = False
                                     )

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print(device)

def evaluate(dataloader_val,
                device='cuda'):

    model.eval()
    
    loss_val_total = 0
    predictions, true_vals = [], []
    
    for batch in tqdm(dataloader_val):
        
        batch = tuple(b.to(device) for b in batch)
        
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'labels':         batch[2],
                 }

        with torch.no_grad():        
            outputs = model(**inputs)
            
        loss = outputs[0]
        logits = outputs[1]
        loss_val_total += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = inputs['labels'].cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)
    
    loss_val_avg = loss_val_total/len(dataloader_val) 
    
    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)
            
    return loss_val_avg, predictions, true_vals


def train(dataloader_train,
        dataloader_val,
        device,
        epochs=5,
        lr=1e-5,
        eps=1e-8,
        num_warmup_steps=0
        ):

    optimizer, scheduler = hyperparameter_init(dataloader_train=dataloader_train,
                                                model=model, 
                                                epochs=epochs,
                                                lr=lr,
                                                eps=eps,
                                                num_warmup_steps=num_warmup_steps)

    for epoch in tqdm(range(1, epochs+1)):
        model.train()
        loss_train_total = 0
        
        progress_bar = tqdm(dataloader_train, 
                            desc='Epoch {:1d}'.format(epoch), 
                            leave=False, 
                            disable=False)
        
        for batch in progress_bar:
            model.zero_grad()
            batch = tuple(b.to(device) for b in batch)
            inputs = {
                'input_ids': batch[0],
                'attention_mask': batch[1],
                'labels': batch[2]
            }
            
            outputs = model(**inputs)
            loss = outputs[0]
            loss_train_total +=loss.item()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            
            progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item()/len(batch))})     
        
        #torch.save(model.state_dict(), f'Models/BERT_ft_Epoch{epoch}.model')
        
        tqdm.write(f'\nEpoch {epoch}')
        
        loss_train_avg = loss_train_total/len(dataloader_train)
        tqdm.write(f'Training loss: {loss_train_avg}')
        
        val_loss, predictions, true_vals = evaluate(dataloader_val)
        val_f1 = f1_score_func(predictions, true_vals)
        tqdm.write(f'Validation loss: {val_loss}')
        tqdm.write(f'F1 Score (weighted): {val_f1}')
        print("Accuracy per class: \n")
        accuracy_per_class(predictions, true_vals)
