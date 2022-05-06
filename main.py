import argparse
from src.utils import *
from src.trainer import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_dataset", default="data/train (1).csv")
    parser.add_argument("--epochs", default=5, help="number of training iterations")
    parser.add_argument("--train_batch_size", default=4, help="number of batches to divide the training set")
    parser.add_argument("--val_batch_size", default=32, help="number of batches to divide the training set")
    parser.add_argument("--lr", default=1e-5, help="learning rate of the model")
    parser.add_argument("--eps", default=1e-8, help="epsilon parameter for AdamW optimizer")
    parser.add_argument("--num_warmup_steps", default=0, help="number of steps to warm up learning")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    df = load_csv(args.training_dataset)
    label_dict, data = train_test_split(df)
    input_ids_train, attention_masks_train, labels_train = train_embeddings(data)
    input_ids_val, attention_masks_val, labels_val = val_embeddings(data)
    dataloader_train = train_dataloader(input_ids_train, attention_masks_train, labels_train, args.train_batch_size)
    dataloader_val = val_dataloader(input_ids_val, attention_masks_val, labels_val, args.val_batch_size)
    train(dataloader_train, dataloader_val, epochs=args.epochs, lr=args.lr, eps=args.eps, num_warmup_steps=args.num_warmup_steps, device=args.device)