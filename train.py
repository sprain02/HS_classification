import pandas as pd
import pickle
import argparse
import time 
import datetime 

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import BertForSequenceClassification
from transformers import ElectraForSequenceClassification, ElectraTokenizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from tqdm.notebook import tqdm

def _parse_args():
    parser = argparse.ArgumentParser(description='Training')

    parser.add_argument('-e', '--epochs', default=100, type=int,
                        help='Number of training epochs')
    parser.add_argument('-b', '--batch_size', default=64, type=int,
                        help='Batch size')
    parser.add_argument('-n', '--num_labels', default=925, type=int,
                        help='Number of classes')
    parser.add_argument('-m', '--model', default="koelectra", type=str,
                        help='Model to use')
    parser.add_argument('--load', default=None, type=str,
                        help='Filename to load trained model.')
    parser.add_argument('-r', '--learning_rate', default = 1e-5, type=float,
                        help='Learning rate')
    parser.add_argument('--data_path', default='./data/train_data.csv', type=str,
                        help='Data path')
    parser.add_argument('--output_path', default='./output', type=str,
                        help='Output path')
    
    return parser.parse_args()


class HSKDataset(Dataset):
  
    def __init__(self, dataset):

        self.labels = []
        self.input_ids = []
        self.attention_mask = []
        self.dataset = dataset

        if args.model == 'kobert':
            self.tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
        if args.model == 'koelectra':
            self.tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
        if args.model == 'klue':
            self.tokenizer = AutoTokenizer.from_pretrained("klue/roberta-large") 

        for i in range(len(dataset)):
            hsk = str(dataset.loc[i]['hsk'].item())[:6] 
            if (subheading_dic.get(int(hsk)) == None):
                continue
            hsk_ind = subheading_dic.get(int(hsk))-1
            if type(dataset.loc[i]['description']) == type(0.1):
                continue
            desc = dataset.loc[i]['description']

            inputs = self.tokenizer(
                desc, 
                return_tensors='pt',
                truncation=True,
                max_length=512,
                pad_to_max_length=True,
                add_special_tokens=True
                )
            input_ids = inputs['input_ids'][0]
            attention_mask = inputs['attention_mask'][0]

            self.labels.append(hsk_ind)
            self.input_ids.append(input_ids)
            self.attention_mask.append(attention_mask)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        attention_mask = self.attention_mask[idx] 
        hsk_ind = self.labels[idx]
        input_ids = self.input_ids[idx]

        return input_ids, attention_mask, hsk_ind #y

def load_model(args, device):
    # make model
    if args.model == 'kobert':
        model = BertForSequenceClassification.from_pretrained('skt/kobert-base-v1', num_labels=args.num_labels)
    elif args.model == 'koelectra':
        model = ElectraForSequenceClassification.from_pretrained("monologg/koelectra-base-v3-discriminator" , num_labels=args.num_labels)
    elif args.model == 'klue':
        model = AutoModelForSequenceClassification.from_pretrained("klue/roberta-large", num_labels=args.num_labels)
    model = nn.DataParallel(model)
    model.cuda()
    if args.load:
        model.load_state_dict(torch.load(args.load))

    return model

def main(args):

    device = torch.device("cuda")

    model = load_model(args, device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    # make dataset
    train_data = pd.read_csv(args.data_path)[['hsk', 'description']]
    train_dataset = HSKDataset(train_data)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    with open('data/subheading_idx.pickle', 'rb') as fp:
        subheading_dic = pickle.load(fp)

    # train and save the model
    for e in list(range(args.epochs)):
        correct = 0
        total = 0

        model.train()

        start = time.time() 

        for input_ids_batch, attention_masks_batch, y_batch in tqdm(train_loader):
            optimizer.zero_grad()
            y_batch = y_batch.to(device)
            y_pred = model(input_ids_batch.to(device), attention_mask=attention_masks_batch.to(device))[0]

            loss = F.cross_entropy(y_pred, y_batch)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(y_pred, 1)
            correct += (predicted == y_batch).sum()
            total += len(y_batch)

        acc = (correct.float() / total).item()
        end = time.time() 
        sec = (end - start)

        torch.save(model.state_dict(), args.output_path+"/model_"+str(e)+".pt")
        print("epoch:", e, "Train Loss:", loss, "Accuracy:", acc, "time:", datetime.timedelta(seconds=sec))


if __name__ == '__main__':
    args = _parse_args()
    main(args)