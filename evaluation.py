import pickle
import numpy as np

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

    parser.add_argument('-b', '--batch_size', default=64, type=int,
                        help='Batch size')
    parser.add_argument('-n', '--num_labels', default=925, type=int,
                        help='Number of classes')
    parser.add_argument('-m', '--model', default="koelectra", type=str, 
                        help='Model to use')
    parser.add_argument('--model_path', default="./output/model_1.pt", type=str,
                        help='Filename to load trained model.')
    parser.add_argument('--data_path', default='./data/test_data.csv', type=str,
                        help='Data path')
    
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
            if (hsk_dic.get(int(hsk)) == None):
                continue
            hsk_ind = hsk_dic.get(int(hsk))-1
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

        return input_ids, attention_mask, hsk_ind

def load_model(args, device):

    if args.model == 'kobert':
        model = BertForSequenceClassification.from_pretrained('skt/kobert-base-v1', num_labels=args.num_labels)
    elif args.model == 'koelectra':
        model = ElectraForSequenceClassification.from_pretrained("monologg/koelectra-base-v3-discriminator" , num_labels=args.num_labels)
    elif args.model == 'klue':
        model = AutoModelForSequenceClassification.from_pretrained("klue/roberta-large", num_labels=args.num_labels)
    model = nn.DataParallel(model)
    model.to(device)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()   

    return model

def main():
    args = _parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = load_model(args, device)

    with open('data/subheading_idx.pickle', 'rb') as fp: #check!
        hsk_dic = pickle.load(fp)
        hsk_dic_inv = {v: k for k, v in hsk_dic.items()}

    # make dataset
    test_data = pd.read_csv(args.data_path)[['hsk', 'description']]
    test_data = test_data.reset_index()
    test_dataset = NSMCDataset(test_data)


    # make model
    test_loader = HSKLoader(test_dataset, args.batch_size, shuffle=True)

    # evaluate
    test_total = 0

    acc_top1 = 0
    acc_top3 = 0
    acc_top5 = 0

    for input_ids_batch, attention_masks_batch, y_batch in tqdm(test_loader):
        y_batch = y_batch.to(device)
        y_pred = model(input_ids_batch.to(device), attention_mask=attention_masks_batch.to(device))[0]

        _, predicted = torch.max(y_pred, 1)

        # top1 accuracy
        acc_top1 += (predicted == y_batch).sum()
        
        # top3 accuracy
        _, top3 = torch.sort(y_pred)
        acc_top3 += (top3[:, -1] == y_batch).sum()
        acc_top3 += (top3[:, -2] == y_batch).sum()
        acc_top3 += (top3[:, -3] == y_batch).sum()

        # top5 accuracy
        _, top5 = torch.sort(y_pred)
        acc_top5 += (top5[:, -1] == y_batch).sum()
        acc_top5 += (top5[:, -2] == y_batch).sum()
        acc_top5 += (top5[:, -3] == y_batch).sum()
        acc_top5 += (top5[:, -4] == y_batch).sum()
        acc_top5 += (top5[:, -5] == y_batch).sum()

        test_total += len(y_batch)

    acc_top1 = (acc_top1.float() / test_total)
    acc_top3 = (acc_top3.float() / test_total)
    acc_top5 = (acc_top5.float() / test_total)
    print(acc_top1, acc_top3, acc_top5)

if __name__ == '__main__':
    main()  