import pickle
import numpy as np
import pandas as pd
import cgi
import random

import torch
from torch import nn
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from transformers import AutoTokenizer, ElectraForSequenceClassification, AdamW, ElectraModel, ElectraTokenizer
from transformers.models.electra.modeling_electra import ElectraModel, ElectraPreTrainedModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModel

from sklearn.neighbors import NearestNeighbors

from tqdm.notebook import tqdm

import json
import os

from Preprocess_datasets import Preprocess_QA_sentences, Preprocess_QA_sentences_Quoref
from get_subgraph import get_alignment_justification  
import math
from nltk.stem.wordnet import WordNetLemmatizer
from googletrans import Translator
import copy

lmtzr = WordNetLemmatizer()

def _parse_args():
    parser = argparse.ArgumentParser(description='Training')

    parser.add_argument('-b', '--batch_size', default=64, type=int,
                        help='Batch size')
    parser.add_argument('-n', '--num_labels', default=925, type=int,
                        help='Number of classes')
    parser.add_argument('-m', '--model', default="koelectra", type=str, #TODO
                        help='Model to use')
    parser.add_argument('--model_path1', default="./output/model_1.pt", type=str,
                        help='First filename to load a trained model')
    parser.add_argument('--model_path2', default="./output/emb_model.pt", type=str,
                        help='Second filename to load a trained model')

    parser.add_argument('--input_desc', type=str,
                        help='Input description')
    parser.add_argument('--highlight_num', default=7, type=int,
                        help='Number of sentences to highlight')
    parser.add_argument('--compete_num', default=3, type=int,
                        help='Number of subheadings to show')
    
    return parser.parse_args()


args = _parse_args()

batch_size = args.batch_size
num_labels = args.num_labels
model_path1 = args.model_path1 
model_path2 = args.model_path2

sent= args.input_desc
highlight_num = args.highlight_num
compete_num = args.compete_num

device = torch.device("cuda")

# Model for the embedding
class ElectraForMultiLabelClassification(ElectraPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.electra = ElectraModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.loss_fct = BCEWithLogitsLoss()

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
    ):
        discriminator_hidden_states = self.electra(
            input_ids, attention_mask, token_type_ids, position_ids, head_mask, inputs_embeds
        )
        pooled_output = discriminator_hidden_states[0][:, 0]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        outputs = (logits,) + discriminator_hidden_states[1:]  

        if labels is not None:
            loss = self.loss_fct(logits, labels)
            outputs = (loss,) + outputs

        return outputs, pooled_output 

# Rescale the array
def rescale(input_list):
    the_array = np.asarray(input_list)
    the_max = np.max(the_array)
    the_min = np.min(the_array)
    rescale = (the_array - the_min)/(the_max-the_min)*100
    return rescale.tolist()

# Text modification for the latex
def clean_word(word_list):
    new_word_list = []
    for word in word_list:
        for latex_sensitive in ["\\", "%", "&", "^", "#", "_",  "{", "}"]:
            if latex_sensitive in word:
                word = word.replace(latex_sensitive, '\\'+latex_sensitive)
        new_word_list.append(word)
    return new_word_list

# Predict the headings and get the supporting facts
def predict():
    inputs = tokenizer(
        sent, 
        return_tensors='pt',
        truncation=True,
        max_length=512,
        pad_to_max_length=True,
        add_special_tokens=True
    )

    input_id = inputs['input_ids'][0]
    attention_mask = inputs['attention_mask'][0]

    input_id = input_id.reshape(1,512)
    attention_mask = attention_mask.reshape(1,len(attention_mask))

    # Get embedding
    result = model2(input_id.to(device), attention_mask)
    test_emb = result[1][0].reshape(1,768)

    # Get probability of the subheading and pick top 10
    test_class = model1(input_id.to(device), attention_mask=attention_mask)[0]
    cands = torch.topk(test_class,10).indices.cpu()[0].tolist()

    # Pick top 2 headings in cand_hsk
    i = 0
    first_hsk = None
    cand_hsk = []
    cand_sentences_scores = dict()
    one_hsk = True
    for cand in cands:
        if i == 0 :
            first_hsk = hsk_dic_rev[cand+1]//100
            cand_hsk.append(first_hsk)
            cand_sents =  manual_sentence_dic[first_hsk]
            for cand_sent in cand_sents: cand_sentences_scores[cand_sent] = 0
        else:
            hsk = hsk_dic_rev[cand+1]//100
            if first_hsk != hsk:
                cand_hsk.append(hsk)
                cand_sents =  manual_sentence_dic[hsk]
                for cand_sent in cand_sents: cand_sentences_scores[cand_sent] = 0
                one_hsk = False
                break
        i += 1
    if one_hsk:
        return("CONFIDENT hsk", first_hsk)
    hsk_weight = torch.topk(test_class,10).values.cpu()[0].tolist()[0] / torch.topk(test_class,10).values.cpu()[0].tolist()[i]
    
    # Calculate score in cand_sentences_scores
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    sim = cos(test_emb, kb)
    k=10
    topk = torch.topk(sim, k)
    for i in range(k):
        top = int(topk.indices[i].cpu())
        top_sim = float(topk.values[i].cpu())
        cands = kb_848590.loc[top]['sentence_index']
        cands = cands[1:len(cands)-1].split(', ')
        for cand in cands:
            ori = cand_sentences_scores.get(int(cand))
            if ori!=None:
                cand_sentences_scores[int(cand)] = ori+top_sim

    # Calculate relevance score
    embeddings_index = {}
    with open('./embedding.txt', 'rb') as fp:
        embeddings_index = pickle.load(fp)
    emb_size = 300
    with open("./manual_IDF_vals.json") as json_file:
        MultiRC_idf_vals = json.load(json_file)
    translator = Translator()
    desc = translator.translate(sent).text

    candidates = []
    candidates_ind = []
    evidences = []

    first_hsk_cands = manual_sentence_dic[cand_hsk[0]]
    for cand in first_hsk_cands:
        try:
            a= manual_df_en[manual_df_en['index']==cand]['sentence'].values[0]
        except:
            print(cand)  
        candidates.append(a)
        candidates_ind.append(cand)

    second_hsk_cands = manual_sentence_dic[cand_hsk[1]]
    for cand in second_hsk_cands:
        try:
            a= manual_df_en[manual_df_en['index']==cand]['sentence'].values[0]
        except:
            print(cand)
        candidates.append(a)
        candidates_ind.append(cand)

    ques_terms = Preprocess_QA_sentences(desc, 1)

    score, index = get_alignment_justification(ques_terms , [], candidates, embeddings_index, emb_size, MultiRC_idf_vals)

    for i in range(len(score)):
        cand_sent = cand_sentences_scores.get(candidates_ind[i])
        if cand_sent != None: 
            cand_sentences_scores[candidates_ind[i]] = cand_sent + score[i]/np.mean(score)

    cand_scores = [] 
    cand_ind = []
    for key in cand_sentences_scores.keys():
        cand_ind.append(key)
        if key in manual_sentence_dic[cand_hsk[0]]:
            cand_scores.append(hsk_weight*cand_sentences_scores[key])
        else:
            cand_scores.append(cand_sentences_scores[key])
    
    # Highlight sentences
    k = highlight_num
    final_candidates = []
    for index in torch.topk(torch.tensor(cand_scores),k).indices:
        final_candidates.append(cand_ind[int(index)])
    paragraph = []
    paragraph_check = []
    starts = []
    count=0
    for i in range(2):
        starts.append(count)
        for sent_i in manual_sentence_dic[cand_hsk[i]]:
            if sent_i in final_candidates: 
                paragraph_check.append(1)
            else: 
                paragraph_check.append(0)
            paragraph.append(manual_df.loc[sent_i]['sentence'])
            count += 1
    
    hsk6_cands = torch.topk(test_class,compete_num).indices.cpu()[0].tolist()
    hsk6_cands_code = []
    for soho_cand in hsk6_cands:
        hsk6_cands_code.append(hsk_dic_rev[int(soho_cand)+1])

    sentence_num = len(paragraph)
    attention = [(x+1.)/sentence_num*100 for x in range(sentence_num)]
    random.seed(42)
    random.shuffle(attention)

    # Generate pdf file with supporting facts
    def generate(text_list, attention_list, color='red', rescale_value = False):
            assert(len(text_list) == len(attention_list))
            if rescale_value:
                attention_list = rescale(attention_list)
            word_num = len(text_list)
            text_list = clean_word(text_list)

            latex_output = r'''\documentclass[varwidth]{oblivoir}
        \usepackage{fapapersize}
        \usefapapersize{210mm,297mm,30mm,*,30mm,32mm}
        \usepackage{xcolor}


        \begin{document}'''+'\n\n'
                
            latex_output += '\section{Item Description}'+sent+'\n'
            for idx in range(word_num):
                if idx == 0: latex_output += r'''\section{Candidate Heading: ''' + str(cand_hsk[0]) + '}'
                if idx == starts[1]: latex_output += r'''\section{Candidate Heading: ''' + str(cand_hsk[1]) + '}'
                if attention_list[idx]==1:
                    latex_output += r'''\textcolor{purple}{''' + text_list[idx] + r'''}\\'''
                else:
                    latex_output += text_list[idx] +r'''\\'''
                
            latex_output += '\section{Candidate Codes}\n'
            
            for soho_cand in hsk6_cands_code :
                latex_output += '\subsection{'+str(soho_cand)+'}\n'
                

            latex_output+= '''
        \end{document}'''
            
            return latex_output

    latex_output = generate(paragraph, paragraph_check, color='red')

    with open("./result"+".tex", 'w') as lat:
        lat.write(latex_output)
    
    # Save as pdf file
    os.path.dirname(os.path.abspath(__file__), "result")
    os.system('pdflatex -interaction=nonstopmode result.tex')

# Load files
manual_df = pd.read_csv('./manual.csv')
manual_df_en = pd.read_csv('./manual_en.csv')
with open('subheading_idx.pickle', 'rb') as fp:
    hsk_dic = pickle.load(fp)    
with open('subheading_idx_reverse.pickle', 'rb') as fp:
    hsk_dic_rev = pickle.load(fp)   
with open('manual_sentence_dic.pickle','rb') as fw:
    manual_sentence_dic = pickle.load(fw)
kb_1 = pd.read_csv('data/council.csv')
kb_2 = pd.read_csv('data/committee.csv')
kb_848590 = pd.concat([kb_1, kb_2])
kb_848590 = kb_848590.reset_index()
for i in range(len(kb_848590)):
    if type(kb_848590['sentence_index'].loc[i])==type(0.1):
        kb_848590 = kb_848590.drop(i)
kb_848590 = kb_848590.reset_index()
pd.read_csv('data/council_committee.csv',)
kb = kb.reshape(315,768)

# Load model
if args.model == 'kobert':
    self.tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
elif args.model == 'koelectra':
    self.tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
elif args.model == 'klue':
    self.tokenizer = AutoTokenizer.from_pretrained("klue/roberta-large") 

if args.model == 'kobert':
    model1 = BertForSequenceClassification.from_pretrained('skt/kobert-base-v1', num_labels=args.num_labels)
elif args.model == 'koelectra':
    model1 = ElectraForSequenceClassification.from_pretrained("monologg/koelectra-base-v3-discriminator" , num_labels=args.num_labels)
elif args.model == 'klue':
    model1 = AutoModelForSequenceClassification.from_pretrained("klue/roberta-large", num_labels=args.num_labels)
model1.load_state_dict(torch.load(model_path1)) 
model1.eval()

model2 = ElectraForMultiLabelClassification.from_pretrained("monologg/koelectra-base-v3-discriminator") #.cuda()
model2 = nn.DataParallel(model2)
model2.cuda()
model2.load_state_dict(torch.load(model_path2)) 
model2.eval()

predict()