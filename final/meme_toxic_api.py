import pandas as pd
import numpy as np
import transformers
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PATH = '../task4/toxic_net_image_best.pth'
    
tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")

class BERT(nn.Module):
    def __init__(self):
        super(BERT, self).__init__()
        self.bert_model = transformers.BertModel.from_pretrained("bert-base-uncased")
        self.out = nn.Linear(768, 1)
        
    def forward(self,ids,mask,token_type_ids):
        _,o2= self.bert_model(ids,attention_mask=mask,token_type_ids=token_type_ids, return_dict=False)
        
        out= self.out(o2)
        
        return out
    

model=BERT().to(DEVICE)

model.load_state_dict(torch.load(PATH))

model.eval()

text = ""
objects = []


def is_toxic(text, objects):
    text += ". "
    if len(objects) > 0:
        text += "objects: " + objects[0]
        for j in objects[1:]:
            text += ", " + j
        text += '.'

    inputs = tokenizer.encode_plus(
                text ,
                None,
                pad_to_max_length=True,
                add_special_tokens=True,
                return_attention_mask=True,
                max_length=100,
            )

    new_input = {
                'ids': torch.tensor(inputs["input_ids"], dtype=torch.long),
                'mask': torch.tensor(inputs["attention_mask"], dtype=torch.long),
                'token_type_ids': torch.tensor(inputs["token_type_ids"], dtype=torch.long),
                'target': torch.tensor(0, dtype=torch.long)
                }

    output=model(
            ids=new_input['ids'].unsqueeze(0).to(DEVICE),
            mask=new_input['mask'].unsqueeze(0).to(DEVICE),
            token_type_ids=new_input['token_type_ids'].unsqueeze(0).to(DEVICE)).to(DEVICE)

    pred = np.where(output.cpu() >= 0, 1, 0)
    return pred[0][0]
