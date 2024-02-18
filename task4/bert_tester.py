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
PATH = './toxic_net_image_best.pth'

class BertDataset(Dataset):
    def __init__(self, tokenizer,max_length):
        super(BertDataset, self).__init__()
        self.train_csv=pd.read_csv('testset.csv', delimiter=',', quotechar='"')
        self.tokenizer=tokenizer
        self.target=self.train_csv.iloc[:,1]
        self.max_length=max_length
        
    def __len__(self):
        return len(self.train_csv)
    
    def __getitem__(self, index):
        
        text1 = self.train_csv.iloc[index,0]
        
        inputs = self.tokenizer.encode_plus(
            text1 ,
            None,
            pad_to_max_length=True,
            add_special_tokens=True,
            return_attention_mask=True,
            max_length=self.max_length,
        )
        ids = inputs["input_ids"]
        token_type_ids = inputs["token_type_ids"]
        mask = inputs["attention_mask"]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'target': torch.tensor(self.train_csv.iloc[index, 1], dtype=torch.long)
            }
    
tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")

dataset= BertDataset(tokenizer, max_length=100)

testloader=DataLoader(dataset=dataset,batch_size=32)

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

num_correct = 0
num_samples = 0

loop=tqdm(enumerate(testloader),leave=False,total=len(testloader))

for batch, dl in loop:
    ids=dl['ids'].to(DEVICE)
    token_type_ids=dl['token_type_ids'].to(DEVICE)
    mask= dl['mask'].to(DEVICE)
    label=dl['target'].to(DEVICE)
    label = label.unsqueeze(1).to(DEVICE)
    output=model(
        ids=ids,
        mask=mask,
        token_type_ids=token_type_ids).to(DEVICE)
    label = label.type_as(output).to(DEVICE)

    pred = np.where(output.cpu() >= 0, 1, 0)

    num_correct += sum(1 for a, b in zip(pred, label) if a[0] == b[0])
    num_samples += pred.shape[0]

accuracy = num_correct/num_samples
print(accuracy)