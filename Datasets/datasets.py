import torch
from transformers import AutoTokenizer
from underthesea import word_tokenize
from torchvision.transforms import ToTensor
from PIL import Image
import requests
from Config.configs import *


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.df = df
        self.tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
#         self.stopwords = stopwords
    
    def __len__(self):
        return len(self.df)

    def preprocess(self, s):
        s = str(s)
        s = s.lower()
        s = ''.join(e for e in s if e.isalnum() or e == ' ')
        return word_tokenize(s, format='text')
        # return ' '.join(e for e in s.split(' ') if e not in self.stopwords)
        
    def get_image(self, url):
        return ToTensor()(Image.open(requests.get(url, stream=True).raw).resize((28, 28))).to(device)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        comment = self.preprocess(row['Comment'])

        encoding = self.tokenizer.encode_plus(
            comment,
            truncation=True,
            add_special_tokens=True,
            max_length=MAX_LEN,
            padding='max_length',
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors='pt',
        )
        
        # train or test
        return {
            'comment': comment,
            'input_ids': torch.flatten(encoding['input_ids']).to(device=device),
            'attention_masks': torch.flatten(encoding['attention_mask']).to(device=device),
            'targets': torch.tensor(row['Rating'], dtype=torch.long).to(device),
            "images": self.get_image(row['image_urls'][2:-2].split("', '")[0]),
        } if self.df.shape[1] == 6 else {
            'comment': comment,
            'input_ids': torch.flatten(encoding['input_ids']).to(device=device),
            'attention_masks': torch.flatten(encoding['attention_mask']).to(device=device),
            "images": self.get_image(row['image_urls'][2:-2].split("', '")[0]),
        }