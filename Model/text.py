import torch
from torch import nn
from transformers import AutoModel


class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.bert = AutoModel.from_pretrained("vinai/phobert-base")
        self.drop = nn.Dropout(p=0.3)
        self.fc = nn.Linear(self.bert.config.hidden_size, 2)
        nn.init.normal_(self.fc.weight, std=0.02)
        nn.init.normal_(self.fc.bias, 0)

    def forward(self, input):
        _, output = self.bert(
            input_ids=input['input_ids'],
            attention_mask=input['attention_masks'],
            return_dict=False
        )

        x = self.drop(output)
        x = self.fc(x)
        return x