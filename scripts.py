from Model.text import MyModel
from Model.img import Basic_CNN_Module
from Config.configs import *
from torch.optim import Adam
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from Functions.functions import *


def training():
    for epoch in range(EPOCHS):
        train(model=model, img_model=cnn, loss_func=loss_func, img_loss_func=img_loss_func, optimizer=optimizer, img_optimizer=img_optimizer, data_loader=train_dataloader, epoch=epoch, lr_scheduler=lr_scheduler)
        test(model=model, img_model=cnn, data_loader=test_dataloader, loss_func=loss_func, img_loss_func=img_loss_func)

def save(model, path):
    torch.save(model, path)

def testing(n):
    predict(model, cnn, n)

if __name__ == "__main__":
    print(device)
    model = torch.load('Model/model.pt').to(device)
    cnn = torch.load('Model/cnn.pt').to(device)
    model.eval()
    cnn.eval()
    # model = MyModel().to(device)
    # cnn = Basic_CNN_Module().to(device)
    loss_func = torch.nn.CrossEntropyLoss()
    img_loss_func = torch.nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=2e-5)
    img_optimizer = Adam(cnn.parameters(), lr=LR)
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
    train_dataloader, test_dataloader = get_dataloader()
    lr_scheduler = get_linear_schedule_with_warmup(
                optimizer, 
                num_warmup_steps=0, 
                num_training_steps=len(train_dataloader)*EPOCHS
            )

    # uncomment this to train and save model
    # training()
    # save(model, 'Model/model.pt')
    # save(cnn, 'Model/cnn.pt')
    testing(n)