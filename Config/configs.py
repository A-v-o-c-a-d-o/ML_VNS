import torch


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 4
MAX_LEN = 256
BATCH_SIZE = 5
LR = 2e-4
# with open('drive/MyDrive/Colab_Notebooks/Foody_data/Stopwords.txt', 'r', encoding="utf-8") as f:
#     stop_set = set(m.strip() for m in f.readlines())
#     stopwords = list(frozenset(stop_set))

n = 3