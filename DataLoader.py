import os
import utils
import torch
from torch import nn
from torch.utils import data
import json
import numpy as np

def read_data():
    """将数据集加载到文本行列表中"""
    data_dir = 'D:/Project/Python/data/News-Headlines-Dataset-For-Sarcasm-Detection-master'
    secntences, res = [], []
    with open(os.path.join(data_dir, 'Sarcasm_Headlines_Dataset.json'), 'r') as f:
        raw_text = f.read()
    text = [line for line in raw_text.split('\n')]
    for line in text:
        tmp = json.loads(line)
        secntences.append(tmp['headline'].split())
        res.append(tmp['is_sarcastic'])
    return secntences, res

# def batchify(data):
#     sentences, reses = [], []
#     max_len = max(len(line) for line, _ in data)
#     for line, res in data:
#         cur_len = len(line)
#         sentences += [line + [[0] * 100] * (max_len - cur_len)]
#         reses.append(res)
#     tmp = torch.tensor(sentences)
#     tmp = tmp.reshape((1, 1) + tmp.shape)
#     return (tmp, torch.tensor(reses))
        

def load_data(batch_size):
    """将原始数据集进行预处理并返回迭代器"""
    num_workers = 0
    embed_size = 100
    sentences, res = read_data()
    vocab = utils.Vocab(sentences, min_freq = 5)
    prenet = nn.Sequential(nn.Embedding(num_embeddings=len(vocab), embedding_dim=embed_size), nn.Embedding(num_embeddings=len(vocab), embedding_dim=embed_size))
    prenet.load_state_dict(torch.load('preprocessing.params'))
    prenet = prenet[0]
    corpus = [[vocab[token] for token in line] for line in sentences]
    vec = []
    max_len = max(len(line) for line in corpus)
    for line in corpus:
        line = prenet(torch.tensor(line))
        line = line.detach().numpy().tolist()
        vec += [line + [[0] * 100] * (max_len - len(line))]
    vec = torch.tensor(vec)
    res = torch.tensor(res)

    class Dataset(torch.utils.data.Dataset):
        def __init__(self, vecs, reses):
            assert len(vecs) == len(reses)
            self.vecs = vecs
            self.reses = reses
        
        def __getitem__(self, index):
            return (self.vecs[index], self.reses[index])

        def __len__(self):
            return len(self.vecs)

    dataset = Dataset(vec, res)
    return data.DataLoader(dataset, batch_size, shuffle=True, num_workers=num_workers)
