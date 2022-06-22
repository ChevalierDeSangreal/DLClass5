import utils
import torch
import os
import math
import random


def read_data():
    """将数据集加载到文本行列表中"""
    data_dir = 'D:/Project/Python/data/News-Headlines-Dataset-For-Sarcasm-Detection-master'
    with open(os.path.join(data_dir, 'Sarcasm_Headlines_Dataset.json'))as f:
        raw_text = f.read()
    return [line.split() for line in raw_text.split('\n')]


def subsample(sentences, vocab):
    """下采样高频词"""
    # 排除未知词元'<unk>'
    sentences = [[token for token in line if vocab[token] != vocab.unk] for line in sentences]
    counter = utils.count_corpus(sentences)
    num_tokens = sum(counter.values())

    # 如果在下采样期间保留词元，则返回True
    def keep(token):
        return (random.uniform(0, 1) < math.sqrt(1e-4 / counter[token] * num_tokens))

    return ([[token for token in line if keep(token)] for line in sentences], counter)

def get_centers_and_contexts(corpus, max_window_size):
    """返回跳元模型中的中心词和上下文词"""
    centers, contexts = [], []
    for line in corpus:
        # 要形成“中心词-上下文词对”，每个句子至少需要有两个词
        if len(line) < 2:
            continue
        centers += line
        for i in range(len(line)):
            window_size = random.randint(1, max_window_size)
            indices = list(range(max(0, i - window_size), min(len(line), i + window_size)))
            # 从上下文词排除中心词
            indices.remove(i)
            contexts.append([line[idx] for idx in indices])
    return centers, contexts

class RandomGenerator:
    """根据n个采样权重在{1, ..., n}中随机抽取"""
    def __init__(self, sampling_weights):
        self.population = list(range(1, len(sampling_weights) + 1))
        self.sampling_weights = sampling_weights
        self.candidates = []
        self.i = 0
    
    def draw(self):
        if self.i == len(self.candidates):
            self.candidates = random.choices(self.population, self.sampling_weights, k=10000)
            self.i = 0
        self.i += 1
        return self.candidates[self.i - 1]


sentences = read_data()
print(f'number of sentence:{len(sentences)}')
vocab = utils.Vocab(sentences, min_freq=5)
print(f'vocab size:{len(vocab)}')
subsampled, counter = subsample(sentences, vocab)
corpus = [vocab[line] for line in subsampled]
# print(corpus[:3])
all_centers, all_contexts = get_centers_and_contexts(corpus, 5)
print(f'number of center-context pair:{sum(len(contexts) for contexts in all_contexts)}')
generator = RandomGenerator([2, 3, 4])
# print([generator.draw() for _ in range(10)])

