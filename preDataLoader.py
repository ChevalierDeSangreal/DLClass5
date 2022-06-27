import utils
import torch
import os
import math
import random
import json

# def read_data():
#     """将数据集加载到文本行列表中"""
#     data_dir = 'D:/Project/Python/data/News-Headlines-Dataset-For-Sarcasm-Detection-master'
#     with open(os.path.join(data_dir, 'Sarcasm_Headlines_Dataset.json'))as f:
#         raw_text = f.read()
#     return [line.split() for line in raw_text.split('\n')]

def read_data():
    """将数据集加载到文本行列表中"""
    data_dir = 'D:/Project/Python/data/News-Headlines-Dataset-For-Sarcasm-Detection-master'
    data, res = [], []
    with open(os.path.join(data_dir, 'Sarcasm_Headlines_Dataset.json'), 'r') as f:
        raw_text = f.read()
    text = [line for line in raw_text.split('\n')]
    for line in text:
        tmp = json.loads(line)
        data.append(tmp['headline'].split())
        res.append(tmp['is_sarcastic'])
    return data, res


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
            indices = list(range(max(0, i - window_size), min(len(line), i + 1 + window_size)))

            #
            # if len(indices) == 0:
            #     print("Error: len of indice = 0!!!")
            # if len(indices) == 1:
            #     print(indices, len(line), i, window_size)
            #
            
            # 从上下文词排除中心词
            indices.remove(i)

            contexts.append([line[idx] for idx in indices])

    #
    # for i in contexts:
    #     if len(i) == 0:
    #         print("Warnning:len of contexts == 0!!!")
    #

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

def get_negatives(all_contexts, vocab, counter, K):
    """返回负采样中的噪声词"""
    # 索引从1开始，0为未知标记
    sampling_weights = [counter[vocab.to_tokens(i)] ** 0.75 for i in range(1, len(vocab))]
    all_negatives, generator = [], RandomGenerator(sampling_weights)
    for contexts in all_contexts:
        negatives = []
        while len(negatives) < len(contexts) * K:
            neg = generator.draw()
            if neg not in contexts:
                negatives.append(neg)
        all_negatives.append(negatives)
    return all_negatives

def batchify(data):
    """
    返回带负采样的跳元模型的小批量样本
    将上下文词和噪声词连接起来并填充零，使长度相同，为最大值
    输入是长度等于批量大小的列表，每个元素是由中心词、上下文词和噪声词组成的样本
    输出一个tensor类型的长度相同的小批量，其中包括一个用于判断是否为填充的mask
    """
    max_len = max(len(c) + len(n) for _, c, n in data)
    centers, contexts_negatives, masks, labels = [], [], [], []
    for center, context, negative in data:
        cur_len = len(context) + len(negative)

        #
        # if cur_len == 0:
        #     print("Warning:cur_len == 0!!!")
        #

        centers += [center]
        contexts_negatives += [context + negative + [0] * (max_len - cur_len)]
        masks += [[1] * cur_len + [0] * (max_len - cur_len)]
        labels += [[1] * len(context) + [0] * (max_len - len(context))]

    # 调试代码
    # tmasks = torch.tensor(masks).sum(axis=1)
    # for i in tmasks:
    #     if i == 0:
    #         print('Warning!!!!')
    #

    return (torch.tensor(centers).reshape((-1, 1)), torch.tensor(contexts_negatives), torch.tensor(masks), torch.tensor(labels))

def load_data(batch_size, max_window_size, num_noise_words):
    """下载数据集并返回数据迭代器和词表"""
    num_workers = 0
    sentences, res = read_data()
    vocab = utils.Vocab(sentences, min_freq=5)
    subsampled, counter = subsample(sentences, vocab)
    corpus = [vocab[line] for line in subsampled]
    all_centers, all_contexts = get_centers_and_contexts(corpus, max_window_size)
    all_negatives = get_negatives(all_contexts, vocab, counter, num_noise_words)

    class Dataset(torch.utils.data.Dataset):
        def __init__(self, centers, contexts, negatives):
            assert len(centers) == len(contexts) == len(negatives)
            self.centers = centers
            self.contexts = contexts
            self.negatives = negatives
        
        def __getitem__(self, index):
            return (self.centers[index], self.contexts[index], self.negatives[index])

        def __len__(self):
            return len(self.centers)
        
    dataset = Dataset(all_centers, all_contexts, all_negatives)
    data_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True, collate_fn=batchify, num_workers=num_workers)
    return data_iter, vocab