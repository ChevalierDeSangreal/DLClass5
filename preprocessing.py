import math
import torch
from torch import nn
from preDataLoader import load_data
from utils import SigmoidBCELoss, try_gpu, Accumulator
import utils
from d2l import torch as d2l
from preDataLoader import read_data

batch_size, max_window_size, num_noise_words = 512, 5, 5
embed_size = 100
data_iter, vocab = load_data(batch_size, max_window_size, num_noise_words)

loss = SigmoidBCELoss()

def skip_gram(center, contexts_and_negatives, embed_v, embed_u):
    """输出中的每个元素是中心词向量和上下文或噪声词向量的点积"""
    v = embed_v(center)
    u = embed_u(contexts_and_negatives)
    pred = torch.bmm(v, u.permute(0, 2, 1))
    return pred

def train(net, data_iter, lr, num_epochs, device=try_gpu()):
    def init_weights(m):
        if type(m) == nn.Embedding:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    net = net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    metric = Accumulator(2)
    for epoch in range(num_epochs):
        timer = utils.Timer()
        for i, batch in enumerate(data_iter):
            optimizer.zero_grad()
            center, context_negative, mask, label = [data.to(device) for data in batch]
            pred = skip_gram(center, context_negative, net[0], net[1])
            l = (loss(pred.reshape(label.shape).float(), label.float(), mask) / mask.sum(axis=1) * mask.shape[1])
            l.sum().backward()
            optimizer.step()
            metric.add(l.sum(), l.numel())
    print(f'loss {metric[0] / metric[1]:.3f}, 'f'{metric[1] / timer.stop():.1f} tokens/sec on {str(device)}')

net = nn.Sequential(nn.Embedding(num_embeddings=len(vocab), embedding_dim=embed_size), nn.Embedding(num_embeddings=len(vocab), embedding_dim=embed_size))
lr, num_epochs = 0.0002, 50
train(net, data_iter, lr, num_epochs)

torch.save(net.state_dict(), 'preprocessing.params')


