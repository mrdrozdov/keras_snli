import os
import itertools

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as _Variable

from snli_rnn import get_data

import h5py
from tqdm import tqdm

tfp = torch.from_numpy

CUDA = True
DATA_PARALLEL = False

def Variable(var):
    if CUDA:
        return _Variable(var.cuda())
    return var

def fk2(f, key):
    return f[key][key]

class Model(nn.Module):
    """
    (u'dense_2', u'dense_2', u'bias:0', (600,))
    (u'dense_2', u'dense_2', u'kernel:0', (600, 600))
    (u'dense_3', u'dense_3', u'bias:0', (600,))
    (u'dense_3', u'dense_3', u'kernel:0', (600, 600))
    (u'dense_4', u'dense_4', u'bias:0', (600,))
    (u'dense_4', u'dense_4', u'kernel:0', (600, 600))
    (u'dense_5', u'dense_5', u'bias:0', (3,))
    (u'dense_5', u'dense_5', u'kernel:0', (600, 3))
    (u'embedding_1', u'embedding_1', u'embeddings:0', (42391, 300))
    (u'time_distributed_1', u'time_distributed_1', u'bias:0', (300,))
    (u'time_distributed_1', u'time_distributed_1', u'kernel:0', (300, 300))
    """

    def __init__(self, fn):
        super(Model, self).__init__()
        with h5py.File(fn, 'r') as f:
            embedding_1 = fk2(f, 'embedding_1')['embeddings:0'][()]
            dense_2b = fk2(f, 'dense_2')['bias:0'][()]
            dense_2 = fk2(f, 'dense_2')['kernel:0'][()]
            dense_3b = fk2(f, 'dense_3')['bias:0'][()]
            dense_3 = fk2(f, 'dense_3')['kernel:0'][()]
            dense_4b = fk2(f, 'dense_4')['bias:0'][()]
            dense_4 = fk2(f, 'dense_4')['kernel:0'][()]
            dense_5b = fk2(f, 'dense_5')['bias:0'][()]
            dense_5 = fk2(f, 'dense_5')['kernel:0'][()]
            time_distributed_1b = fk2(f, 'time_distributed_1')['bias:0'][()]
            time_distributed_1 = fk2(f, 'time_distributed_1')['kernel:0'][()]

            self.embedding_1 = nn.Embedding(embedding_1.shape[0], embedding_1.shape[1])
            self.embedding_1.weight.data.set_(tfp(embedding_1))

            self.time_distributed_1 = nn.Linear(300, 300)
            self.time_distributed_1.weight.data.set_(tfp(time_distributed_1.T))
            self.time_distributed_1.bias.data.set_(tfp(time_distributed_1b))

            self.dense_2 = nn.Linear(600, 600)
            self.dense_3 = nn.Linear(600, 600)
            self.dense_4 = nn.Linear(600, 600)
            self.dense_5 = nn.Linear(600, 3)

            self.dense_2.weight.data.set_(tfp(dense_2.T))
            self.dense_2.bias.data.set_(tfp(dense_2b))
            self.dense_3.weight.data.set_(tfp(dense_3.T))
            self.dense_3.bias.data.set_(tfp(dense_3b))
            self.dense_4.weight.data.set_(tfp(dense_4.T))
            self.dense_4.bias.data.set_(tfp(dense_4b))
            self.dense_5.weight.data.set_(tfp(dense_5.T))
            self.dense_5.bias.data.set_(tfp(dense_5b))

    def forward(self, p, h):
        b, _ = p.size()
        _, _ = h.size()
        ew = self.embedding_1.weight.data.size(1)

        x = torch.cat([p, h], 0)
        e = self.embedding_1(x)
        e = e.view(-1, ew)

        zt = F.relu(self.time_distributed_1(e))  # translate
        zs = zt.view(b * 2, -1, ew).sum(1).squeeze()  # sum
        psum, hsum = zs[:b], zs[b:]
        z = torch.cat([psum, hsum], 1)

        h2 = F.relu(self.dense_2(z))
        h3 = F.relu(self.dense_3(h2))
        h4 = F.relu(self.dense_4(h3))
        h5 = self.dense_5(h4)

        y = F.softmax(h5)

        return y


MAX_LEN = 42
batch_size = 10
seq_length = 30

nli_data = get_data(os.path.expanduser('~/data/snli_1.0/snli_1.0_dev.jsonl'))
tokenizer = Tokenizer(lower=False, filters='')
tokenizer.fit_on_texts(nli_data[0] + nli_data[1])

VOCAB = len(tokenizer.word_counts) + 1
LABELS = {'contradiction': 0, 'neutral': 1, 'entailment': 2}
to_seq = lambda X: pad_sequences(tokenizer.texts_to_sequences(X), maxlen=MAX_LEN)
prepare_data = lambda data: (to_seq(data[0]), to_seq(data[1]), data[2])
nli_data = prepare_data(nli_data)

fn = 'noreg.h5'
model = Model(fn)
if CUDA:
    model.cuda()
    if DATA_PARALLEL:
        model = torch.nn.DataParallel(model, device_ids=[0, 1])
print(model)

fake_p = Variable(torch.arange(0, batch_size * seq_length).view(batch_size, seq_length).long())
fake_h = Variable(torch.arange(0, batch_size * seq_length).view(batch_size, seq_length).long())

out = model(fake_p, fake_h)
print(out)

def brute_force_iterator(nli_data, batch_size):
    prems = nli_data[0]
    hyps = nli_data[1]

    pairs = list(itertools.product(range(prems.shape[0]), range(hyps.shape[0])))

    size = prems.shape[0] * hyps.shape[0]
    num_batches = size // batch_size
    remainder = size % num_batches

    def _it():
        for i in range(num_batches):
            prem_index, hyp_index = zip(*pairs[i*batch_size:(i+1)*batch_size])
            prem_batch = Variable(tfp(prems[list(prem_index)]).long())
            hyp_batch = Variable(tfp(hyps[list(hyp_index)]).long())
            yield (prem_batch, hyp_batch)

    return size, num_batches, remainder, _it


outfn = 'out.txt'
SAVE_EVERY = 10000
buffer = []


batch_size = 100
size, num_batches, remainder, iterator = brute_force_iterator(nli_data, batch_size)
print("Skipping {} items.".format(remainder))

with open(outfn, 'w') as f:
    f.write('{}\n'.format(num_batches * batch_size))

for i, (prem_batch, hyp_batch) in enumerate(tqdm(iterator(), total=num_batches)):
    out = model(prem_batch, hyp_batch)

    buffer.append(out.data.cpu())

    if i % SAVE_EVERY == 0:
        with open(outfn, 'a') as f:
            for out in buffer:
                for preds in out.tolist():
                    preds = ' '.join(map(str, preds))
                    f.write('{}\n'.format(preds))
            del buffer
            buffer = []

