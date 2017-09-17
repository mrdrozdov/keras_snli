import torch
from torch.autograd import Variable

import time

from demo import Model


class Timer(object):

    def __enter__(self):
        self.start = time.process_time()
        return self

    def __exit__(self, *args):
        self.end = time.process_time()
        self.interval = self.end - self.start


def product_copy(model, p, h):
    bp, bh = p.size(0), h.size(0)
    multip = p.repeat(bh, 1).view(bp * bh, p.size(1))
    multih = h.repeat(bp, 1)
    return model(multip, multih, product=False)


def run():
    fn = 'noreg.h5'
    model = Model(fn)

    batch_size = 100
    seq_length = 30

    fake_p = Variable(torch.arange(0, batch_size * seq_length).view(batch_size, seq_length).long())
    fake_h = Variable(torch.arange(0, batch_size * seq_length).view(batch_size, seq_length).long())

    with Timer() as t:
        out = model(fake_p, fake_h)
    print("[out] Elapsed:", t.interval)

    with Timer() as t:
        out_copy = product_copy(model, fake_p, fake_h)
    print("[copy] Elapsed:", t.interval)

    with Timer() as t:
        out_broadcast = model(fake_p, fake_h, product=True)
    print("[broadcast] Elapsed:", t.interval)

    assert (out_broadcast - out_broadcast).sum().abs().data[0] < 1e-8


if __name__ == '__main__':
    run()
