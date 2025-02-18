#!/bin/env python3
#SBATCH -p full
#SBATCH -J cs6785_hw1
#SBATCH -c 8
#SBATCH -G a100:1
#SBATCH -N 1 -n 1
#SBATCH --mem=64G
#SBATCH -t 48:00:00
#SBATCH -o slurm_out/%j.out
#SBATCH -e slurm_out/%j.err

### Move the code from notebook.ipynb to a python script to run on slurm cluster

import os,sys,time,math,textwrap

import numpy as np

import torch
import torch.nn as nn

sys.path.append('/home/fs01/zq84/CS6785/HW1/p8')
import dataset, transformer

root = 'data/wikitext-2'

lr = .00035
context = 150
batch_size = 32
log_interval = 50
dropout = 0.2
dropoutio = 0.6

heads = 10
depth = 16

torch.manual_seed(0)
device = torch.device("cuda")

train_data = dataset.WikiText2(root, context, dataset.DatasetSplit.train)
valid_data = dataset.WikiText2(root, context, dataset.DatasetSplit.valid)
test_data = dataset.WikiText2(root, context, dataset.DatasetSplit.test)

def evaluate(data):
    model.eval()
    with torch.no_grad():
        loss = 0.
        loader = torch.utils.data.DataLoader(dataset=data,batch_size=batch_size,shuffle=False)
        for i, (x,y) in enumerate(loader):
            x, y = x.permute(1,0).to(device), y.permute(1,0).to(device)
            yhat = model(x).view(-1, train_data.word_count())
            loss += criterion(yhat, y.contiguous().view(-1))

    print()
    model.train()
    return loss / len(loader)

model = transformer.Transformer(context, train_data.word_count(), 400, 40, 900, heads, depth, tied_weights=True, dropout=dropout, dropoutio=dropoutio).to(device)
count = sum([np.prod(parm.shape) for parm in model.parameters() if parm.requires_grad])
print('Initialized graph with {} parameters'.format(count), flush=True)

criterion = nn.NLLLoss()
curr_lr = .0001
clip = .25
best_val_loss = None
epochs = 80
save = f'model_{dropout}_{dropoutio}_{epochs}.pt'

train_loader = torch.utils.data.DataLoader(dataset=train_data,batch_size=batch_size,shuffle=True)
print('Initiating training, {} iterations/epoch.'.format(len(train_loader)), flush=True)

try:
    optimizer = torch.optim.Adam(model.parameters(), lr=curr_lr)
    for epoch in range(epochs):
        t0 = time.time()
        val_loss = evaluate(valid_data)
        print('-' * 100)
        print('| checkpoint | epoch {:3d} | time: {:5.2f}s | validation loss {:5.2f} | '
                'validation perplexity {:8.2f}'.format(epoch, (time.time() - t0),
                                                       val_loss, math.exp(val_loss)))
        print('-' * 100)
        print('epoch\t\tms/batch\tlr\tloss\tperplexity', flush=True)

        if not best_val_loss or val_loss < best_val_loss:
            with open(save, 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss

        model.train()
        total_loss = 0.
        t0 = time.time()
        if epoch == 1: optimizer.param_groups[0]['lr'] = curr_lr = lr # finished warmup
        for i, (x,y) in enumerate(train_loader):
            if i % log_interval == 0 and i > 0:
                cur_loss = total_loss / log_interval
                elapsed = time.time() - t0
                print('{:3d} ({:2.1f}%)\t{:5.2f}\t\t{:1.3}\t{:5.2f}\t{:8.2f}'.format(
                    epoch, 100*i/float(len(train_loader)),
                    elapsed * 1000 / log_interval, curr_lr, cur_loss, math.exp(cur_loss)), flush=True)
                total_loss = 0
                t0 = time.time()

            x, y = x.permute(1,0).to(device), y.permute(1,0).to(device)
            model.zero_grad()
            yhat = model(x).view(-1, train_data.word_count())
            loss = criterion(yhat, y.contiguous().view(-1))
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

            total_loss += loss.item()

except KeyboardInterrupt:
    print('Graceful Exit')

print('Restoring best checkpointed model...')
with open(save, 'rb') as f:
    model = torch.load(f, weights_only=False)

test_loss = evaluate(test_data)
print('=' * 89)
print('| end of training | test loss {:5.2f} | test perplexity {:8.2f}'.format(test_loss, math.exp(test_loss)))
print('=' * 89)

print('\nUncurated samples')
print('-' * 89)

def sample():
    words = []
    model.eval()
    history = torch.randint(train_data.word_count(), (1, 1), dtype=torch.long).cuda()
    for i in range(context):
        output = model(history)
        word_weights = output[-1].squeeze().exp().cpu()
        word_idx = torch.multinomial(word_weights, 1)[0]
        word_tensor = torch.Tensor([[word_idx]]).long().cuda()
        history = torch.cat([history, word_tensor], 0)

        words.append(train_data.idx2word[word_idx])

    return '\n'.join(textwrap.wrap(' '.join(words),80))

for i in range(5):
    print('({})'.format(i), sample(), flush=True)