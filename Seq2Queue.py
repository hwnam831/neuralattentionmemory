import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from Models import LSTMS2S, DNCS2S
import argparse
import Options
import AM
import NAM
from NSPDataset import NSPDatasetS2S, fib

VOCAB_SIZE=10

#100100011 -> 1111
class BinaryReductionDataset(Dataset):
    def __init__(self, maxdigits, mindigits=1, size=6400):
        assert maxdigits > mindigits
        self.maxdigits = maxdigits
        self.mindigits = mindigits
        self.size = size
        #vocab: 0,1,2(Pad)
        self.inputs = np.ones([size, self.maxdigits], dtype=np.int64)*VOCAB_SIZE
        self.targets = np.ones([size, self.maxdigits], dtype=np.int64)*VOCAB_SIZE
        self.iscreated = [False for i in range(size)]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if not self.iscreated[idx]:
            ndigits = ((self.maxdigits-self.mindigits+1)*idx)//self.size + self.mindigits
            seq = [np.random.randint(2)*np.random.randint(1,10) for _ in range(ndigits)]
            tidx = 0
            for s in seq:
                if s > 0:
                    self.targets[idx][tidx] = s
                    tidx = tidx+1
            pos = np.random.randint(self.maxdigits-ndigits+1)
            for i in range(pos, pos+ndigits):
                self.inputs[idx][i] = seq[i-pos]         
            self.iscreated[idx] = True
        return self.inputs[idx], self.targets[idx]


class TFEncoder(nn.Module):
    def __init__(self, model_size=512, nhead=2, num_layers=3):
        super().__init__()
        self.model_size=model_size
        self.enclayer = nn.TransformerEncoderLayer(d_model=model_size, nhead=nhead)
        self.encoder = nn.TransformerEncoder(self.enclayer, \
            num_layers=num_layers)
    #Seq-first in-out (S,N,C)
    def forward(self, src):
        memory = self.encoder(src)
        return memory

class TFDecoder(nn.Module):
    def __init__(self, model_size=512, tgt_vocab_size=16, nhead=2, num_layers=3):
        super().__init__()
        self.model_size=model_size
        self.declayer = nn.TransformerDecoderLayer(d_model=model_size, nhead=nhead)
        self.decoder = nn.TransformerDecoder(self.declayer, num_layers=num_layers)
        self.fc = nn.Linear(model_size, tgt_vocab_size)
    #Seq-first in (S,N,C), batch-first out (N,C,S)
    def forward(self, target, memory):
        tmask = self.generate_square_subsequent_mask(target.size(0)).to(target.device)
        out = self.decoder(target, memory, tgt_mask=tmask)
        return self.fc(out)
    def generate_square_subsequent_mask(self, sz):
        """Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

class TfS2S(nn.Module):
    def __init__(self, model_size=512, maxlen=256, nhead=4, num_layers=6, vocab_size = 256, tgt_vocab_size=10):
        super().__init__()
        self.model_size=model_size
        self.maxlen=maxlen
        self.embedding = nn.Embedding(vocab_size, model_size)
        self.tgt_emb = nn.Embedding(tgt_vocab_size, model_size)
            
        self.posembed = nn.Embedding(maxlen, model_size)
        self.encoder = TFEncoder(model_size, nhead=nhead, num_layers=num_layers)
        self.decoder = TFDecoder(model_size, tgt_vocab_size, nhead=nhead, num_layers=num_layers)
        
    #Batch-first in (N,S), batch-first out (N,C,S)
    def forward(self, input, target):
        input2 = input.permute(1,0)
        target2 = target.permute(1,0)
        ipos = torch.arange(input2.size(0), device=input.device)[:,None].expand(input2.shape)
        tpos = torch.arange(target2.size(0), device=target.device)[:,None].expand(target2.shape)

        src = self.embedding(input2) + self.posembed(ipos)
        tgt = self.tgt_emb(target2)+ self.posembed(tpos)
        memory = self.encoder(src)
        return self.decoder(tgt, memory).permute(1,2,0)

class Seq2Queue(nn.Module):
    def __init__(self, model_size, tgt_vocab_size=10):
        super().__init__()
        self.model_size=model_size
        # (push, no-op)
        self.actionlayer = nn.Sequential(nn.Linear(self.model_size, 2))
        self.encnorm = nn.LayerNorm(self.model_size)
        self.decoder = nn.Linear(self.model_size, tgt_vocab_size)
        self.encoder1 = nn.LSTM(self.model_size, self.model_size, num_layers=1, bidirectional=False)
        self.dropout = nn.Dropout(0.1)
    #Seq-first in (S,N,C), seq-first out (S,N,C)
    #Stack: initial stack state (zeros or null embeddings)
    def forward(self, memory, qsize):
        #(Q,N,C)
        queue = torch.zeros([qsize, memory.shape[1], memory.shape[-1]],
                            dtype=memory.dtype, device=memory.device)
        #(Q,N)
        posvec = torch.zeros([qsize, memory.shape[1]],
                            dtype=memory.dtype, device=memory.device)
        posvec[0,:] = 1.0
        #(S,N,2)
        actions = self.actionlayer(memory)
        if self.training :
            actions = F.softmax(actions, dim=-1)
        else:
            actions = F.gumbel_softmax(actions, hard=True, dim=-1)
        for i,action in enumerate(actions):
            newmem = torch.einsum('qn,nc->qnc',posvec,memory[i])
            queue = queue + newmem*action[None,:,1:2]
            nextpos = torch.roll(posvec, 1, dims=0)
            posvec = posvec*action[None,:,0] + nextpos*action[None,:,1]
        
        out, _ = self.encoder1(queue)
        out=self.dropout(out)
        out = self.encnorm(out)
        return self.decoder(out)

class NAMTuring(nn.Module):
    def __init__(self, model_size, tgt_vocab_size=10, default_tsize=32):
        super().__init__()
        self.default_tsize = default_tsize
        self.model_size=model_size
        # (push, no-op)
        self.directionlayer = nn.Linear(self.model_size, 3)
        self.rwlayer = nn.Sequential(nn.Linear(self.model_size, 2),
                                    nn.Sigmoid())
        self.decoder = nn.Sequential(nn.LayerNorm(self.model_size),
                                     nn.Dropout(0.1),
                                    nn.Linear(self.model_size, tgt_vocab_size))
    #Seq-first in (S,N,C), seq-first out (S,N,C)
    #Stack: initial stack state (zeros or null embeddings)
    def forward(self, values, tsize=-1):
        #(L,N,C)
        tsize = self.default_tsize if tsize <= 0 else tsize
        tape = torch.zeros([tsize, values.shape[1], values.shape[-1]],
                            dtype=values.dtype, device=values.device)
        #(L,N)
        posvec = torch.zeros([tsize, values.shape[1]],
                            dtype=values.dtype, device=values.device)
        posvec[0,:] = 1.0
        #(S,N,3) -> (L,N,R)
        directions = self.directionlayer(values)
        if self.training :
            directions = F.softmax(directions, dim=-1)
        else:
            directions = F.gumbel_softmax(directions, hard=True, dim=-1)
        #(S,N,2)
        rwprobs = self.rwlayer(values)
        for i,direction in enumerate(directions):
            rw = rwprobs[i]
            oldval = torch.einsum('lnc,ln->nc',tape, posvec)
            newmem = torch.einsum('ln,nc->lnc',posvec,values[i]-oldval)
            tape = tape + newmem*rw[None,:,1:2]
            nextpos = torch.roll(posvec, 1, dims=0)
            prevpos = torch.roll(posvec, -1, dims=0)
            posvec = prevpos*direction[None,:,0] + \
                      posvec*direction[None,:,1] + nextpos*direction[None,:,2]
        
        return self.decoder(tape)

class QueueReducer(nn.Module):
    def __init__(self, model_size=512, maxlen=256, max_tokens=64, vocab_size = 256, tgt_vocab_size=10, queue=Seq2Queue):
        super().__init__()
        self.model_size=model_size
        self.maxlen=maxlen
        self.max_tokens = max_tokens
        self.embedding = nn.Sequential(nn.Embedding(vocab_size, model_size),
                                        nn.Linear(model_size, model_size),
                                     nn.Dropout(0.2))
        self.rnnlayer = nn.LSTM(self.model_size, self.model_size//2, num_layers=1, bidirectional=True)
        
        self.encnorm = nn.LayerNorm(self.model_size)
        self.decoder = queue(model_size, tgt_vocab_size)

    #Batch-first in (N,S), batch-first out (N,C,S)
    def forward(self, input):
        input2 = input.permute(1,0)

        src = self.embedding(input2)
        src, _ = self.rnnlayer(src)
        src = self.encnorm(src)
        memory = src
        return self.decoder(memory, self.max_tokens).permute(1,2,0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "--lr",
            type=float,
            default='1e-4')
    parser.add_argument(
            "--epochs",
            type=int,
            default='100')
    parser.add_argument(
            "--batch_size",
            type=int,
            default='64')
    parser.add_argument(
            "--net",
            type=str,
            choices=['tf', 'lstm', 'nam', 'turing', 'dnc', 'lsam', 'test'],
            default='turing',
            help='network choices')
    parser.add_argument(
            "--task",
            type=str,
            choices=['fib','reduce'],
            default='reduce',
            help='task choices')
    args = parser.parse_args()
    if args.task == 'reduce':
        trainset = BinaryReductionDataset(12,2, size=25600)
        valset = BinaryReductionDataset(12,8, size=2048)
        testset = BinaryReductionDataset(20,12, size=2048)
        vocab_size = 11
    elif args.task == 'fib':
        trainset = NSPDatasetS2S(fib, 12,2, size=25600)
        valset = NSPDatasetS2S(fib, 12,8, size=2048)
        testset = NSPDatasetS2S(fib, 16,13, size=2048)
        vocab_size=16

    trainloader = DataLoader(trainset, batch_size=args.batch_size, num_workers=4, shuffle=True)
    valloader = DataLoader(valset, batch_size=args.batch_size, num_workers=4)
    testloader = DataLoader(testset, batch_size=args.batch_size, num_workers=4)
    if args.net == 'tf':
        model       = TfS2S(512, nhead=8, num_layers=2, tgt_vocab_size=vocab_size).cuda()
    elif args.net == 'lstm':
        model       = LSTMS2S(768, num_layers=2, vocab_size=256, tgt_vocab_size=vocab_size).cuda()
    elif args.net == 'turing':
        #model       = QueueReducer(512, maxlen=64, max_tokens=64, tgt_vocab_size=vocab_size, queue=NAMTuring).cuda()
        model = NAM.NAMTMS2Q(512, vocab_size).cuda()
    elif args.net == 'dnc':
        model       = DNCS2S(512, nr_cells = 32, vocab_size=vocab_size).cuda()
    elif args.net == 'lsam':
        model       = AM.LSAMS2S(512, nhead = 4, vocab_size=vocab_size).cuda()
    elif args.net == 'nam':
        model       = AM.LSAMAE(512, nhead = 4, vocab_size=vocab_size).cuda()
    else:
        model       = AM.LSAMS2S(512, nhead = 4, vocab_size=vocab_size).cuda()
    print(model)
    print("Parameter count: {}".format(Options.count_params(model)))
    optimizer   = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.lr/2)
    criterion   = nn.CrossEntropyLoss()
    model.train(mode=True)
    
    for e in range(args.epochs):
        tcorrect = 0
        tlen     = 0
        tloss    = 0
        for x,y in trainloader:
            xdata       = x.cuda()
            ydata       = y.cuda()
            optimizer.zero_grad()
            tgt         = torch.ones_like(ydata)*(vocab_size-1)
            tgt[:,1:]   = ydata[:,:-1]
            if args.net in ['nam','turing']:
                output      = model(xdata)[:,:,:ydata.shape[1]]
            else:
                output = model(xdata, tgt)
            loss        = criterion(output, ydata)
            loss.backward()
            tloss       = tloss + loss.item()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            pred        = output.argmax(axis=1)
            seqcorrect  = (pred==ydata).prod(-1)
            tcorrect    = tcorrect + seqcorrect.sum().item()
            tlen        = tlen + seqcorrect.nelement()
        print('Epoch {}:'.format(e+1))
        print('train seq acc:\t'+str(tcorrect/tlen))
        print('train loss:\t{}'.format(tloss/len(trainloader)))
        tcorrect = 0
        tlen     = 0
        tloss    = 0
        for x,y in valloader:
            xdata       = x.cuda()
            ydata       = y.cuda()
            with torch.no_grad():
                tgt         = torch.ones_like(ydata)*(vocab_size-1)
                tgt[:,1:]   = ydata[:,:-1]
                if args.net in ['nam','turing']:
                    output      = model(xdata)[:,:,:ydata.shape[1]]
                else:
                    output = model(xdata, tgt)
                loss        = criterion(output, ydata)
                tloss       = tloss + loss.item()
                
                pred        = output.argmax(axis=1)
                seqcorrect  = (pred==ydata).prod(-1)
                tcorrect    = tcorrect + seqcorrect.sum().item()
                tlen        = tlen + seqcorrect.nelement()

        print('val seq acc:\t'+str(tcorrect/tlen))
        print('val loss:\t{}\n'.format(tloss/len(valloader)))
        tcorrect = 0
        tlen     = 0
        tloss    = 0
        for x,y in testloader:
            xdata       = x.cuda()
            ydata       = y.cuda()
            with torch.no_grad():
                tgt         = torch.ones_like(ydata)*(vocab_size-1)
                tgt[:,1:]   = ydata[:,:-1]
                if args.net in ['nam','turing']:
                    output      = model(xdata)[:,:,:ydata.shape[1]]
                else:
                    output = model(xdata, tgt)
                loss        = criterion(output, ydata)
                tloss       = tloss + loss.item()
                
                pred        = output.argmax(axis=1)
                seqcorrect  = (pred==ydata).prod(-1)
                tcorrect    = tcorrect + seqcorrect.sum().item()
                tlen        = tlen + seqcorrect.nelement()

        print('test seq acc:\t'+str(tcorrect/tlen))
        print('test loss:\t{}\n'.format(tloss/len(testloader)))

            
    