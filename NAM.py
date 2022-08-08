from unicodedata import bidirectional
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import AM

def unitnorm(v):
    return F.normalize(v, dim=-1)

class NAMAttention(nn.Module):
    def __init__(self, d_model, nhead, gated=True):
        super().__init__()
        assert d_model % nhead == 0
        self.nhead = nhead
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)
        self.gated = gated
        if self.gated:
            self.Ww = nn.Linear(d_model, nhead)
            self.Wr = nn.Linear(d_model, nhead)

    #assuming (S,B,C) layout
    def forward(self, inp, src=None):
        
        if src is None:
            src = inp
        S,B = inp.size(0), inp.size(1)
        S2 = src.size(0)
        k = self.Wk(src).reshape(S2,B,self.nhead,-1) #(S1,B,n,Dk/n)
        v = self.Wv(src).reshape(S2,B,self.nhead,-1) #(S1,B,n,Dv/n)
        q = self.Wq(inp).reshape(S,B,self.nhead,-1) #(S2,B,n,Dq=Dk/n)
        k = unitnorm(k)
        q = unitnorm(q)
        if self.gated:
            w = torch.sigmoid(self.Ww(src)) #(S,B,n)
            A = torch.einsum('sbnq,sbnv->bnvq', k,v*w[:,:,:,None])
            r = torch.sigmoid(self.Wr(inp))
            out = torch.einsum('sbnq,bnvq->sbnv',q,A)*r[:,:,:,None]
        else:
            A = torch.einsum('sbnq,sbnv->bnvq', k,v)
            out = torch.einsum('sbnq,bnvq->sbnv',q,A)
        out = out.reshape(S,B,-1)
        out = self.Wo(out)
        return out, A

#Multi-head NAM Turing tape
class NAMTuring(nn.Module):
    def __init__(self,dim, n_tapes=4, default_tapelen=32):
        super().__init__()
        assert dim%n_tapes == 0
        self.head_dim=dim//n_tapes
        self.dim = dim
        self.n_tapes = n_tapes
        self.default_tapelen = default_tapelen
        # (prev, next, no-op)
        #direction layers for read/write heads
        #action: (read_direction(3), write_direction(3), rwprob(2))
        self.controller = nn.GRU(self.dim, self.dim//2,bidirectional=True)
        self.actionlayer = nn.Linear(self.dim, 8*self.n_tapes)
        self.valuelayer = nn.Linear(self.dim, self.head_dim*self.n_tapes)
        self.outlayer = nn.Linear(dim,dim)
        
    #Seq-first in (S,N,C), seq-first out (S,N,C)
    #Stack: initial stack state (zeros or null embeddings)
    def forward(self, inputs, tapelen=-1, tape_in=None, pos_in=None):
        seqlen = inputs.shape[0]
        batchsize = inputs.shape[1]

        values = self.valuelayer(inputs).reshape(seqlen, batchsize, self.n_tapes, self.head_dim)
        #(L,N,T,C)
        if tape_in is None:
            tapelen = tapelen if tapelen > 0 else self.default_tapelen
            tape = torch.zeros([tapelen, batchsize, self.n_tapes,self.head_dim],
                            dtype=inputs.dtype, device=inputs.device)
        else:
            tapelen = tape_in.size(0)
            tape = tape_in.reshape(tapelen, batchsize, self.n_tapes, self.head_dim)

        rpos = torch.zeros([tapelen, batchsize, self.n_tapes],
            dtype=inputs.dtype, device=inputs.device)
        wpos = torch.zeros([tapelen, batchsize, self.n_tapes],
            dtype=inputs.dtype, device=inputs.device)
        rpos[0,:,:] = 1.0
        wpos[0,:,:] = 1.0
       
        #(S,N,C) -> (S,N,nh,8)
        hidden, _ = self.controller(inputs)
        actions = self.actionlayer(hidden).reshape(seqlen,batchsize,self.n_tapes,8)
        directions_r = F.softmax(actions[:,:,:,:3], dim=-1)
        directions_w = F.softmax(actions[:,:,:,3:6], dim=-1)
        #(S,N,nh,2)
        rwprobs = torch.sigmoid(actions[:,:,:,6:])
        read_outs = []
        for i in range(seqlen):
            rw = rwprobs[i]
            read_dir=directions_r[i]
            write_dir=directions_w[i]
            oldval = torch.einsum('lntc,lnt->ntc',tape, wpos)
            newmem = torch.einsum('lnt,ntc->lntc',wpos,values[i]-oldval)
            tape = tape + newmem*rw[None,:,:,1:2]
            next_w = torch.roll(rpos, 1, dims=0)
            prev_w = torch.roll(rpos, -1, dims=0)
            wpos = prev_w*write_dir[None,:,:,0] + \
                      wpos*write_dir[None,:,:,1] + next_w*write_dir[None,:,:,2]
            read_out = torch.einsum('lntc,lnt->ntc',tape,rpos*rw[None,:,:,0])
            #read_outs.append(read_out + (1-rw)[None,:,:,:1]*values[i])
            read_outs.append(read_out)
            next_r = torch.roll(rpos, 1, dims=0)
            prev_r = torch.roll(rpos, -1, dims=0)
            rpos = prev_r*read_dir[None,:,:,0] + \
                      rpos*read_dir[None,:,:,1] + next_r*read_dir[None,:,:,2]
        outputs = torch.stack(read_outs).reshape(seqlen,batchsize,-1)
        outputs = self.outlayer(outputs)
        return outputs, tape.reshape(tapelen, batchsize, self.dim)

#TODO: attn-like jump?
class NAMTuringJump(nn.Module):
    def __init__(self,dim, n_tapes=4, default_tapelen=32):
        super().__init__()
        assert dim%n_tapes == 0
        self.head_dim=dim//n_tapes
        self.dim = dim
        self.n_tapes = n_tapes
        self.default_tapelen = default_tapelen
        # (prev, next, no-op, jump)
        #direction layers for read/write heads
        #action: (read_direction(4), write_direction(4), rwprob(2))
        self.controller = nn.GRU(self.dim, self.dim//2,bidirectional=True)
        self.actionlayer = nn.Linear(self.dim, 10*self.n_tapes)
        self.valuelayer = nn.Linear(self.dim, self.head_dim*self.n_tapes)
        self.outlayer = nn.Linear(dim,dim)
        self.Wk = nn.Linear(self.head_dim, self.head_dim)
        self.Wq = nn.Linear(self.dim, self.dim)
        
    #Seq-first in (S,N,C), seq-first out (S,N,C)
    #Stack: initial stack state (zeros or null embeddings)
    def forward(self, inputs, tapelen=-1, tape_in=None, pos_in=None):
        seqlen = inputs.shape[0]
        batchsize = inputs.shape[1]

        values = self.valuelayer(inputs).reshape(seqlen, batchsize, self.n_tapes, self.head_dim)
        #(L,N,T,C)
        if tape_in is None:
            tapelen = tapelen if tapelen > 0 else self.default_tapelen
            tape = torch.zeros([tapelen, batchsize, self.n_tapes,self.head_dim],
                            dtype=inputs.dtype, device=inputs.device)
        else:
            tapelen = tape_in.size(0)
            tape = tape_in.reshape(tapelen, batchsize, self.n_tapes, self.head_dim)

        rpos = torch.zeros([tapelen, batchsize, self.n_tapes],
            dtype=inputs.dtype, device=inputs.device)
        wpos = torch.zeros([tapelen, batchsize, self.n_tapes],
            dtype=inputs.dtype, device=inputs.device)
        rpos[0,:,:] = 1.0
        wpos[0,:,:] = 1.0
       
        #(S,N,C) -> (S,N,nh,8)
        hidden, _ = self.controller(inputs)
        actions = self.actionlayer(hidden).reshape(seqlen,batchsize,self.n_tapes,10)
        directions_r = F.softmax(actions[:,:,:,:4], dim=-1)
        directions_w = F.softmax(actions[:,:,:,4:8], dim=-1)
        #(S,N,nh,2)
        rwprobs = torch.sigmoid(actions[:,:,:,8:])
        queries = self.Wq(hidden).reshape(seqlen,batchsize,self.n_tapes,self.head_dim)
        read_outs = []
        for i in range(seqlen):
            rw = rwprobs[i]
            read_dir=directions_r[i]
            write_dir=directions_w[i]
            oldval = torch.einsum('lntc,lnt->ntc',tape, wpos)
            newmem = torch.einsum('lnt,ntc->lntc',wpos,values[i]-oldval)
            tape = tape + newmem*rw[None,:,:,1:2]
            read_out = torch.einsum('lntc,lnt->ntc',tape,rpos)
            
            keys = self.Wk(tape)
            jpos = torch.softmax(torch.einsum('lntc,ntc->lnt',keys,queries[i]),dim=0)
            next_w = torch.roll(rpos, 1, dims=0)
            prev_w = torch.roll(rpos, -1, dims=0)
            wpos = prev_w*write_dir[None,:,:,0] + wpos*write_dir[None,:,:,1] +\
                    next_w*write_dir[None,:,:,2] + jpos*write_dir[None,:,:,3]
            
            read_outs.append(read_out*rw[:,:,0:1])
            next_r = torch.roll(rpos, 1, dims=0)
            prev_r = torch.roll(rpos, -1, dims=0)
            rpos = prev_r*read_dir[None,:,:,0] + rpos*read_dir[None,:,:,1] +\
                    next_r*read_dir[None,:,:,2] + jpos*read_dir[None,:,:,3]
        outputs = torch.stack(read_outs).reshape(seqlen,batchsize,-1)
        outputs = self.outlayer(outputs)
        return outputs, tape.reshape(tapelen, batchsize, self.dim)

class NAMTMS2Q(nn.Module):
    def __init__(self, dim, vocab_size, nhead=4, defalt_tapelen=32):
        super().__init__()
        self.dim=dim
        self.embedding = nn.Sequential(nn.Embedding(vocab_size, dim))
        #self.rnnlayer = nn.LSTM(self.dim, self.dim//2, num_layers=1, bidirectional=True)
        #self.rnnlayer = AM.BLSAM(self.dim, self.dim, nhead=nhead)
        self.encnorm = nn.LayerNorm(self.dim)
        self.tm = NAMTuring(dim, n_tapes=nhead, default_tapelen=defalt_tapelen)
        self.fc = nn.Sequential(nn.LayerNorm(dim),
            nn.Dropout(0.1), nn.ReLU(),
            nn.Linear(dim, vocab_size))

    #Batch-first in (N,S), batch-first out (N,C,S)
    def forward(self, input):
        input2 = input.permute(1,0)

        src = self.embedding(input2)
        #src, _ = self.rnnlayer(src)
        src = self.encnorm(src)
        outputs, tape = self.tm(src)
        #S,N,C to N,C,S
        return self.fc(tape).permute(1,2,0)

class NAMTMAE(nn.Module):
    def __init__(self, dim, vocab_size, nhead=4, defalt_tapelen=32):
        super().__init__()
        self.dim=dim
        self.embedding = nn.Sequential(nn.Embedding(vocab_size, dim),
                                        nn.Linear(dim, dim),
                                     nn.Dropout(0.2))
        #self.rnnlayer = nn.LSTM(self.dim, self.dim//2, num_layers=1, bidirectional=True)
        #self.rnnlayer = AM.BLSAM(self.dim, self.dim, nhead=nhead)
        self.encnorm = nn.LayerNorm(self.dim)

        self.tm = NAMTuringJump(dim, n_tapes=nhead, default_tapelen=defalt_tapelen)
        self.tm2 = NAMTuringJump(dim, n_tapes=nhead, default_tapelen=defalt_tapelen)
        self.attn = NAMAttention(dim, nhead=nhead)
        self.fc = nn.Sequential(nn.LayerNorm(dim),
            nn.Dropout(0.1), nn.ReLU(),
            nn.Linear(dim, vocab_size))

    #Batch-first in (N,S), batch-first out (N,C,S)
    def forward(self, input):
        input2 = input.permute(1,0)

        src = self.embedding(input2)
        #src, _ = self.rnnlayer(src)
        src = self.encnorm(src)
        outputs, tape = self.tm(src)
        outputs, tape = self.tm2(src, tape_in=tape)

        #res, _ = self.attn(src, tape)
        #outputs = res + src

        #S,N,C to N,C,S
        return self.fc(outputs).permute(1,2,0)