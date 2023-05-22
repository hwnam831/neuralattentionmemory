import torch
import torch.nn as nn
from torch.nn import functional as F
import math
def unitnorm(v):
    return F.normalize(v, dim=-1)

#Multi-head NAM Turing tape
class NAMTuringNoJump(nn.Module):
    def __init__(self,dim, n_tapes=4, default_tapelen=32, mem_size=64):
        super().__init__()
        assert dim%n_tapes == 0
        self.head_dim=mem_size
        self.dim = dim
        self.n_tapes = n_tapes
        self.default_tapelen = default_tapelen
        # (prev, next, no-op)
        #direction layers for read/write heads
        #action: (read_direction(3), write_direction(3), rwprob(2))
        self.controller = nn.LSTM(self.dim, self.dim,bidirectional=False)

        self.actionlayer = nn.Linear(self.dim, 9*self.n_tapes)
        self.valuelayer = nn.Linear(self.dim, self.head_dim*self.n_tapes)

        self.outlayer = nn.Linear(self.head_dim*self.n_tapes,dim)
        
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
        actions = self.actionlayer(hidden).reshape(seqlen,batchsize,self.n_tapes,9)
        directions_r = F.softmax(actions[:,:,:,:3], dim=-1)
        directions_w = F.softmax(actions[:,:,:,3:6], dim=-1)
        #(S,N,nh,2)
        rweprobs = torch.sigmoid(actions[:,:,:,6:])
        read_outs = []
        for i in range(seqlen):
            rwe = rweprobs[i]
            read_dir=directions_r[i]
            write_dir=directions_w[i]
            oldval = torch.einsum('lntc,lnt->ntc',tape, wpos)
            newmem = torch.einsum('lnt,ntc->lntc',wpos,values[i]*rwe[:,:,1:2]-oldval*rwe[:,:,2:3])
            tape = tape + newmem
            next_w = torch.roll(wpos, 1, dims=0)
            prev_w = torch.roll(wpos, -1, dims=0)
            wpos = prev_w*write_dir[None,:,:,0] + \
                      wpos*write_dir[None,:,:,1] + next_w*write_dir[None,:,:,2]
            read_out = torch.einsum('lntc,lnt->ntc',tape,rpos*rwe[None,:,:,0])

            read_outs.append(read_out)
            next_r = torch.roll(rpos, 1, dims=0)
            prev_r = torch.roll(rpos, -1, dims=0)
            rpos = prev_r*read_dir[None,:,:,0] + \
                      rpos*read_dir[None,:,:,1] + next_r*read_dir[None,:,:,2]
        outputs = torch.stack(read_outs).reshape(seqlen,batchsize,-1)
        outputs = self.outlayer(outputs)
        return outputs, tape.reshape(tapelen, batchsize, self.dim)

class NAMTuring(nn.Module):
    def __init__(self,dim, n_tapes=4, default_tapelen=32, mem_size=64, rwprob=True, noerase=False, debug=False):
        super().__init__()
        assert dim%n_tapes == 0
        self.head_dim=mem_size
        self.dim = dim
        self.n_tapes = n_tapes
        self.default_tapelen = default_tapelen
        self.rwprob = rwprob
        self.noerase = noerase
        self.debug=debug

        # (prev, next, no-op, jump)
        #direction layers for read/write heads
        #action: (read_direction(4), write_direction(4), rweprob(3))
        #self.controller = nn.LSTM(self.dim, self.dim//2,bidirectional=True)
        self.controller = nn.LSTM(self.dim, self.dim,bidirectional=False)

        self.actionlayer = nn.Linear(self.dim, 11*self.n_tapes)
        self.valuelayer = nn.Linear(self.dim, self.head_dim*self.n_tapes)
        self.keylayer = nn.Linear(self.dim, self.head_dim*self.n_tapes)
        self.outlayer = nn.Linear(self.head_dim*self.n_tapes,dim)

        self.Wq = nn.Linear(self.dim, self.head_dim*self.n_tapes)
        
    #Seq-first in (S,N,C), seq-first out (S,N,C)
    #Stack: initial stack state (zeros or null embeddings)
    def forward(self, inputs, tapelen=-1, tape_in=None, pos_in=None):
        seqlen = inputs.shape[0]
        batchsize = inputs.shape[1]

        values = self.valuelayer(inputs).reshape(seqlen, batchsize, self.n_tapes, self.head_dim)

        
        keys = self.keylayer(inputs).reshape(seqlen, batchsize, self.n_tapes, self.head_dim)
        keys = unitnorm(keys)
        #(L,N,T,C)
        if tape_in is None:
            tapelen = tapelen if tapelen > 0 else self.default_tapelen
            tape = torch.zeros([tapelen, batchsize, self.n_tapes,self.head_dim],
                            dtype=inputs.dtype, device=inputs.device)
        else:
            tapelen = tape_in.size(0)
            tape = tape_in.reshape(tapelen, batchsize, self.n_tapes, self.head_dim)
        tape_key = torch.zeros_like(tape)
        rpos = torch.zeros([tapelen, batchsize, self.n_tapes],
            dtype=inputs.dtype, device=inputs.device)
        wpos = torch.zeros([tapelen, batchsize, self.n_tapes],
            dtype=inputs.dtype, device=inputs.device)
        rpos[0,:,:] = 1.0
        wpos[0,:,:] = 1.0
       
        #(S,N,C) -> (S,N,nh,8)
        hidden, _ = self.controller(inputs)

        actions = self.actionlayer(hidden).reshape(seqlen,batchsize,self.n_tapes,11)
        directions_r = F.softmax(actions[:,:,:,:4], dim=-1)
        directions_w = F.softmax(actions[:,:,:,4:8], dim=-1)
        #(S,N,nh,3)
        rweprobs = torch.sigmoid(actions[:,:,:,8:])
        queries = self.Wq(hidden).reshape(seqlen,batchsize,self.n_tapes,self.head_dim)
        queries = unitnorm(queries)
        read_outs = []
        for i in range(seqlen):
            rwe = rweprobs[i]
            read_dir=directions_r[i]
            write_dir=directions_w[i]
            nrpos = F.normalize(rpos, dim=0)
            nwpos = F.normalize(wpos, dim=0)
            oldval = torch.einsum('lntc,lnt->ntc',tape, nwpos)
            

            if self.noerase:
                newmem = torch.einsum('lnt,ntc->lntc',nwpos,values[i]*rwe[:,:,1:2])
            elif self.rwprob:
                #newmem = torch.einsum('lnt,ntc->lntc',wpos,values[i]-oldval)
                newmem = torch.einsum('lnt,ntc->lntc',nwpos,(values[i]*rwe[:,:,1:2]-oldval*rwe[:,:,2:3]))
            else:
                newmem = torch.einsum('lnt,ntc->lntc',nwpos,(values[i]-oldval))
            tape = tape + newmem



            read_out = torch.einsum('lntc,lnt->ntc',tape,nrpos)

            oldpos = torch.einsum('lntc,ntc->lnt',tape_key, keys[i])
            if self.noerase:
                newkey = torch.einsum('lnt,ntc->lntc',wpos,keys[i])
            elif self.rwprob:
                #newkey = torch.einsum('lnt,ntc->lntc',wpos,(keys[i]-oldkey))
                newkey = torch.einsum('lnt,ntc->lntc',(wpos*rwe[:,:,1]-oldpos*rwe[:,:,2]),keys[i])
            else:
                newkey = torch.einsum('lnt,ntc->lntc',(wpos-oldpos),keys[i])

            
            if self.rwprob:
                #tape_key = tape_key + newkey*rw[None,:,:,1:2]
                tape_key = tape_key + newkey
            else:
                tape_key = tape_key + newkey

            jpos = torch.einsum('lntc,ntc->lnt',tape_key,queries[i])
            #jpos = F.normalize(jpos,dim=0)
            next_w = torch.roll(wpos, 1, dims=0)
            prev_w = torch.roll(wpos, -1, dims=0)
            wpos = prev_w*write_dir[None,:,:,0] + wpos*write_dir[None,:,:,1] +\
                    next_w*write_dir[None,:,:,2] + jpos*write_dir[None,:,:,3]
            #wpos = F.normalize(wpos,dim=0)
            if self.rwprob:
                read_outs.append(read_out*rwe[:,:,0:1])
            else:
                read_outs.append(read_out)
            next_r = torch.roll(rpos, 1, dims=0)
            prev_r = torch.roll(rpos, -1, dims=0)
            rpos = prev_r*read_dir[None,:,:,0] + rpos*read_dir[None,:,:,1] +\
                    next_r*read_dir[None,:,:,2] + jpos*read_dir[None,:,:,3]
            #rpos = F.normalize(rpos,dim=0)
            
        outputs = torch.stack(read_outs).reshape(seqlen,batchsize,-1)
        outputs = self.outlayer(outputs)
        return outputs, tape.reshape(tapelen, batchsize, self.head_dim*self.n_tapes)

class NAMTuringOnlyJump(nn.Module):
    def __init__(self,dim, n_tapes=4, default_tapelen=32, mem_size=64, rwprob=True, noerase=False, debug=False):
        super().__init__()
        assert dim%n_tapes == 0
        self.head_dim=mem_size
        self.dim = dim
        self.n_tapes = n_tapes
        self.default_tapelen = default_tapelen
        self.rwprob = rwprob
        self.noerase = noerase
        self.debug=debug

        # (prev, next, no-op, jump)
        #direction layers for read/write heads
        #action: (read_direction(4), write_direction(4), rweprob(3))
        #self.controller = nn.LSTM(self.dim, self.dim//2,bidirectional=True)
        self.controller = nn.LSTM(self.dim, self.dim,bidirectional=False)

        self.actionlayer = nn.Linear(self.dim, 7*self.n_tapes)
        self.valuelayer = nn.Linear(self.dim, self.head_dim*self.n_tapes)
        self.keylayer = nn.Linear(self.dim, self.head_dim*self.n_tapes)
        self.outlayer = nn.Linear(self.head_dim*self.n_tapes,dim)

        self.Wq = nn.Linear(self.dim, self.head_dim*self.n_tapes)
        
    #Seq-first in (S,N,C), seq-first out (S,N,C)
    #Stack: initial stack state (zeros or null embeddings)
    def forward(self, inputs, tapelen=-1, tape_in=None, pos_in=None):
        seqlen = inputs.shape[0]
        batchsize = inputs.shape[1]

        values = self.valuelayer(inputs).reshape(seqlen, batchsize, self.n_tapes, self.head_dim)

        
        keys = self.keylayer(inputs).reshape(seqlen, batchsize, self.n_tapes, self.head_dim)
        keys = unitnorm(keys)
        #(L,N,T,C)
        if tape_in is None:
            tapelen = tapelen if tapelen > 0 else self.default_tapelen
            tape = torch.zeros([tapelen, batchsize, self.n_tapes,self.head_dim],
                            dtype=inputs.dtype, device=inputs.device)
        else:
            tapelen = tape_in.size(0)
            tape = tape_in.reshape(tapelen, batchsize, self.n_tapes, self.head_dim)
        tape_key = torch.zeros_like(tape)
        rpos = torch.zeros([tapelen, batchsize, self.n_tapes],
            dtype=inputs.dtype, device=inputs.device)
        wpos = torch.zeros([tapelen, batchsize, self.n_tapes],
            dtype=inputs.dtype, device=inputs.device)
        rpos[0,:,:] = 1.0
        wpos[0,:,:] = 1.0
       
        #(S,N,C) -> (S,N,nh,8)
        hidden, _ = self.controller(inputs)

        actions = self.actionlayer(hidden).reshape(seqlen,batchsize,self.n_tapes,7)
        directions_r = F.softmax(actions[:,:,:,:2], dim=-1)
        directions_w = F.softmax(actions[:,:,:,2:4], dim=-1)
        #(S,N,nh,3)
        rweprobs = torch.sigmoid(actions[:,:,:,4:])
        queries = self.Wq(hidden).reshape(seqlen,batchsize,self.n_tapes,self.head_dim)
        queries = unitnorm(queries)
        read_outs = []
        for i in range(seqlen):
            rwe = rweprobs[i]
            read_dir=directions_r[i]
            write_dir=directions_w[i]
            nrpos = F.normalize(rpos, dim=0)
            nwpos = F.normalize(wpos, dim=0)
            oldval = torch.einsum('lntc,lnt->ntc',tape, nwpos)
            

            if self.noerase:
                newmem = torch.einsum('lnt,ntc->lntc',nwpos,values[i]*rwe[:,:,1:2])
            elif self.rwprob:
                #newmem = torch.einsum('lnt,ntc->lntc',wpos,values[i]-oldval)
                newmem = torch.einsum('lnt,ntc->lntc',nwpos,(values[i]*rwe[:,:,1:2]-oldval*rwe[:,:,2:3]))
            else:
                newmem = torch.einsum('lnt,ntc->lntc',nwpos,(values[i]-oldval))
            tape = tape + newmem



            read_out = torch.einsum('lntc,lnt->ntc',tape,nrpos)

            oldpos = torch.einsum('lntc,ntc->lnt',tape_key, keys[i])
            if self.noerase:
                newkey = torch.einsum('lnt,ntc->lntc',wpos,keys[i])
            elif self.rwprob:
                #newkey = torch.einsum('lnt,ntc->lntc',wpos,(keys[i]-oldkey))
                newkey = torch.einsum('lnt,ntc->lntc',(wpos*rwe[:,:,1]-oldpos*rwe[:,:,2]),keys[i])
            else:
                newkey = torch.einsum('lnt,ntc->lntc',(wpos-oldpos),keys[i])

            
            if self.rwprob:
                #tape_key = tape_key + newkey*rw[None,:,:,1:2]
                tape_key = tape_key + newkey
            else:
                tape_key = tape_key + newkey

            jpos = torch.einsum('lntc,ntc->lnt',tape_key,queries[i])
            #jpos = F.normalize(jpos,dim=0)

            wpos = wpos*write_dir[None,:,:,0] + jpos*write_dir[None,:,:,1]
            #wpos = F.normalize(wpos,dim=0)
            if self.rwprob:
                read_outs.append(read_out*rwe[:,:,0:1])
            else:
                read_outs.append(read_out)

            rpos = rpos*read_dir[None,:,:,0] + jpos*read_dir[None,:,:,1]
            #rpos = F.normalize(rpos,dim=0)
            
        outputs = torch.stack(read_outs).reshape(seqlen,batchsize,-1)
        outputs = self.outlayer(outputs)
        return outputs, tape.reshape(tapelen, batchsize, self.head_dim*self.n_tapes)

class NAMTMAE(nn.Module):
    def __init__(self, dim, vocab_size, nhead=4, defalt_tapelen=32, option='default', debug=False, mem_size=64):
        super().__init__()
        self.dim=dim
        self.embedding = nn.Sequential(nn.Embedding(vocab_size, dim),
                                        nn.Linear(dim, dim),
                                     nn.Dropout(0.2))

        self.encnorm = nn.LayerNorm(self.dim)
        if option=='nojump':
            self.tm = NAMTuringNoJump(dim, n_tapes=nhead, mem_size=mem_size, default_tapelen=defalt_tapelen)
        elif option=='onlyjump':
            self.tm = NAMTuringOnlyJump(dim, n_tapes=nhead, mem_size=mem_size, default_tapelen=defalt_tapelen)
        elif option=='norwprob':
            self.tm = NAMTuring(dim, n_tapes=nhead, default_tapelen=defalt_tapelen, mem_size=mem_size, rwprob=False)
        elif option=='noerase':
            self.tm = NAMTuring(dim, n_tapes=nhead, default_tapelen=defalt_tapelen, mem_size=mem_size, noerase=True)
        else:
            self.tm = NAMTuring(dim, n_tapes=nhead, default_tapelen=defalt_tapelen, mem_size=mem_size)
        #self.tm2 = NAMTuring(dim, n_tapes=nhead, default_tapelen=defalt_tapelen)
        self.fc = nn.Sequential(nn.LayerNorm(dim),
            nn.Dropout(0.1), nn.ReLU(),
            nn.Linear(dim, vocab_size))

    #Batch-first in (N,S), batch-first out (N,C,S)
    def forward(self, input):
        input2 = input.permute(1,0)

        src = self.embedding(input2)

        src = self.encnorm(src)
        outputs, tape = self.tm(src)
        #outputs, tape = self.tm2(src, tape_in=tape)

        #S,N,C to N,C,S
        return self.fc(outputs).permute(1,2,0)


