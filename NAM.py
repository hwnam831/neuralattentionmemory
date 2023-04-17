from unicodedata import bidirectional
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
            oldval = torch.einsum('lntc,lnt->ntc',tape, wpos)
            

            if self.noerase:
                newmem = torch.einsum('lnt,ntc->lntc',wpos,values[i]*rwe[:,:,1:2])
            elif self.rwprob:
                #newmem = torch.einsum('lnt,ntc->lntc',wpos,values[i]-oldval)
                newmem = torch.einsum('lnt,ntc->lntc',wpos,(values[i]*rwe[:,:,1:2]-oldval*rwe[:,:,2:3]))
            else:
                newmem = torch.einsum('lnt,ntc->lntc',wpos,(values[i]-oldval))
            tape = tape + newmem



            read_out = torch.einsum('lntc,lnt->ntc',tape,rpos)

            oldkey = torch.einsum('lntc,lnt->ntc',tape_key, wpos)
            if self.noerase:
                newkey = torch.einsum('lnt,ntc->lntc',wpos,keys[i])
            elif self.rwprob:
                #newkey = torch.einsum('lnt,ntc->lntc',wpos,(keys[i]-oldkey))
                newkey = torch.einsum('lnt,ntc->lntc',wpos,(keys[i]*rwe[:,:,1:2]-oldkey*rwe[:,:,2:3]))
            else:
                newkey = torch.einsum('lnt,ntc->lntc',wpos,(keys[i]-oldkey))

            
            if self.rwprob:
                #tape_key = tape_key + newkey*rw[None,:,:,1:2]
                tape_key = tape_key + newkey
            else:
                tape_key = tape_key + newkey

            jpos = torch.einsum('lntc,ntc->lnt',tape_key,queries[i])
            jpos = F.normalize(jpos,dim=0)
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

class NAMTuring2(nn.Module):
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
        self.controller = nn.LSTM(self.dim, self.dim//2,bidirectional=True)
        #self.controller = nn.LSTM(self.dim, self.dim,bidirectional=False)

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
            oldval = torch.einsum('lntc,lnt->ntc',tape, wpos)
            

            if self.noerase:
                newmem = torch.einsum('lnt,ntc->lntc',wpos,values[i]*rwe[:,:,1:2])
            elif self.rwprob:
                #newmem = torch.einsum('lnt,ntc->lntc',wpos,values[i]-oldval)
                newmem = torch.einsum('lnt,ntc->lntc',wpos,(values[i]*rwe[:,:,1:2]-oldval*rwe[:,:,2:3]))
            else:
                newmem = torch.einsum('lnt,ntc->lntc',wpos,(values[i]-oldval))
            tape = tape + newmem



            read_out = torch.einsum('lntc,lnt->ntc',tape,rpos)

            oldkey = torch.einsum('lntc,lnt->ntc',tape_key, wpos)
            if self.noerase:
                newkey = torch.einsum('lnt,ntc->lntc',wpos,keys[i])
            elif self.rwprob:
                #newkey = torch.einsum('lnt,ntc->lntc',wpos,(keys[i]-oldkey))
                newkey = torch.einsum('lnt,ntc->lntc',wpos,(keys[i]*rwe[:,:,1:2]-oldkey*rwe[:,:,2:3]))
            else:
                newkey = torch.einsum('lnt,ntc->lntc',wpos,(keys[i]-oldkey))

            
            if self.rwprob:
                #tape_key = tape_key + newkey*rw[None,:,:,1:2]
                tape_key = tape_key + newkey
            else:
                tape_key = tape_key + newkey

            jpos = torch.einsum('lntc,ntc->lnt',tape_key,queries[i])
            jpos = F.normalize(jpos,dim=0)
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

class NAMTuringRecurrent(nn.Module):
    def __init__(self,dim, n_tapes=4, default_tapelen=32, mem_size=64, rwprob=True, normalize_head=False):
        super().__init__()
        assert dim%n_tapes == 0
        self.head_dim=mem_size
        self.dim = dim
        self.n_tapes = n_tapes
        self.default_tapelen = default_tapelen
        self.rwprob = rwprob
        self.normalize_head = normalize_head
        # (prev, next, no-op, jump)
        #direction layers for read/write heads
        #action: (read_direction(4), write_direction(4), rwprob(2))
        #self.controller = nn.LSTM(self.dim, self.dim//2,bidirectional=True)
        self.controller = nn.LSTM(self.dim, self.dim,bidirectional=False)
        self.actionlayer = nn.Linear(self.dim+self.head_dim*self.n_tapes, 10*self.n_tapes)
        self.valuelayer = nn.Linear(self.dim, self.head_dim*self.n_tapes)
        self.keylayer = nn.Linear(self.dim, self.head_dim*self.n_tapes)
        self.outlayer = nn.Linear(self.head_dim*self.n_tapes,self.dim)

        self.Wq = nn.Linear(self.dim+self.head_dim*self.n_tapes, self.head_dim*self.n_tapes)
        
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
        #actions = self.actionlayer(hidden).reshape(seqlen,batchsize,self.n_tapes,10)
        #directions_r = F.softmax(actions[:,:,:,:4], dim=-1)
        #directions_w = F.softmax(actions[:,:,:,4:8], dim=-1)
        #(S,N,nh,2)
        #rwprobs = torch.sigmoid(actions[:,:,:,8:])
        #queries = self.Wq(hidden).reshape(seqlen,batchsize,self.n_tapes,self.head_dim)
        #queries = unitnorm(queries)
        read_outs = []
        read_out = torch.zeros_like(keys[0])
        for i in range(seqlen):
            # (N,nh,8)
            hidden_and_read = torch.concat((hidden[i],read_out.reshape(batchsize,-1)),dim=-1)
            action = self.actionlayer(hidden_and_read).reshape(batchsize,self.n_tapes,10)
            query = self.Wq(hidden_and_read).reshape(batchsize,self.n_tapes,self.head_dim)
            rw = torch.sigmoid(action[:,:,8:])
            read_dir=F.softmax(action[:,:,:4],dim=-1)
            write_dir=F.softmax(action[:,:,4:8],dim=-1)
            oldval = torch.einsum('lntc,lnt->ntc',tape, wpos)
            newmem = torch.einsum('lnt,ntc->lntc',wpos,values[i]-oldval)
            if self.rwprob:
                tape = tape + newmem*rw[None,:,:,1:2]
            else:
                tape = tape + newmem
            read_out = torch.einsum('lntc,lnt->ntc',tape,rpos)
            
            oldkey = torch.einsum('lntc,lnt->ntc',tape_key, wpos)
            newkey = torch.einsum('lnt,ntc->lntc',wpos,keys[i]-oldkey)
            if self.rwprob:
                tape_key = tape_key + newkey*rw[None,:,:,1:2]
            else:
                tape_key = tape_key + newkey
            
            jpos = torch.einsum('lntc,ntc->lnt',tape_key,query)
            jpos = F.normalize(jpos,dim=0)
            next_w = torch.roll(rpos, 1, dims=0)
            prev_w = torch.roll(rpos, -1, dims=0)
            wpos = prev_w*write_dir[None,:,:,0] + wpos*write_dir[None,:,:,1] +\
                    next_w*write_dir[None,:,:,2] + jpos*write_dir[None,:,:,3]
            if self.rwprob:
                read_out = read_out*rw[:,:,0:1]
            read_outs.append(read_out)
            next_r = torch.roll(rpos, 1, dims=0)
            prev_r = torch.roll(rpos, -1, dims=0)
            rpos = prev_r*read_dir[None,:,:,0] + rpos*read_dir[None,:,:,1] +\
                    next_r*read_dir[None,:,:,2] + jpos*read_dir[None,:,:,3]
            if self.normalize_head:
                wpos = F.normalize(wpos,dim=0)
                rpos = F.normalize(rpos,dim=0)
            
        outputs = torch.stack(read_outs).reshape(seqlen,batchsize,-1)
        outputs = self.outlayer(outputs)
        return outputs, tape.reshape(tapelen, batchsize, self.head_dim*self.n_tapes)

class NAMTuringOnlyJump(nn.Module):
    def __init__(self,dim, n_tapes=4, default_tapelen=32, mem_size=64):
        super().__init__()
        assert dim%n_tapes == 0
        self.head_dim=mem_size
        self.dim = dim
        self.n_tapes = n_tapes
        self.default_tapelen = default_tapelen
        # (no-op, jump)
        #direction layers for read/write heads
        #action: (read_direction(2), write_direction(2), rwprob(2))
        #self.controller = nn.LSTM(self.dim, self.dim//2,bidirectional=True)
        self.controller = nn.LSTM(self.dim, self.dim,bidirectional=False)
        self.actionlayer = nn.Linear(self.dim, 6*self.n_tapes)
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
        actions = self.actionlayer(hidden).reshape(seqlen,batchsize,self.n_tapes,6)
        directions_r = F.softmax(actions[:,:,:,:2], dim=-1)
        directions_w = F.softmax(actions[:,:,:,2:4], dim=-1)
        #(S,N,nh,2)
        rwprobs = torch.sigmoid(actions[:,:,:,4:])
        queries = self.Wq(hidden).reshape(seqlen,batchsize,self.n_tapes,self.head_dim)
        queries = unitnorm(queries)
        read_outs = []
        for i in range(seqlen):
            rw = rwprobs[i]
            read_dir=directions_r[i]
            write_dir=directions_w[i]
            oldval = torch.einsum('lntc,lnt->ntc',tape, wpos)
            newmem = torch.einsum('lnt,ntc->lntc',wpos,values[i]-oldval)
            tape = tape + newmem*rw[None,:,:,1:2]
            read_out = torch.einsum('lntc,lnt->ntc',tape,rpos)
            
            oldkey = torch.einsum('lntc,lnt->ntc',tape_key, wpos)
            newkey = torch.einsum('lnt,ntc->lntc',wpos,keys[i]-oldkey)
            tape_key = tape_key + newkey*rw[None,:,:,1:2]
            
            jpos = torch.einsum('lntc,ntc->lnt',tape_key,queries[i])
            jpos = F.normalize(jpos,dim=0)

            wpos = wpos*write_dir[None,:,:,0] + jpos*write_dir[None,:,:,1]
            
            read_outs.append(read_out*rw[:,:,0:1])
            rpos = rpos*read_dir[None,:,:,0] + jpos*read_dir[None,:,:,1]
            
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
        elif option=='namtm2':
            self.tm = NAMTuring2(dim, n_tapes=nhead, default_tapelen=defalt_tapelen, debug=debug, mem_size=mem_size)
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


#simple version, only input tape
class NAMTuringDecoder(nn.Module):
    def __init__(self,dim, n_tapes=4, default_tapelen=32, mem_size=64, debug=False):
        super().__init__()
        assert dim%n_tapes == 0
        self.head_dim=mem_size
        self.dim = dim
        self.n_tapes = n_tapes
        self.default_tapelen = default_tapelen
        self.debug=debug

        # (prev, next, no-op, jump)
        #direction layers for read/write heads

        #Input: read, x -> output: hidden
        #self.controller = nn.LSTMCell(self.dim + self.head_dim*self.n_tapes, self.dim)
        self.controller = nn.LSTMCell(self.dim, self.dim)
        self.Wk = nn.Linear(self.head_dim*self.n_tapes, self.head_dim*self.n_tapes)

        self.Wqkva = nn.Linear(self.dim, self.head_dim*self.n_tapes*3+11*self.n_tapes)


    def forward_step(self, next_tgt, tape, rpos, wpos, cntl_hidden):
        bsize = next_tgt.shape[0]
        
        #h,c = self.controller(torch.concat((next_tgt,rval),dim=-1), cntl_hidden)
        h,c = self.controller(next_tgt, cntl_hidden)
        qkva = self.Wqkva(h)
        query = qkva[:,:self.head_dim*self.n_tapes].reshape(bsize, self.n_tapes, -1)
        query = F.normalize(query,dim=-1)
        key = qkva[:,2*self.head_dim*self.n_tapes:self.head_dim*self.n_tapes].reshape(bsize, self.n_tapes, -1)
        key = F.normalize(key,dim=-1)
        value = qkva[:,2*self.head_dim*self.n_tapes:3*self.head_dim*self.n_tapes].reshape(bsize, self.n_tapes, -1)
        action = qkva[:,3*self.head_dim*self.n_tapes:].reshape(bsize, self.n_tapes, -1)
        read_dir = torch.softmax(action[:,:,:4], dim=-1)
        write_dir = torch.softmax(action[:,:,4:8], dim=-1)
        rwe = torch.sigmoid(action[:,:,8:])
        oldval = torch.einsum('lntc,lnt->ntc',tape, wpos)
        newmem = torch.einsum('lnt,ntc->lntc',wpos,(value*rwe[:,:,1:2]-oldval*rwe[:,:,2:3]))
        ntape = tape + newmem



        rval = torch.einsum('lntc,lnt->ntc',tape,rpos*rwe[:,:,0]).reshape(bsize,-1)
        jpos = torch.einsum('lntc,ntc->lnt',tape,query)
        jpos = F.normalize(jpos, dim=0)

        next_w = torch.roll(wpos, 1, dims=0)
        prev_w = torch.roll(wpos, -1, dims=0)
        nwpos = prev_w*write_dir[None,:,:,0] + wpos*write_dir[None,:,:,1] +\
                next_w*write_dir[None,:,:,2] + jpos*write_dir[None,:,:,3]
        next_r = torch.roll(rpos, 1, dims=0)
        prev_r = torch.roll(rpos, -1, dims=0)
        nrpos = prev_r*read_dir[None,:,:,0] + rpos*read_dir[None,:,:,1] +\
                next_r*read_dir[None,:,:,2] + jpos*read_dir[None,:,:,3]

        return rval, ntape, nrpos, nwpos, (h,c)
        
    def init_states(self, src):
        #seqlen = tgt.shape[0]
        batchsize = src.shape[1]

        tapelen_in = src.size(0)
        tape = src.reshape([tapelen_in, batchsize, self.n_tapes, self.head_dim])
        rpos = torch.zeros([tapelen_in, batchsize, self.n_tapes],
            dtype=src.dtype, device=src.device)
        rpos[0,:,:] = 1.0

        wpos = torch.zeros([tapelen_in, batchsize, self.n_tapes],
            dtype=src.dtype, device=src.device)
        wpos[0,:,:] = 1.0
        
        
        cntl_hidden = (torch.zeros([batchsize, self.dim],dtype=src.dtype, device=src.device),
                       torch.zeros([batchsize, self.dim],dtype=src.dtype, device=src.device))
        return tape, rpos, wpos, cntl_hidden
    #Seq-first in (S,N,C), seq-first out (S,N,C)
    #Stack: initial stack state (zeros or null embeddings)
    def forward(self, tgt, src):
        seqlen = tgt.shape[0]
        batchsize = tgt.shape[1]
        
        tape, rpos, wpos, cntl_hidden = \
            self.init_states(src)
        read_outs=[]
        for i in range(seqlen):
            rval, tape, rpos, wpos, cntl_hidden = \
                self.forward_step(tgt[i], tape, rpos, wpos, cntl_hidden)
            read_outs.append(rval)
        return torch.stack(read_outs,dim=0)
class NAMTuringDecoder2(nn.Module):
    def __init__(self,dim, n_tapes=4, default_tapelen=32, mem_size=64, debug=False):
        super().__init__()
        assert dim%n_tapes == 0
        self.head_dim=mem_size
        self.dim = dim
        self.n_tapes = n_tapes
        self.default_tapelen = default_tapelen
        self.debug=debug

        # (prev, next, no-op, jump)
        #direction layers for read/write heads

        #Input: read, x -> output: hidden
        self.controller = nn.LSTMCell(self.dim + self.head_dim*self.n_tapes, self.dim,bidirectional=False)

        self.Wk_in = nn.Linear(self.head_dim*self.n_tapes, self.head_dim*self.n_tapes)

        self.Wqkva = nn.Linear(self.dim, self.head_dim*self.n_tapes*4+11*self.n_tapes)


    def forward_step(self, next_tgt, tape_in, keytape_in, rpos, tape_out, keytape_out, wpos, cntl_hidden):
        bsize = next_tgt.shape[0]
        rval = torch.einsum('lntc,lnt->ntc',tape_in,rpos).resize(bsize,-1)
        hidden, (h,c) = self.controller(torch.concat((next_tgt,rval),dim=-1), cntl_hidden)
        qkva = self.Waqkv(hidden)
        q_in = qkva[:,:self.head_dim*self.n_tapes].reshape(bsize, self.n_tapes, -1)
        q_in = F.normalize(q_in,dim=-1)
        q_out = qkva[:,self.head_dim*self.n_tapes:2*self.head_dim*self.n_tapes].reshape(bsize, self.n_tapes, -1)
        q_out = F.normalize(q_out,dim=-1)
        key = qkva[:,2*self.head_dim*self.n_tapes:3*self.head_dim*self.n_tapes].reshape(bsize, self.n_tapes, -1)
        key = F.normalize(key,dim=-1)
        value = qkva[:,3*self.head_dim*self.n_tapes:4*self.head_dim*self.n_tapes].reshape(bsize, self.n_tapes, -1)
        action = qkva[:,4*self.head_dim*self.n_tapes:].reshape(bsize, self.n_tapes, -1)
        read_dir = torch.softmax(action[:,:,:4], dim=-1)
        write_dir = torch.softmax(action[:,:,4:8], dim=-1)
        rwe = torch.softmax(action[:,:,4:8], dim=-1)
        oldval = torch.einsum('lntc,lnt->ntc',tape_out, wpos)
        newmem = torch.einsum('lnt,ntc->lntc',wpos,(value*rwe[:,:,1:2]-oldval*rwe[:,:,2:3]))
        ntape_out = tape_out + newmem

        oldkey = torch.einsum('lntc,lnt->ntc',keytape_out, wpos)
        newkey = torch.einsum('lnt,ntc->lntc',wpos,key*rwe[:,:,1:2]-oldkey*rwe[:,:,2:3])
        nkeytape_out = keytape_out + newkey

        jpos_in = torch.einsum('lntc,ntc->lnt',keytape_in,q_in)
        jpos_in = F.normalize(jpos_in,dim=0)
        jpos_out = torch.einsum('lntc,ntc->lnt',keytape_out,q_out)
        jpos_out = F.normalize(jpos_out,dim=0)
        next_w = torch.roll(wpos, 1, dims=0)
        prev_w = torch.roll(wpos, -1, dims=0)
        nwpos = prev_w*write_dir[None,:,:,0] + wpos*write_dir[None,:,:,1] +\
                next_w*write_dir[None,:,:,2] + jpos_out*write_dir[None,:,:,3]
        next_r = torch.roll(rpos, 1, dims=0)
        prev_r = torch.roll(rpos, -1, dims=0)
        nrpos = prev_r*read_dir[None,:,:,0] + rpos*read_dir[None,:,:,1] +\
                next_r*read_dir[None,:,:,2] + jpos_in*read_dir[None,:,:,3]

        return nrpos, ntape_out, nkeytape_out, nwpos, (h,c)
        
    def init_tapes(self, src, tapelen=-1):
        #seqlen = tgt.shape[0]
        batchsize = src.shape[1]

        tapelen_in = src.size(0)
        tape_in = src.reshape([tapelen_in, batchsize, self.n_tapes, self.head_dim])
        rpos = torch.zeros([tapelen_in, batchsize, self.n_tapes],
            dtype=src.dtype, device=src.device)
        rpos[0,:,:] = 1.0

        tapelen_out = tapelen if tapelen > 0 else self.default_tapelen
        wpos = torch.zeros([tapelen_out, batchsize, self.n_tapes],
            dtype=src.dtype, device=src.device)
        wpos[0,:,:] = 1.0
        
        keytape_in = self.Wk_in(src).reshape([tapelen_in, batchsize, self.n_tapes, self.head_dim])
        #(L,N,T,C)
        tape_out = torch.zeros([tapelen_out, batchsize, self.n_tapes,self.head_dim],
                            dtype=src.dtype, device=src.device)
        keytape_out = torch.zeros_like(tape_out)
        
        
        cntl_hidden = (torch.zeros([batchsize, self.dim],dtype=src.dtype, device=src.device),
                       torch.zeros([batchsize, self.dim],dtype=src.dtype, device=src.device))
        return tape_in, rpos, keytape_in, tape_out, wpos, keytape_out, cntl_hidden
    #Seq-first in (S,N,C), seq-first out (S,N,C)
    #Stack: initial stack state (zeros or null embeddings)
    def forward(self, tgt, src, tapelen=-1):
        seqlen = tgt.shape[0]
        batchsize = tgt.shape[1]

        tapelen_out = tapelen if tapelen > 0 else self.default_tapelen
        
        tape_in, rpos, keytape_in, tape_out, wpos, keytape_out, cntl_hidden = \
            self.init_tapes(src, tapelen)
        for i in range(seqlen):
            rpos, tape_out, keytape_out, wpos, cntl_hidden = \
                self.forward_step(tgt[i],tape_in,keytape_in,rpos,tape_out,keytape_out,wpos,cntl_hidden)
        return tape_out.reshape(tapelen_out, batchsize, self.head_dim*self.n_tapes)


