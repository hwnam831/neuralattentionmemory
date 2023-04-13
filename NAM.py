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



#TODO: Gated AM? Feed AM to next layer?
class LSAMCell(nn.Module):
    def __init__(self, input_dim, d_model, nhead, sigma=unitnorm):
        super().__init__()
        assert d_model % nhead == 0
        self.nhead = nhead
        self.input_dim = input_dim
        self.Wqkv = nn.Linear(d_model+input_dim, d_model*3) # x:h -> q:k:v
        self.Ww = nn.Linear(d_model+input_dim, nhead)
        self.Wr = nn.Linear(d_model+input_dim, nhead)
        self.sigma = sigma
        self.d_head = d_model//nhead
        self.d_model = d_model

    def forward(self, x, h, AM):
        #assuming (B,C) layout
        B = x.size(0)
        
        xh = torch.cat((x,h), dim=-1)
        qkv = self.Wqkv(xh).reshape(B,3,self.nhead,-1)
        q = self.sigma(qkv[:,0])
        k = self.sigma(qkv[:,1])
        v = qkv[:,2]

        #RW probability (gate) per head : [B,n]
        w = torch.sigmoid(self.Ww(xh))
        
        #Memory to override. kvT - kv_rT = k(v-v_r)T
        v_r = torch.einsum('bnvq,bnq->bnv', AM,k)
        vp = w[:,:,None]*(v-v_r)
        A_w = torch.einsum('bnq,bnv->bnvq', k,vp)

        #update AM using write gates
        AM = AM + A_w
        
        #gated read
        r = torch.sigmoid(self.Wr(xh))
        h = torch.einsum('bnvq,bnq->bnv', AM,q)*r[:,:,None]
        h = h.reshape(B,-1)

        return h, AM

class LSAMDecoder(nn.Module):
    def __init__(self, input_dim, d_model, nhead, sigma=unitnorm, activation = nn.ReLU(), drop=0.1):
        super().__init__()
        assert d_model % nhead == 0
        self.nhead = nhead
        self.encoder = LSAMCell(input_dim=input_dim, d_model=d_model, nhead=nhead)
        
        self.sigma = sigma
        self.d_head = d_model//nhead
        self.d_model = d_model
        self.Wo = nn.Sequential(nn.Linear(d_model, d_model),
                                nn.Dropout(drop), activation)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, AM):
        #assuming (S,B,C) layout
        B = x.size(1)
        h = torch.zeros(B, self.d_model, dtype=x.dtype, device=x.device)
        out = []
        for x_i in x:
            h, AM = self.encoder(x_i, h, AM)
            out.append(h)
        out = torch.stack(out)
        out = self.Wo(out)
        return self.norm(out), (h,AM)

class BLSAM(nn.Module):
    def __init__(self, input_dim, d_model, nhead, sigma=unitnorm, activation = nn.ReLU(), drop=0.1):
        super().__init__()
        assert d_model % nhead == 0
        assert d_model % 2 == 0
        assert nhead % 2 == 0
        self.nhead = nhead
        self.enc_fw = LSAMCell(input_dim=input_dim, d_model=d_model//2, nhead=nhead//2)
        self.enc_bw = LSAMCell(input_dim=input_dim, d_model=d_model//2, nhead=nhead//2)
        self.input_dim = input_dim
        self.sigma = sigma
        self.d_head = d_model//nhead
        self.d_model = d_model
        self.Wo = nn.Sequential(nn.Linear(d_model, d_model),
                                nn.Dropout(drop), activation)
        self.norm = nn.LayerNorm(d_model)
    def forward(self, x, hA=None):
        #assuming (S,B,C) layout
        B = x.size(1)
        if hA is None:
            h_f = torch.zeros([B, self.d_model//2],
                        dtype=x.dtype, device=x.device)
            h_b = torch.zeros([B, self.d_model//2],
                        dtype=x.dtype, device=x.device)
            #B,N,D/N,D/N
            AM_f = torch.zeros([B, self.nhead//2, self.d_head, self.d_head],
                        dtype=x.dtype, device=x.device) 
            AM_b = torch.zeros([B, self.nhead//2, self.d_head, self.d_head],
                        dtype=x.dtype, device=x.device) 
        else:
            h,AM = hA
            h_f = h[:,:self.d_model//2]
            h_b = h[:,self.d_model//2:]
            AM_f = AM[:,:self.nhead//2]
            AM_f = AM[:,self.nhead//2:]

        out_f = []
        out_b = []
        for i in range(len(x)):
            x_f = x[i]
            h_f, AM_f = self.enc_fw(x_f, h_f, AM_f)
            out_f.append(h_f)
            x_b = x[-i-1]
            h_b, AM_b = self.enc_bw(x_b, h_b, AM_b)
            out_b.append(h_b)
        out_b.reverse()
        out = torch.concat((torch.stack(out_f), torch.stack(out_b)),dim=-1)
        out = self.Wo(out)

        #Concat at channel
        h = torch.concat((h_f, h_b), dim=-1)
        #Concat at head
        AM = torch.concat((AM_f, AM_b), dim=1)
        return out, (h,AM)

class LSAMEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.attn = BLSAM(input_dim=d_model, d_model=d_model, nhead=nhead)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    def forward(self, src):
        src2, kq = self.attn(src)
        src = src + self.dropout1(src2)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, kq
#TODO: LSAM decoder
class LSAMAE(nn.Module):
    def __init__(self, d_model=256, nhead=4, num_layers=2, vocab_size=16):
        super().__init__()
        self.d_model=d_model
        self.vocab_size = vocab_size
        assert d_model%2 == 0
        self.embedding = nn.Embedding(vocab_size, d_model)

        self.encoder = LSAMEncoderLayer(d_model=d_model, nhead=nhead)

        self.decoder = LSAMDecoder(input_dim=d_model, d_model=d_model, nhead=nhead)
        self.fc = nn.Linear(d_model, vocab_size)

    #Batch-first in (N,S), batch-first out (N,C,S)
    def forward(self, input, encoder_mode=False):
        input2 = input.permute(1,0)

        src = self.embedding(input2)

        out, hA = self.encoder(src)
        out, hA = self.decoder(out, hA[1])

        out = self.fc(out).permute(1,2,0)
        if encoder_mode:
            return out, hA[1]
        else:
            return out
