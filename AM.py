from http.client import responses
from ssl import RAND_pseudo_bytes
from unicodedata import bidirectional
import torch
import torch.nn as nn
from torch.nn import functional as F
import math

def unitnorm(v):
    return F.normalize(v, dim=-1)

def unitrelu(v):
    vec = torch.relu(v)
    eps = 1e-6
    return vec/(torch.norm(vec,dim=-1)+eps).unsqueeze(-1)
def unitelu(v):
    vec = v/torch.norm(v,dim=-1).unsqueeze(-1)
    return F.elu(vec)

def eluunit(v):
    vec = F.elu(v)
    return vec/torch.norm(vec,dim=-1).unsqueeze(-1)

def unitsq(v):
    return v/torch.sqrt(torch.norm(v,dim=-1)).unsqueeze(-1)

def unitmn(v):
    mn = torch.mean(torch.norm(v,dim=-1),dim=0)
    return v/mn.unsqueeze(-1)

def softmaxnorm(v):
    sqrtd = 1/math.sqrt(v.size(-1))
    return F.softmax(v*sqrtd,dim=-1)

def tanhnorm(v):
    return torch.tanh(v)/math.sqrt(v.size(-1))


class AttentionalMemory(nn.Module):
    def __init__(self, d_model, nhead, sigma=unitnorm):
        super().__init__()
        assert d_model % nhead == 0
        self.nhead = nhead
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)

        self.Wo = nn.Linear(d_model, d_model)
        self.sigma = sigma
        #self.sigma = nn.LayerNorm(d_model//nhead)
    def forward(self, h):
        #assuming (S,B,C) layout
        S,B = h.size(0), h.size(1)
        k = self.Wk(h).reshape(S,B,self.nhead,-1) #(S,B,n,Dk/n)
        v = self.Wv(h).reshape(S,B,self.nhead,-1) #(S,B,n,Dv/n)
        k = self.sigma(k)
        #k = unitelu(k)
        A = torch.einsum('sbnq,sbnv->bnvq', k,v)
        q = self.Wq(h).reshape(S,B,self.nhead,-1) #(S,B,n,Dq=Dk/n)
        q = self.sigma(q)
        out = torch.einsum('sbnq,bnvq->sbnv',q,A).reshape(S,B,-1)
        out = self.Wo(out)
        return out, (k,q)

class GatedAM(nn.Module):
    def __init__(self, d_model, nhead, sigma=unitnorm):
        super().__init__()
        assert d_model % nhead == 0
        self.nhead = nhead
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)
        self.Ww = nn.Linear(d_model, nhead)
        self.Wr = nn.Linear(d_model, nhead)
        self.sigma = sigma
        #self.sigma = nn.LayerNorm(d_model//nhead)
    def forward(self, h):
        #assuming (S,B,C) layout
        S,B = h.size(0), h.size(1)
        
        k = self.Wk(h).reshape(S,B,self.nhead,-1) #(S,B,n,Dk/n)
        v = self.Wv(h).reshape(S,B,self.nhead,-1) #(S,B,n,Dv/n)
        k = self.sigma(k)
        #k = unitelu(k)
        w = torch.sigmoid(self.Ww(h)) #(S,B,n)

        A = torch.einsum('sbnq,sbnv->bnvq', k,v*w[:,:,:,None])
        q = self.Wq(h).reshape(S,B,self.nhead,-1) #(S,B,n,Dq=Dk/n)
        q = self.sigma(q)
        
        r = torch.sigmoid(self.Wr(h))
        out = torch.einsum('sbnq,bnvq->sbnv',q,A)*r[:,:,:,None]
        out = out.reshape(S,B,-1)
        out = self.Wo(out)
        return out, (k,q)

class LinearAttention(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)
        self.nhead = nhead
    def forward(self, h):
        #assuming (S,N,C) layout
        S,B = h.size(0), h.size(1)
        k = self.Wk(h).reshape(S,B,self.nhead,-1) #(S,B,n,Dk/n)
        v = self.Wv(h).reshape(S,B,self.nhead,-1) #(S,B,n,Dv/n)
        k = F.elu(k)+1
        #k = unitelu(k)
        ksum = k.sum(dim=0) #(B,n, Dk)
        A = torch.einsum('sbnq,sbnv->bnvq', k,v)
        q = self.Wq(h).reshape(S,B,self.nhead,-1) #(S,B,n,Dq=Dk/n)
        q = F.elu(q)+1
        #q = unitnorm(q)
        Z = 1/torch.einsum('sbnq,bnq->sbn',q,ksum)
        out = torch.einsum('sbnq,bnvq,sbn->sbnv',q,A,Z).reshape(S,B,-1)
        out = self.Wo(out)
        return out, (k,q)

class AMEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, attn=AttentionalMemory):
        super().__init__()
        self.attn = attn(d_model, nhead)
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

class AMEncoder(nn.Module):
    def __init__(self, d_model=512, nhead=4, num_layers=6, maxlen=512, vocab_size=16, attn=AttentionalMemory):
        super().__init__()
        self.d_model=d_model
        self.maxlen=maxlen
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.posembed = nn.Embedding(maxlen, d_model)
        self.encoder = nn.ModuleList([
            AMEncoderLayer(d_model=d_model, nhead=nhead, attn=attn) for _ in range(num_layers)
        ])
        self.fc = nn.Linear(d_model, vocab_size)

    @staticmethod
    def positional_embedding(pos_seq, inv_freq, bsz=None):
        sinusoid_inp = torch.einsum("i,d->id", pos_seq, inv_freq)
        pos_emb = torch.cat([torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)], dim=-1)
        pos_emb = pos_emb[:, None, :]

        if bsz is not None:
            pos_emb = pos_emb.expand(-1, bsz, -1)

        return pos_emb

    #Batch-first in (N,S), batch-first out (N,C,S)
    def forward(self, input, print_kq=False):
        input2 = input.permute(1,0)
        ipos = torch.arange(input2.size(0), device=input.device)[:,None].expand(input2.shape[:2])
        h = self.embedding(input2)
        h = h + self.posembed(ipos)
        for layer in self.encoder:
            h, kq = layer(h)
        #out = torch.cat(out,dim=-1)
        out = self.fc(h).permute(1,2,0)
        if print_kq:
            return out, kq
        else:
            return out

class AMIBERT(nn.Module):
    def __init__(self, d_model=512, nhead=4, num_layers=6, maxlen=256, vocab_size=16, attn=AttentionalMemory):
        super().__init__()
        self.d_model=d_model
        self.maxlen=maxlen
        self.vocab_size = vocab_size
        assert d_model%2 == 0
        self.embedding = nn.Sequential(
            nn.Embedding(vocab_size, d_model),
            nn.LSTM(d_model, d_model//2, 1, bidirectional=True)
        )
        self.posembed = nn.Embedding(maxlen, d_model)
        self.encoder = nn.ModuleList([
            AMEncoderLayer(d_model=d_model, nhead=nhead, attn=attn) for _ in range(num_layers)
        ])
        self.fc = nn.Linear(d_model, vocab_size)

    @staticmethod
    def positional_embedding(pos_seq, inv_freq, bsz=None):
        sinusoid_inp = torch.einsum("i,d->id", pos_seq, inv_freq)
        pos_emb = torch.cat([torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)], dim=-1)
        pos_emb = pos_emb[:, None, :]

        if bsz is not None:
            pos_emb = pos_emb.expand(-1, bsz, -1)

        return pos_emb

    #Batch-first in (N,S), batch-first out (N,C,S)
    def forward(self, input, print_kq=False):
        input2 = input.permute(1,0)
        ipos = torch.arange(input2.size(0), device=input.device)[:,None].expand(input2.shape[:2])
        klen = input2.shape[0]
        rpos = torch.arange(self.maxlen-klen, self.maxlen+klen, device=input.device)
        # r = self.relembed(rpos[:,None].expand(2*klen,input2.shape[1]))
        src, _ = self.embedding(input2)
        #h = src + self.posembed(ipos)
        h = src
        for layer in self.encoder:
            h, kq = layer(h)
        #out = torch.cat(out,dim=-1)
        out = self.fc(h).permute(1,2,0)
        if print_kq:
            return out, kq
        else:
            return out

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

#Long Short-term Attentional Memory
# k,q,v = FF(x:h)
#A' = A + wt(kvT-k(Ak)T)
#h = Aq
class LSAM(nn.Module):
    def __init__(self, d_model, nhead, num_layers=2, sigma=unitnorm, activation = nn.ReLU(), drop=0.1):
        super().__init__()
        assert d_model % nhead == 0
        self.nhead = nhead
        self.encoder = nn.ModuleList([
            LSAMCell(input_dim=d_model, d_model=d_model, nhead=nhead) for _ in range(num_layers)
        ])
        
        self.sigma = sigma
        self.d_head = d_model//nhead
        self.d_model = d_model
        self.Wo = nn.Sequential(nn.Linear(d_model, d_model),
                                nn.Dropout(drop), activation)
        self.norm = nn.LayerNorm(d_model)
        self.num_layers = num_layers
    def forward(self, x, hA=None):
        #assuming (S,B,C) layout
        B = x.size(1)
        if hA is None:
            h_list = [torch.zeros([B, self.d_model],
            dtype=x.dtype, device=x.device) for _ in range(self.num_layers)]
            #B,N,D/N,D/N
            AM_list = [torch.zeros([B, self.nhead, self.d_head, self.d_head],
            dtype=x.dtype, device=x.device) for _ in range(self.num_layers)]
        else:
            h_list,AM_list = hA
        out = []
        for x_i in x:
            h = x_i
            for i, layer in enumerate(self.encoder):
                h, AM = layer(h, h_list[i], AM_list[i])
                h_list[i] = h
                AM_list[i] = AM
            out.append(h)
        out = torch.stack(out)
        out = self.Wo(out) + x
        return self.norm(out), (h,AM)

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

    def forward(self, x, hA):
        #assuming (S,B,C) layout
        B = x.size(1)
        h,AM = hA
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
        fw_module = LSAMCell(input_dim=input_dim, d_model=d_model//2, nhead=nhead//2)
        bw_module = LSAMCell(input_dim=input_dim, d_model=d_model//2, nhead=nhead//2)
        self.enc_fw = torch.jit.script(fw_module)
        self.enc_bw = torch.jit.script(bw_module)
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
        #if self.input_dim == self.d_model:
        #    out = out + x
        #out = self.norm(out)
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
        #self.encoder = LSAM(d_model=d_model, nhead=nhead, num_layers=num_layers)
        #self.encoder = BLSAM(input_dim=d_model, d_model=d_model, nhead=nhead)
        #self.encoder2 = BLSAM(input_dim=d_model, d_model=d_model, nhead=nhead)
        self.encoder = LSAMEncoderLayer(d_model=d_model, nhead=nhead)
        self.encoder2 = AMEncoderLayer(d_model=d_model, nhead=nhead)
        #self.encoder3 = AMEncoderLayer(d_model=d_model, nhead=nhead)
        self.decoder = LSAMDecoder(input_dim=d_model, d_model=d_model, nhead=nhead)
        self.fc = nn.Linear(d_model, vocab_size)

    #Batch-first in (N,S), batch-first out (N,C,S)
    def forward(self, input, encoder_mode=False):
        input2 = input.permute(1,0)

        src = self.embedding(input2)

        out, hA = self.encoder(src)
        out, hA = self.decoder(out, hA)
        #out, hA = self.encoder2(out)
        #out, hA = self.encoder3(out)
        out = self.fc(out).permute(1,2,0)
        if encoder_mode:
            return out, hA[1]
        else:
            return out

class LSAMS2S(nn.Module):
    def __init__(self, d_model=256, nhead=4, vocab_size=16):
        super().__init__()
        self.d_model=d_model
        self.vocab_size = vocab_size
        assert d_model%2 == 0
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(vocab_size, d_model)
        self.encoder = LSAMEncoderLayer(d_model=d_model, nhead=nhead)
        self.decoder = LSAMDecoder(input_dim=d_model, d_model=d_model, nhead=nhead)
        self.fc = nn.Linear(d_model, vocab_size)

    #Batch-first in (N,S), batch-first out (N,C,S)
    def forward(self, input, tgt, encoder_mode=False):
        input2 = input.permute(1,0)

        src = self.embedding(input2)
        tsrc = self.tgt_embedding(tgt.permute(1,0))
        out, hA = self.encoder(src)
        out, hA = self.decoder(tsrc, hA)
        #out, hA = self.encoder2(out)
        #out, hA = self.encoder3(out)
        out = self.fc(out).permute(1,2,0)
        if encoder_mode:
            return out, hA[1]
        else:
            return out


class NAMTuring(nn.Module):
    def __init__(self,in_dim, hidden_dim, n_tapes=2):
        super().__init__()
        assert hidden_dim%n_tapes == 0
        self.dim=hidden_dim//n_tapes
        self.in_dim = in_dim
        self.n_tapes = n_tapes
        # (prev, next, no-op)
        #direction layers for read/write heads
        #action: (read_direction(3), write_direction(3), rwprob(2))
        self.actionlayer = nn.LSTM(self.in_dim, 8*self.n_tapes, 1, bidirectional=False)
        self.valuelayer = nn.Linear(self.in_dim, self.dim*self.n_tapes)
        self.outlayer = nn.Linear(hidden_dim,hidden_dim)
        
    #Seq-first in (S,N,C), seq-first out (S,N,C)
    #Stack: initial stack state (zeros or null embeddings)
    def forward(self, inputs, tapelen, tape_in=None, pos_in=None):
        seqlen = inputs.shape[0]
        batchsize = inputs.shape[1]

        values = self.valuelayer(inputs).reshape(seqlen, batchsize, self.n_tapes, self.dim)
        #(L,N,T,C)
        if tape_in is None:
            tape = torch.zeros([tapelen, batchsize, self.n_tapes,self.dim],
                            dtype=inputs.dtype, device=inputs.device)
        else:
            tape = tape_in
        #(L,N,T)
        if pos_in is None:
            rpos = torch.zeros([tapelen, batchsize, self.n_tapes],
                dtype=inputs.dtype, device=inputs.device)
            wpos = torch.zeros([tapelen, batchsize, self.n_tapes],
                dtype=inputs.dtype, device=inputs.device)
            rpos[0,:,:] = 1.0
            wpos[0,:,:] = 1.0
        else:
            rpos,wpos = pos_in
       
        #(S,N,C) -> (S,N,nh,8)
        actions = self.actionlayer(inputs)[0].reshape(seqlen,batchsize,self.n_tapes,8)

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
        return outputs, tape, (rpos,wpos)

class NAMTuringAE(nn.Module):
    def __init__(self,d_model, vocab_size, n_tapes=2, default_tapelen=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.dim=d_model
        self.n_tapes = n_tapes
        self.controller =  nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.LayerNorm(d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.tm=NAMTuring(d_model,d_model,n_tapes)
        self.d_tapelen=default_tapelen
        self.fc = nn.Sequential(nn.LayerNorm(d_model), 
            nn.ReLU(),
            nn.Linear(d_model, vocab_size))

        
    #Seq-first in (S,N), seq-first out (S,N,C)
    #Stack: initial stack state (zeros or null embeddings)
    def forward(self, inputs, tapelen=-1):
        if tapelen <= 0:
            tapelen = self.d_tapelen
        embedded = self.embedding(inputs.permute(1,0))
        hidden = self.controller(embedded)
        hidden =  self.norm1(hidden)
        output, _, _= self.tm(hidden, tapelen)
        output = output+hidden
        
        return self.fc(output).permute(1,2,0)