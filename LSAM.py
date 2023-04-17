from unicodedata import bidirectional
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
def unitnorm(v):
    return F.normalize(v, dim=-1)

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