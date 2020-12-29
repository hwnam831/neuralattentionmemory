import torch
import torch.nn as nn
from torch.nn import functional as F
import math

def unitnorm(v):
    return v/torch.norm(v,dim=-1).unsqueeze(-1)

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
'''
class RecurrentAM(nn.Module):
    def __init__(self, d_model, nhead, sigma=unitnorm):
        super().__init__()
        assert d_model % nhead == 0
        self.nhead = nhead
        self.dim = d_model//nhead
        self.Wq = nn.Linear(d_model*2, d_model)
        self.Wk = nn.Linear(d_model*2, d_model)
        self.Wv = nn.Linear(d_model*2, d_model)
        self.Wo = nn.Linear(d_model, d_model)
        self.sigma = sigma
    def forward(self, x):
        #assuming (S,B,C) layout
        B,C = x.size(1), x.size(2)
        r = None
        A = torch.zeros([B, self.nhead, self.dim, self.dim], device=x.device)
        for x_i in x: #(B,C) shapes
            if r is not None:
                x_r = torch.cat([x_i,r],dim=-1)
            else:
                x_r = torch.cat([x_i,torch.zeros_like(x_i)],dim=-1)
            k = self.Wk(x_r).reshape(B,self.nhead,-1) #(S,B,n,Dk/n)
            v = self.Wv(x_r).reshape(B,self.nhead,-1) #(S,B,n,Dv/n)
            #k = self.sigma(k)
            k = unitelu(k)
            A = A + torch.einsum('bnq,bnv->bnvq', k,v)
            q = self.Wq(x_r).reshape(B,self.nhead,-1) #(S,B,n,Dq=Dk/n)
            q = self.sigma(q)
            r = torch.einsum('bnq,bnvq->bnv',q,A).reshape(B,-1)
            #r = torch.matmul(q,A).view(B,-1)
        out = self.Wo(r)
        return out, (k,q)
'''

class RecurrentAM(nn.Module):
    def __init__(self, d_model, nhead, sigma=unitnorm):
        super().__init__()
        assert d_model % nhead == 0
        self.nhead = nhead
        self.d_model = d_model
        self.dim = d_model//nhead
        self.Wqkv = nn.GRUCell(d_model*2, d_model*3)
        self.Wo = nn.Linear(d_model, d_model)
        self.sigma = sigma
    def forward(self, x):
        #assuming (S,B,C) layout
        B,C = x.size(1), x.size(2)
        r = None
        A = torch.zeros([B, self.nhead, self.dim, self.dim], device=x.device)
        h = None
        for x_i in x: #(B,C) shapes
            if r is not None:
                x_r = torch.cat([x_i,r],dim=-1)
            else:
                x_r = torch.cat([x_i,torch.zeros_like(x_i)],dim=-1)
            if h is None:
                h = self.Wqkv(x_r)
            else:
                h = self.Wqkv(x_r,h)
            k = unitelu(h[:,self.d_model:2*self.d_model].reshape(B,self.nhead,-1))
            v = h[:,2*self.d_model:3*self.d_model].reshape(B,self.nhead,-1)
            A = A + torch.einsum('bnq,bnv->bnvq', k,v)
            q = h[:,:self.d_model].reshape(B,self.nhead,-1)
            q = self.sigma(q)
            r = torch.einsum('bnq,bnvq->bnv',q,A).reshape(B,-1)
        out = self.Wo(r)
        return out, (k,q)

class AttentionMatrix(nn.Module):
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
        #k = self.sigma(k)
        k = unitelu(k)
        A = torch.einsum('sbnq,sbnv->bnvq', k,v)
        q = self.Wq(h).reshape(S,B,self.nhead,-1) #(S,B,n,Dq=Dk/n)
        q = self.sigma(q)
        out = torch.einsum('sbnq,bnvq->sbnv',q,A).reshape(S,B,-1)
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
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, attn=AttentionMatrix):
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
    def __init__(self, d_model=512, nhead=4, num_layers=6, maxlen=256, vocab_size=16, attn=AttentionMatrix):
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
        #h = h + self.posembed(ipos)
        for layer in self.encoder:
            h, kq = layer(h)
        #out = torch.cat(out,dim=-1)
        out = self.fc(h).permute(1,2,0)
        if print_kq:
            return out, kq
        else:
            return out

class AMIBERT(nn.Module):
    def __init__(self, d_model=512, nhead=4, num_layers=6, maxlen=256, vocab_size=16, attn=AttentionMatrix):
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


#reset
#forward
class AttentionalMemory(nn.Module):

  def __init__(self, input_size, mem_size=512, cell_size=32, read_heads=4, gpu_id=-1, independent_linears=True):
    super(Memory, self).__init__()

    self.mem_size = mem_size
    self.cell_size = cell_size
    self.read_heads = read_heads
    self.gpu_id = gpu_id
    self.input_size = input_size
    self.independent_linears = independent_linears

    m = self.mem_size
    w = self.cell_size
    r = self.read_heads

    if self.independent_linears:
      self.read_keys_transform = nn.Linear(self.input_size, w * r)
      self.read_strengths_transform = nn.Linear(self.input_size, r)
      self.write_key_transform = nn.Linear(self.input_size, w)
      self.write_strength_transform = nn.Linear(self.input_size, 1)
      self.erase_vector_transform = nn.Linear(self.input_size, w)
      self.write_vector_transform = nn.Linear(self.input_size, w)
      self.free_gates_transform = nn.Linear(self.input_size, r)
      self.allocation_gate_transform = nn.Linear(self.input_size, 1)
      self.write_gate_transform = nn.Linear(self.input_size, 1)
      self.read_modes_transform = nn.Linear(self.input_size, 3 * r)
    else:
      self.interface_size = (w * r) + (3 * w) + (5 * r) + 3
      self.interface_weights = nn.Linear(self.input_size, self.interface_size)

    self.I = cuda(1 - T.eye(m).unsqueeze(0), gpu_id=self.gpu_id)  # (1 * n * n)

  def reset(self, batch_size=1, hidden=None, erase=True): #
    m = self.mem_size
    w = self.cell_size
    r = self.read_heads
    b = batch_size

    if hidden is None:
      return {
          'memory': cuda(T.zeros(b, m, w).fill_(0), gpu_id=self.gpu_id),
          'link_matrix': cuda(T.zeros(b, 1, m, m), gpu_id=self.gpu_id),
          'precedence': cuda(T.zeros(b, 1, m), gpu_id=self.gpu_id),
          'read_weights': cuda(T.zeros(b, r, m).fill_(0), gpu_id=self.gpu_id),
          'write_weights': cuda(T.zeros(b, 1, m).fill_(0), gpu_id=self.gpu_id),
          'usage_vector': cuda(T.zeros(b, m), gpu_id=self.gpu_id)
      }
    else:
      hidden['memory'] = hidden['memory'].clone()
      hidden['link_matrix'] = hidden['link_matrix'].clone()
      hidden['precedence'] = hidden['precedence'].clone()
      hidden['read_weights'] = hidden['read_weights'].clone()
      hidden['write_weights'] = hidden['write_weights'].clone()
      hidden['usage_vector'] = hidden['usage_vector'].clone()

      if erase:
        hidden['memory'].data.fill_(0)
        hidden['link_matrix'].data.zero_()
        hidden['precedence'].data.zero_()
        hidden['read_weights'].data.fill_(0)
        hidden['write_weights'].data.fill_(0)
        hidden['usage_vector'].data.zero_()
    return hidden

  def get_usage_vector(self, usage, free_gates, read_weights, write_weights): #
    # write_weights = write_weights.detach()  # detach from the computation graph
    usage = usage + (1 - usage) * (1 - T.prod(1 - write_weights, 1))
    ψ = T.prod(1 - free_gates.unsqueeze(2) * read_weights, 1)
    return usage * ψ

  def allocate(self, usage, write_gate): #
    # ensure values are not too small prior to cumprod.
    usage = δ + (1 - δ) * usage
    batch_size = usage.size(0)
    # free list
    sorted_usage, φ = T.topk(usage, self.mem_size, dim=1, largest=False)

    # cumprod with exclusive=True
    # https://discuss.pytorch.org/t/cumprod-exclusive-true-equivalences/2614/8
    v = var(sorted_usage.data.new(batch_size, 1).fill_(1))
    cat_sorted_usage = T.cat((v, sorted_usage), 1)
    prod_sorted_usage = T.cumprod(cat_sorted_usage, 1)[:, :-1]

    sorted_allocation_weights = (1 - sorted_usage) * prod_sorted_usage.squeeze()

    # construct the reverse sorting index https://stackoverflow.com/questions/2483696/undo-or-reverse-argsort-python
    _, φ_rev = T.topk(φ, k=self.mem_size, dim=1, largest=False)
    allocation_weights = sorted_allocation_weights.gather(1, φ_rev.long())

    return allocation_weights.unsqueeze(1), usage

  def write_weighting(self, memory, write_content_weights, allocation_weights, write_gate, allocation_gate): #
    ag = allocation_gate.unsqueeze(-1)
    wg = write_gate.unsqueeze(-1)

    return wg * (ag * allocation_weights + (1 - ag) * write_content_weights)

  def get_link_matrix(self, link_matrix, write_weights, precedence): #
    precedence = precedence.unsqueeze(2)
    write_weights_i = write_weights.unsqueeze(3)
    write_weights_j = write_weights.unsqueeze(2)

    prev_scale = 1 - write_weights_i - write_weights_j
    new_link_matrix = write_weights_i * precedence

    link_matrix = prev_scale * link_matrix + new_link_matrix
    # trick to delete diag elems
    return self.I.expand_as(link_matrix) * link_matrix

  def update_precedence(self, precedence, write_weights): #
    return (1 - T.sum(write_weights, 2, keepdim=True)) * precedence + write_weights

  def write(self, write_key, write_vector, erase_vector, free_gates, read_strengths, write_strength, write_gate, allocation_gate, hidden): #
    # get current usage
    hidden['usage_vector'] = self.get_usage_vector(
        hidden['usage_vector'],
        free_gates,
        hidden['read_weights'],
        hidden['write_weights']
    )

    # lookup memory with write_key and write_strength
    write_content_weights = self.content_weightings(hidden['memory'], write_key, write_strength)

    # get memory allocation
    alloc, _ = self.allocate(
        hidden['usage_vector'],
        allocation_gate * write_gate
    )

    # get write weightings
    hidden['write_weights'] = self.write_weighting(
        hidden['memory'],
        write_content_weights,
        alloc,
        write_gate,
        allocation_gate
    )

    weighted_resets = hidden['write_weights'].unsqueeze(3) * erase_vector.unsqueeze(2)
    reset_gate = T.prod(1 - weighted_resets, 1)
    # Update memory
    hidden['memory'] = hidden['memory'] * reset_gate

    hidden['memory'] = hidden['memory'] + \
        T.bmm(hidden['write_weights'].transpose(1, 2), write_vector)

    # update link_matrix
    hidden['link_matrix'] = self.get_link_matrix(
        hidden['link_matrix'],
        hidden['write_weights'],
        hidden['precedence']
    )
    hidden['precedence'] = self.update_precedence(hidden['precedence'], hidden['write_weights'])

    return hidden

  def content_weightings(self, memory, keys, strengths): #
    d = θ(memory, keys)
    return σ(d * strengths.unsqueeze(2), 2)

  def directional_weightings(self, link_matrix, read_weights): #
    rw = read_weights.unsqueeze(1)

    f = T.matmul(link_matrix, rw.transpose(2, 3)).transpose(2, 3)
    b = T.matmul(rw, link_matrix)
    return f.transpose(1, 2), b.transpose(1, 2)

  def read_weightings(self, memory, content_weights, link_matrix, read_modes, read_weights): #
    forward_weight, backward_weight = self.directional_weightings(link_matrix, read_weights)

    content_mode = read_modes[:, :, 2].contiguous().unsqueeze(2) * content_weights
    backward_mode = T.sum(read_modes[:, :, 0:1].contiguous().unsqueeze(3) * backward_weight, 2)
    forward_mode = T.sum(read_modes[:, :, 1:2].contiguous().unsqueeze(3) * forward_weight, 2)

    return backward_mode + content_mode + forward_mode

  def read_vectors(self, memory, read_weights): #
    return T.bmm(read_weights, memory)

  def read(self, read_keys, read_strengths, read_modes, hidden): #
    content_weights = self.content_weightings(hidden['memory'], read_keys, read_strengths)

    hidden['read_weights'] = self.read_weightings(
        hidden['memory'],
        content_weights,
        hidden['link_matrix'],
        read_modes,
        hidden['read_weights']
    )
    read_vectors = self.read_vectors(hidden['memory'], hidden['read_weights'])
    return read_vectors, hidden

  def forward(self, ξ, hidden):

    # ξ = ξ.detach()
    m = self.mem_size
    w = self.cell_size
    r = self.read_heads
    b = ξ.size()[0]

    if self.independent_linears:
      # r read keys (b * r * w)
      read_keys = T.tanh(self.read_keys_transform(ξ).view(b, r, w))
      # r read strengths (b * r)
      read_strengths = F.softplus(self.read_strengths_transform(ξ).view(b, r))
      # write key (b * 1 * w)
      write_key = T.tanh(self.write_key_transform(ξ).view(b, 1, w))
      # write strength (b * 1)
      write_strength = F.softplus(self.write_strength_transform(ξ).view(b, 1))
      # erase vector (b * 1 * w)
      erase_vector = T.sigmoid(self.erase_vector_transform(ξ).view(b, 1, w))
      # write vector (b * 1 * w)
      write_vector = T.tanh(self.write_vector_transform(ξ).view(b, 1, w))
      # r free gates (b * r)
      free_gates = T.sigmoid(self.free_gates_transform(ξ).view(b, r))
      # allocation gate (b * 1)
      allocation_gate = T.sigmoid(self.allocation_gate_transform(ξ).view(b, 1))
      # write gate (b * 1)
      write_gate = T.sigmoid(self.write_gate_transform(ξ).view(b, 1))
      # read modes (b * r * 3)
      read_modes = σ(self.read_modes_transform(ξ).view(b, r, 3), -1)
    else:
      ξ = self.interface_weights(ξ)
      # r read keys (b * w * r)
      read_keys = T.tanh(ξ[:, :r * w].contiguous().view(b, r, w))
      # r read strengths (b * r)
      read_strengths = F.softplus(ξ[:, r * w:r * w + r].contiguous().view(b, r))
      # write key (b * w * 1)
      write_key = T.tanh(ξ[:, r * w + r:r * w + r + w].contiguous().view(b, 1, w))
      # write strength (b * 1)
      write_strength = F.softplus(ξ[:, r * w + r + w].contiguous().view(b, 1))
      # erase vector (b * w)
      erase_vector = T.sigmoid(ξ[:, r * w + r + w + 1: r * w + r + 2 * w + 1].contiguous().view(b, 1, w))
      # write vector (b * w)
      write_vector = T.tanh(ξ[:, r * w + r + 2 * w + 1: r * w + r + 3 * w + 1].contiguous().view(b, 1, w))
      # r free gates (b * r)
      free_gates = T.sigmoid(ξ[:, r * w + r + 3 * w + 1: r * w + 2 * r + 3 * w + 1].contiguous().view(b, r))
      # allocation gate (b * 1)
      allocation_gate = T.sigmoid(ξ[:, r * w + 2 * r + 3 * w + 1].contiguous().unsqueeze(1).view(b, 1))
      # write gate (b * 1)
      write_gate = T.sigmoid(ξ[:, r * w + 2 * r + 3 * w + 2].contiguous()).unsqueeze(1).view(b, 1)
      # read modes (b * 3*r)
      read_modes = σ(ξ[:, r * w + 2 * r + 3 * w + 3: r * w + 5 * r + 3 * w + 3].contiguous().view(b, r, 3), -1)

    hidden = self.write(write_key, write_vector, erase_vector, free_gates,
                        read_strengths, write_strength, write_gate, allocation_gate, hidden)
    return self.read(read_keys, read_strengths, read_modes, hidden)