import torch
import torch.nn as nn
import torch.nn.functional as F
from Encoder import CNNEncoder, XLNetEncoderLayer
from dncmds.Models.DNC import DNC as DNCMDS
from dncmds.Models.DNC import LSTMController
from transformer_generalization.layers.transformer.universal_transformer import UniversalTransformerEncoder
from transformer_generalization.layers.transformer.universal_relative_transformer import RelativeTransformerEncoderLayer


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

class TfAE(nn.Module):
    def __init__(self, model_size=512, nhead=4, num_layers=6, maxlen=512, vocab_size=16):
        super().__init__()
        self.model_size=model_size
        self.maxlen=maxlen
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, model_size)
        self.posembed = nn.Embedding(maxlen, model_size)
        self.enclayer = nn.TransformerEncoderLayer(d_model=model_size, nhead=nhead)
        self.norm = nn.LayerNorm(model_size)
        self.tfmodel = nn.TransformerEncoder(self.enclayer, num_layers=num_layers, norm=self.norm)
        self.fc = nn.Linear(model_size, vocab_size)
    #Batch-first in (N,S,C), batch-first out (N,C,S)
    def forward(self, input): 
        input2 = input.permute(1,0)
        ipos = torch.arange(input2.size(0), device=input.device)[:,None].expand(input2.shape[:2])
        src = self.embedding(input2) + self.posembed(ipos)
        out = self.tfmodel(src)
        return self.fc(out).permute(1,2,0)

class CNNAE(nn.Module):
    def __init__(self, model_size=512,vocab_size=16):
        super().__init__()
        self.model_size=model_size
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, model_size)
        self.norm = nn.LayerNorm(model_size)
        self.encoder = CNNEncoder(model_size)
        self.fc = nn.Linear(model_size, vocab_size)
    
    #Batch-first in (N,S), batch-first out (N,C,S)
    def forward(self, input):
        embed = self.norm(self.embedding(input.permute(1,0)))
        out = self.encoder(embed)
        return self.fc(out).permute(1,2,0)

class XLNetAE(nn.Module):
    def __init__(self, d_model=512, nhead=4, maxlen=256, num_layers=6, vocab_size=16):
        super().__init__()
        self.d_model=d_model
        self.maxlen=maxlen
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.posembed = nn.Embedding(maxlen, d_model)
        self.relembed = nn.Embedding(2*maxlen, d_model)

        self.encoder = nn.ModuleList([
            XLNetEncoderLayer(d_model=d_model, nhead=nhead) for _ in range(num_layers)
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

    def relative_positional_encoding(self, qlen, klen, bsz=None):
        # create relative positional encoding.
        freq_seq = torch.arange(0, self.d_model, 2.0, dtype=torch.float)
        inv_freq = 1 / torch.pow(10000, (freq_seq / self.d_model))

        beg, end = klen, -qlen

        fwd_pos_seq = torch.arange(beg, end, -1.0)
        pos_emb = self.positional_embedding(fwd_pos_seq, inv_freq, bsz)

        pos_emb = pos_emb.to(next(self.parameters()))
        return pos_emb

    #Batch-first in (N,S), batch-first out (N,C,S)
    def forward(self, input):
        input2 = input.permute(1,0)
        ipos = torch.arange(input2.size(0), device=input.device)[:,None].expand(input2.shape[:2])
        r = self.relative_positional_encoding(input2.size(0),input2.size(0),input.size(0))
        klen = input2.shape[0]
        rpos = torch.arange(self.maxlen-klen, self.maxlen+klen, device=input.device)

        src = self.embedding(input2)
        h,g = (src, src)
        for layer in self.encoder:
            h,g = layer(h,g,r)

        return self.fc(g).permute(1,2,0)

class DNCMDSAE(nn.Module):
    def __init__(self, model_size=64, nhead=4, nr_cells=32, vocab_size=16, mem_size=64):
        super().__init__()
        self.model_size=model_size
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, model_size)

        self.cell = DNCMDS(model_size, model_size, mem_size, nr_cells, nhead, LSTMController([model_size]),
                           batch_first=True,mask=True, dealloc_content=True, link_sharpness_control=True)
        
        self.fc = nn.Linear(model_size, vocab_size)
    #Batch-first in (N,S), batch-first out (N,C,S)
    def forward(self, input):
        src = self.cell(self.embedding(input)) #N, S, C
        return self.fc(src).permute(0,2,1)

# From https://github.com/pytorch/fairseq/blob/master/fairseq/models/lstm.py
class AttentionLayer(nn.Module):
    def __init__(self, input_embed_dim, source_embed_dim, output_embed_dim, bias=False):
        super().__init__()

        self.input_proj = nn.Linear(input_embed_dim, source_embed_dim, bias=bias)
        self.output_proj = nn.Linear(input_embed_dim + source_embed_dim, output_embed_dim, bias=bias)

    def forward(self, input, source_hids):
        # input: tgtlen x bsz x input_embed_dim
        # source_hids: srclen x bsz x source_embed_dim

        # x: bsz x source_embed_dim
        x = self.input_proj(input)

        # compute attention
        attn_scores = torch.einsum('sbh,tbh->tsb', source_hids, x)

        attn_scores = F.softmax(attn_scores, dim=1)  # srclen x bsz

        # sum weighted sources
        x = torch.einsum('tsb, sbh->tbh',attn_scores, source_hids)

        x = torch.tanh(self.output_proj(torch.cat((x, input), dim=-1)))
        return x, attn_scores

class LSTMAE(nn.Module):
    def __init__(self, model_size, vocab_size=16):
        super().__init__()
        assert model_size %2 == 0
        self.model_size = model_size
        self.embed = nn.Embedding(vocab_size, self.model_size)
        self.encoder = nn.LSTM(self.model_size, self.model_size//2, 1, bidirectional=True)
        self.dropout = nn.Dropout(0.1)
        self.attn = AttentionLayer(model_size, model_size, model_size)
        self.decoder = nn.LSTM(self.model_size, self.model_size//2, 1, bidirectional=True)
        self.fc = nn.Linear(model_size, vocab_size)

    def forward(self, input):
        outputs = self.dropout(self.embed(input.permute(1,0)))
        outputs, state = self.encoder(outputs)
        outputs, _ = self.attn(outputs, outputs)
        outputs, state = self.decoder(self.dropout(outputs))
        return self.fc(self.dropout(outputs)).permute(1,2,0)
    
class LSTMNoAtt(nn.Module):
    def __init__(self, model_size, num_layers=2, vocab_size=16):
        super().__init__()
        assert model_size %2 == 0
        self.model_size = model_size
        self.embed = nn.Embedding(vocab_size, self.model_size)
        self.encoder = nn.LSTM(self.model_size, self.model_size//2, num_layers, bidirectional=True, dropout=0.1)
        self.fc = nn.Linear(model_size, vocab_size)

    def forward(self, input):
        outputs = self.embed(input.permute(1,0))
        outputs, state = self.encoder(outputs)
        return self.fc(F.relu(outputs)).permute(1,2,0)


    
class UTRelAE(nn.Module):
    def __init__(self, model_size=64, nhead=4, num_layers=2, vocab_size=16):
        super().__init__()
        self.model_size=model_size
        self.vocab_size = vocab_size
        assert model_size%2 == 0
        self.embedding = nn.Embedding(vocab_size, model_size)
        self.cell = UniversalTransformerEncoder(RelativeTransformerEncoderLayer,
                                                depth=num_layers, d_model=model_size, nhead=nhead)
        self.fc = nn.Linear(model_size, vocab_size)

    #Batch-first in (N,S), batch-first out (N,C,S)
    def forward(self, input):

        src = self.cell(self.embedding(input)) #UT takes N, S, C
        return self.fc(src).permute(0,2,1)