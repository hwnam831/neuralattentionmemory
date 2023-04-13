from SAM.baselines.sam.stm_basic import STM
import torch
import torch.nn as nn
from torch.nn import functional as F


class STMAE(nn.Module):
    def __init__(self, dim, vocab_size, nhead=4, defalt_tapelen=32, option='default', debug=False, mem_size=64):
        super().__init__()
        self.dim=dim
        self.embedding = nn.Sequential(nn.Embedding(vocab_size, dim),
                                        nn.Linear(dim, dim),
                                     nn.Dropout(0.2))

        
        #S,N,C in, S,N,C out?
        self.stm = STM(self.dim,self.dim,num_slot=nhead, slot_size=mem_size,
                       rel_size=mem_size,out_att_size=mem_size, mlp_size=self.dim)
        self.fc = nn.Sequential(nn.LayerNorm(dim),
            nn.Dropout(0.1), nn.ReLU(),
            nn.Linear(dim, vocab_size))
    #Batch-first in (N,S), batch-first out (N,C,S)
    def forward(self, input):
        input2 = input.permute(1,0)
        self.stm.init_sequence(batch_size=input.shape[0])
        outputs = []
        src = self.embedding(input2)
        for i in range(input.shape[1]):
            sout, _ = self.stm(src[i])
            outputs.append(sout)
        out_vec = torch.stack(outputs) #S,N,C

        #S,N,C to N,C,S
        return self.fc(out_vec).permute(1,2,0)