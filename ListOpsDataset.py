import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader

class ListopsDataset_old(Dataset):
    def __init__(self, tsv_file, max_len=-1):
        
        vocabs = ['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15',
                    '(', ')', '[MIN', '[MAX', '[MED', '[FIRST', '[LAST', '[SM', ']', '<MASK>', '<PAD>']
        self.dict = {}
        
        for i,v in enumerate(vocabs):
            self.dict[v] = i
        self.wordtoix = self.dict
        self.vocab_size = len(vocabs) 
        self.inputs = []
        self.targets = []
        with open(tsv_file, "r") as fd:
            for l in fd:
                inp, tgt, guide = l.split('\t')
                tokens = inp.split(' ')
                if len(tokens) > max_len:
                    max_len = len(tokens)
                seq = [self.dict[tok] for tok in tokens]
                self.inputs.append(seq)
                self.targets.append(int(tgt))
        self.max_len = max_len
        self.inp_arr = np.ones([len(self.targets), self.max_len+2], dtype=np.int64)*(self.dict['<PAD>']) #pre-fill with padding
        self.out_arr = np.ones([len(self.targets), self.max_len+2], dtype=np.int64)*(self.dict['<PAD>']) #pre-fill with padding

        for idx, inp in enumerate(self.inputs):
            for i,t in enumerate(inp):
                self.inp_arr[idx, i] = t
                self.out_arr[idx, i] = t
            self.inp_arr[idx,len(inp)] = self.dict['<MASK>']
            self.out_arr[idx,len(inp)] = self.targets[idx]
        self.size = len(self.targets)


    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.inp_arr[idx], self.out_arr[idx]

class ListopsDataset(Dataset):
    def __init__(self, tsv_file, max_len=-1):
        
        vocabs = ['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15',
                    '(', ')', '[MIN', '[MAX', '[MED', '[FIRST', '[LAST', '[SM', ']', '<MASK>', '<PAD>']
        self.dict = {}
        
        for i,v in enumerate(vocabs):
            self.dict[v] = i
        self.wordtoix = self.dict
        self.ixtoword = vocabs
        self.vocab_size = len(vocabs) 
        self.inputs = []
        self.ops = []
        self.vals = []
        self.targets = []
        with open(tsv_file, "r") as fd:
            for l in fd:
                inp, tgt, guide = l.split('\t')
                tokens = inp.split(' ')
                guide_pairs = guide.strip().split(' ')
                if len(tokens) > max_len:
                    max_len = len(tokens)
                seq = [self.dict[tok] for tok in tokens]
                guide_op = [self.dict[g.split(',')[0]] for g in guide_pairs]
                guide_val = [self.dict[g.split(',')[1]] for g in guide_pairs]
                self.inputs.append(seq)
                self.ops.append(guide_op)
                self.vals.append(guide_val)
                self.targets.append(int(tgt))
        self.max_len = max_len
        self.inp_arr = np.ones([len(self.targets), self.max_len+2], dtype=np.int64)*(self.dict['<PAD>']) #pre-fill with padding
        self.out_arr = np.ones([len(self.targets), self.max_len+2], dtype=np.int64)*(self.dict['<PAD>']) #pre-fill with padding

        for idx, inp in enumerate(self.inputs):
            for i,t in enumerate(inp):
                self.inp_arr[idx, i] = t
                self.out_arr[idx, i] = self.vals[idx][i]
            self.inp_arr[idx,len(inp)] = self.dict['<MASK>']
            self.out_arr[idx,len(inp)] = self.targets[idx]
        self.size = len(self.targets)


    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.inp_arr[idx], self.out_arr[idx]

class GuidedListops(Dataset):
    def __init__(self, tsv_file, max_len=-1):
        
        vocabs = ['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15',
                    '(', ')', '[MIN', '[MAX', '[MED', '[FIRST', '[LAST', '[SM', ']', '<MASK>', '<PAD>']
        self.dict = {}
        
        for i,v in enumerate(vocabs):
            self.dict[v] = i
        self.wordtoix = self.dict
        self.ixtoword = vocabs
        self.vocab_size = len(vocabs) 
        self.inputs = []
        self.guides = []
        with open(tsv_file, "r") as fd:
            for l in fd:
                inp, tgt, guide = l.split('\t')
                tokens = inp.split(' ')
                guide_pairs = guide.strip().split(' ')
                if len(tokens) > max_len:
                    max_len = len(tokens)
                seq = [self.dict[tok] for tok in tokens]
                self.inputs.append(seq)
                self.guides.append(guide_pairs)
                
        self.max_len = max_len
        self.inp_arr = np.ones([len(self.inputs), self.max_len], dtype=np.int64)*(self.dict['<PAD>']) #pre-fill with padding
        self.out_arr = np.ones([len(self.inputs), self.max_len, 4], dtype=np.int64)*(self.dict['<PAD>']) #pre-fill with padding

        for idx, inp in enumerate(self.inputs):
            for i,t in enumerate(inp):
                self.inp_arr[idx, i] = t
                self.out_arr[idx, i] = [self.dict[g] for g in self.guides[idx][i].split(',')]
        self.size = len(self.inputs)


    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.inp_arr[idx], self.out_arr[idx]

if __name__ == '__main__':
    dset = GuidedListops('listopsdata/basic_train.tsv')
    inp, out = dset[10]
    print([dset.ixtoword[t] for t in inp])
    print([ ','.join([dset.ixtoword[t] for t in row])  for row in out])
    #print(out)
    dset2 = ListopsDataset('listopsdata/basic_train.tsv')
    inp, out = dset2[10]
    print([dset.ixtoword[t] for t in out])
    #print([ ','.join([dset.ixtoword[t] for t in row])  for row in out])