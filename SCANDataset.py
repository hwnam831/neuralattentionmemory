import io  
import numpy as np
#from torchtext import data
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
import torch
import random
from transformer_generalization.dataset.scan_length_resplit import ScanLengthResplit
ixtoword = ['IN:', 'OUT:', 'walk', 'run', 'look', 'jump', 'turn',\
            'right', 'left', 'opposite', 'around', 'twice', 'thrice', 'after', 'and',\
            'I_TURN_RIGHT', 'I_TURN_LEFT', 'I_RUN', 'I_WALK', 'I_LOOK', 'I_JUMP', '_', '<eos>']
wordtoix = {}
for idx, w in enumerate(ixtoword):
    wordtoix[w] = idx

#    'IN:':0, 'OUT:':1, 'walk':2, 'run':3, 'look':4, 'jump':5, 'turn':6,\
#        'right':7, 'left':8, 'opposite':9, 'around':10, 'twice':11, 'thrice':12, 'after':13, 'and':14,\
#        'I_TURN_RIGHT':15, 'I_TURN_LEFT':16, 'I_RUN':17, 'I_WALK':18, 'I_LOOK':19, 'I_JUMP':20, '_':21, '<eos>':22

class SCANDatasetAE(Dataset):
    
    def __init__(self, filepath):
        self.ixtoword = ixtoword
        
        self.wordtoix = wordtoix
        

        self.vocab_size = len(self.ixtoword)
        self.inputs = []
        self.targets = []
        with open(filepath, 'r') as txtfile:
            for line in txtfile:
                tokens = line.strip().split(' ')
                iseq = np.ones([len(tokens)+1],dtype=np.int64)*self.wordtoix['_']
                tseq = np.ones([len(tokens)+1],dtype=np.int64)*self.wordtoix['_']
                idx = 0
                while (tokens[idx] != 'OUT:'):
                    iseq[idx] = self.wordtoix[tokens[idx]]
                    idx += 1
                while (idx < len(tokens)):
                    tseq[idx] = self.wordtoix[tokens[idx]]
                    idx += 1
                tseq[-1] = self.wordtoix['<eos>']
                self.inputs.append(iseq)
                self.targets.append(tseq)

    def __getitem__(self, index):
        return self.inputs[index], self.targets[index]

    def __len__(self):
        return len(self.inputs)
    def collate_batch(batch):
        seq_len = max([len(seq) for seq, _ in batch])
        input_tensor = torch.ones([len(batch), seq_len], dtype=torch.int64)*wordtoix['_']
        target_tensor = torch.ones([len(batch), seq_len], dtype=torch.int64)*wordtoix['_']
        for i, (iseq, tseq) in enumerate(batch):
            for j in range(len(iseq)):
                input_tensor[i,j] = iseq[j]
                target_tensor[i,j] = tseq[j]
        return input_tensor, target_tensor
    
class SCANResplitAE(Dataset):
    words = {'after': 0, 'and': 1, 'around': 2, 'jump': 3, 'left': 4, 
                'look': 5, 'opposite': 6, 'right': 7, 'run': 8, 'thrice': 9, 
                'turn': 10, 'twice': 11, 'walk': 12, 'I_JUMP': 13, 'I_LOOK': 14, 
                'I_RUN': 15, 'I_TURN_LEFT': 16, 'I_TURN_RIGHT': 17, 'I_WALK': 18, 
                '_': 19, '<EOS>': 20, '<PAD>':21}
    ixtoword = list(words.keys())
    def __init__(self, dset: str, len_range, train_proprtion: float = 0.9,
                 cache_dir: str = "./cache/scan_resplit"):
        assert dset in ["train", "test", "all"]
        self.resplitdset = ScanLengthResplit(dset, len_range, train_proprtion, cache_dir)

        self.wordtoix = SCANResplitAE.words
        #print(self.vocab.words)
        
        self.vocab_size = len(SCANResplitAE.words)
    def __getitem__(self, index):
        item = self.resplitdset[index]
        in_str = self.resplitdset.in_vocabulary.indices_to_sentence(item['in'])
        out_str = self.resplitdset.out_vocabulary.indices_to_sentence(item['out'])
        eosid = SCANResplitAE.words['<EOS>']
        
        
        ae_in = [SCANResplitAE.words[c] for c in in_str] + [eosid] + [SCANResplitAE.words['_'] for _ in out_str] + [eosid]
        ae_out = [SCANResplitAE.words[c] for c in in_str] + [eosid] + [SCANResplitAE.words[c] for c in out_str] + [eosid]

        return ae_in, ae_out

    def __len__(self):
        return len(self.resplitdset)
    def collate_batch(batch):
        seq_len = max([len(seq) for seq, _ in batch])
        input_tensor = torch.ones([len(batch), seq_len], dtype=torch.int64)*SCANResplitAE.words['<PAD>']
        target_tensor = torch.ones([len(batch), seq_len], dtype=torch.int64)*SCANResplitAE.words['<PAD>']
        for i, (iseq, tseq) in enumerate(batch):
            for j in range(len(iseq)):
                input_tensor[i,j] = iseq[j]
                target_tensor[i,j] = tseq[j]
        return input_tensor, target_tensor
if __name__ == '__main__':
    #dataset     = SCANDatasetAE('SCAN/length_split/tasks_train_length.txt') 
    dataset = SCANResplitAE('train',(0,22))
    loader = DataLoader(dataset, batch_size=4, collate_fn=SCANResplitAE.collate_batch)
    for i in range(2):
        idx = np.random.randint(0,len(dataset))
        x,y = dataset.__getitem__(idx)
        print(x)
        print([SCANResplitAE.ixtoword[d] for d in x])
        #print([])
        #print(dataset.vocab.indices_to_sentence(x))
        print(y)
        #print(dataset.vocab.indices_to_sentence(y))
        
    for x,y in loader:
        print(x[0])
        print(y[0])
        break
