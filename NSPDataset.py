import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

'''
# 16-dim one-hot vectors
# 0-9:  decimal digits
# 10:   Delimiter (space)
# 11:   Pad
# 12:   Start
# 13:   EOS
# 14:   Mask
# 15:   Custom token    
'''

class Token():
    delim = 10 
    pad = 11
    start = 12
    eos = 13
    mask = 14
    custom = 15

def fib(seed1, seed2, numbers):
    seq = [seed1, seed2]
    for i in range(2, numbers):
        seq.append(seq[i-2]+seq[i-1])
    target = seq[-2]+seq[-1]
    return seq, target

def arith(seed1, seed2, numbers):
    seq = [seed1, seed2] if seed1<seed2 else [seed2,seed1]
    for i in range(2, numbers):
        seq.append(2*seq[i-1]-seq[i-2])
    target = 2*seq[-1]-seq[-2]
    return seq, target

def count(seed1, seed2, numbers):
    seq = [seed1]
    diff = seed2%10+10
    for i in range(1, numbers):
        seq.append(seq[i-1]+diff)
    target = seq[-1]+diff
    return seq, target

def palindrome(seed1, seed2, numbers):
    seq = [seed2]
    for i in range(1, numbers):
        nstr = str(seq[i-1])
        seq[i] = int(nstr[::-1])
    target = int(str(seq[-1])[::-1])
    return seq, target

def copy(seed1, seed2, numbers):
    seq = [seed2]
    for i in range(1, numbers):
        seq.append(seed2)
    target = seed2
    return seq, target

def num2vec(num, ndigits, lendian=True):
    digits = [int(c) for c in str(num)]
    if len(digits)<ndigits:
        digits = [0 for i in range(ndigits-len(digits))]+digits
    elif len(digits)>ndigits:
        digits = digits[len(digits)-ndigits:]
    if lendian:
        digits.reverse()
    return np.array(digits)

#Not zero-padded
def num2vec2(num, lendian=True):
    digits = [int(c) for c in str(num)]
    if lendian:
        digits.reverse()
    return np.array(digits)

#Dataset class for autoencoding setup (CNN, BERT, etc)
class NSPDatasetAE(Dataset):
    def __init__(self, rule, maxdigits, mindigits=1, numbers=2, size=25600, lendian=True):
        self.rule = rule
        assert maxdigits > mindigits
        self.maxdigits = maxdigits
        self.mindigits = mindigits
        self.size = size
        self.lendian = lendian
        self.numbers = numbers
        self.maxlen = (maxdigits+1)*(numbers+1) + 1
        self.inputs = np.zeros([size, self.maxlen, 16], dtype=np.float32)
        self.targets = np.ones([size, self.maxlen], dtype=np.int64)*Token.pad
        self.iscreated = [False for i in range(size)]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if not self.iscreated[idx]:
            ndigits = ((self.maxdigits-self.mindigits+1)*idx)//self.size + self.mindigits
            seed1 = np.random.randint(1, 10**ndigits, dtype=np.int64)
            seed2 = np.random.randint(max(seed1,10**(ndigits-1)), 10**ndigits, dtype=np.int64)
            seq, target = self.rule(seed1, seed2, self.numbers)
            pos = 1
            self.inputs[idx][0][Token.delim] = 1
            self.targets[idx][0] = Token.delim
            
            for i in range(self.numbers):
                vec = num2vec(seq[i], ndigits, self.lendian)
                for j,v in enumerate(vec):
                    self.inputs[idx][pos+j][v] = 1
                    self.targets[idx][pos+j] = v
                self.inputs[idx][pos+ndigits][Token.delim] = 1
                self.targets[idx][pos+ndigits] = Token.delim
                pos = pos + ndigits + 1

            y = num2vec(target, ndigits, self.lendian)

            self.inputs[idx][pos:pos+len(y),Token.mask] = 1
            self.targets[idx][pos:pos+len(y)] = y
            self.inputs[idx][pos+len(y)][Token.eos] = 1
            self.targets[idx][pos+len(y)] = Token.eos

            if pos+len(y)+1 < self.maxlen:
                self.inputs[idx][pos+len(y)+1:,Token.pad] = 1
            
            self.iscreated[idx] = True


        return self.inputs[idx], self.targets[idx]

#Dataset class for autoencoding setup (CNN, BERT, etc)
class NSPDatasetAE2(Dataset):
    def __init__(self, rule, maxdigits, mindigits=1, numbers=2, size=25600, lendian=True):
        self.rule = rule
        assert maxdigits > mindigits
        self.maxdigits = maxdigits
        self.mindigits = mindigits
        self.size = size
        self.lendian = lendian
        self.numbers = numbers
        self.maxlen = (maxdigits+1)*(numbers+1) + 2
        self.inputs = np.ones([size, self.maxlen], dtype=np.int64)*Token.pad
        self.targets = np.ones([size, self.maxlen], dtype=np.int64)*Token.pad
        self.iscreated = [False for i in range(size)]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if not self.iscreated[idx]:
            ndigits = ((self.maxdigits-self.mindigits+1)*idx)//self.size + self.mindigits
            s1digits = np.random.randint(1, ndigits+1, dtype=np.int64)
            seed1 = np.random.randint(1, (10**s1digits)-1, dtype=np.int64)
            seed2 = np.random.randint(10**(ndigits-1), 10**ndigits, dtype=np.int64)
            seq, target = self.rule(seed1, seed2, self.numbers)
            pos = 1
            self.inputs[idx][0] = Token.delim
            self.targets[idx][0] = Token.delim
            
            for i in range(self.numbers):
                vec = num2vec2(seq[i], self.lendian)
                for j,v in enumerate(vec):
                    self.inputs[idx][pos+j] = v
                    self.targets[idx][pos+j] = v
                self.inputs[idx][pos+len(vec)] = Token.delim
                self.targets[idx][pos+len(vec)] = Token.delim
                pos = pos + len(vec) + 1

            y = num2vec2(target, self.lendian)

            self.inputs[idx][pos:pos+len(y)] = Token.mask
            self.targets[idx][pos:pos+len(y)] = y
            self.inputs[idx][pos+len(y)] = Token.eos
            self.targets[idx][pos+len(y)] = Token.eos

            if pos+len(y)+1 < self.maxlen:
                shift = np.random.randint(0, self.maxlen-pos-len(y)-1, dtype=np.int64)
                curinput = self.inputs[idx][:pos+len(y)+1].copy()
                curtarget = self.targets[idx][:pos+len(y)+1].copy()
                self.inputs[idx][:shift] = Token.pad
                self.inputs[idx][shift:shift+pos+len(y)+1] = curinput
                self.targets[idx][:shift] = Token.pad
                self.targets[idx][shift:shift+pos+len(y)+1] = curtarget
            
            self.iscreated[idx] = True


        return self.inputs[idx], self.targets[idx]

#Seq2Seq version + no one-hot encoding
class NSPDatasetS2S(Dataset):
    def __init__(self, rule, maxdigits, mindigits=1, size=25600, lendian=True):
        self.rule = rule
        assert maxdigits > mindigits
        self.maxdigits = maxdigits
        self.mindigits = mindigits
        self.size = size
        self.lendian = lendian
        self.maxlen = (maxdigits+1)*2 + 1
        self.inputs = np.ones([size, self.maxlen], dtype=np.int64)*Token.pad
        self.targets = np.ones([size, self.maxdigits+1], dtype=np.int64)*Token.pad
        self.iscreated = [False for i in range(size)]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if not self.iscreated[idx]:
            ndigits = ((self.maxdigits-self.mindigits+1)*idx)//self.size + self.mindigits
            ndigits2 = np.random.randint(1,ndigits+1)
            seed1 = np.random.randint(10**(ndigits-1), 10**ndigits, dtype=np.int64)
            seed2 = np.random.randint(10**(ndigits2-1), 10**ndigits2, dtype=np.int64)
            seq, target = self.rule(seed1, seed2, 2)
            pos = 1
            self.inputs[idx][0] = Token.delim

            vec = num2vec(seq[0], ndigits, self.lendian)
            self.inputs[idx][pos:pos+ndigits] = vec
            self.inputs[idx][pos+ndigits] = Token.delim
            pos = pos + ndigits + 1

            vec = num2vec(seq[1], ndigits2, self.lendian)
            self.inputs[idx][pos:pos+ndigits2] = vec
            self.inputs[idx][pos+ndigits2] = Token.delim
            pos = pos + ndigits2 + 1
            if target < 10**ndigits:
                self.targets[idx][:ndigits] = num2vec(target, ndigits, self.lendian)
            else:
                self.targets[idx][:ndigits+1] = num2vec(target, ndigits+1, self.lendian)            
            self.iscreated[idx] = True
        return self.inputs[idx], self.targets[idx]

#exclusive for copy and palindrome datasets
class StringDataset(Dataset):
    def __init__(self, rule, maxdigits, mindigits=1, size=25600):
        assert rule in ['copy', 'palin']
        self.reverse = rule == 'palin'
        assert maxdigits > mindigits
        self.maxdigits = maxdigits
        self.mindigits = mindigits
        self.size = size
        self.maxlen = (maxdigits+1)*(2) + 1
        self.inputs = np.ones([size, self.maxlen], dtype=np.int64)*Token.pad
        self.targets = np.ones([size, self.maxlen], dtype=np.int64)*Token.pad
        self.iscreated = [False for i in range(size)]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if not self.iscreated[idx]:
            ndigits = ((self.maxdigits-self.mindigits+1)*idx)//self.size + self.mindigits
            seq = np.random.randint(0, 10, [ndigits], dtype=np.int64)
            
            ans = seq[::-1] if self.reverse else seq
            #random shift
            pos = np.random.randint(1, self.maxlen - 2*ndigits - 1)
            
            self.inputs[idx][pos-1] = Token.delim
            self.targets[idx][pos-1] = Token.delim
            
            self.inputs[idx][pos:pos+ndigits] = seq
            self.targets[idx][pos:pos+ndigits] = seq
            
            self.inputs[idx][pos+ndigits] = Token.delim
            self.targets[idx][pos+ndigits] = Token.delim

            pos = pos + ndigits + 1
            self.inputs[idx][pos:pos+ndigits] = ans
            self.targets[idx][pos:pos+ndigits] = Token.mask
            
            self.inputs[idx][pos+ndigits] = Token.eos
            self.targets[idx][pos+ndigits] = Token.eos
            
            self.iscreated[idx] = True

        return self.inputs[idx], self.targets[idx]

class ReductionDatasetAE(Dataset):
    def __init__(self, maxdigits, mindigits=1, size=6400):
        assert maxdigits > mindigits
        self.maxdigits = maxdigits
        self.mindigits = mindigits
        self.size = size
        #vocab: [0-9],<PAD>,<MASK>,<EOS>
        self.vocab_size = 13
        self.inputs = np.ones([size, self.maxdigits*2+1], dtype=np.int64)*10
        self.targets = np.ones([size, self.maxdigits*2+1], dtype=np.int64)*10
        for idx in range(size):
            seqlen = ((self.maxdigits-self.mindigits+1)*idx)//self.size + self.mindigits
            #seqlen = self.maxdigits
            #seqlen = np.random.randint(ndigits, self.maxdigits+1)
            #nonzeros = np.random.randint(self.mindigits, self.maxdigits)
            nonzeros = np.random.randint(self.mindigits, seqlen+1)
            nzidxs = np.random.permutation(np.arange(seqlen))[:nonzeros]
            seq = [0 for _ in range(seqlen)]
            for ni in nzidxs:
                seq[ni] = np.random.randint(1,10)
            offset = np.random.randint(0, self.maxdigits-seqlen+1)
            for i in range(seqlen):
                self.inputs[idx][i+offset] = seq[i]
                #self.targets[idx][i+offset] = seq[i]
            self.inputs[idx][seqlen+offset] =  12 #<EOS>
            self.targets[idx][seqlen+offset] =  12 #<EOS>
            tidx = seqlen+1+offset
            for s in seq:
                if s > 0:
                    self.targets[idx][tidx] = s
                    tidx = tidx+1
    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

#100100011 -> 1111
class ReductionDatasetAE2(Dataset):
    def __init__(self, maxdigits, mindigits=1, size=6400):
        assert maxdigits > mindigits
        self.maxdigits = maxdigits
        self.mindigits = mindigits
        self.size = size
        #vocab: [0-9],<DELIM>, <Pad>
        self.vocab_size = 12
        self.inputs = np.ones([size, self.maxdigits*2+1], dtype=np.int64)*(self.vocab_size-1)
        self.targets = np.ones([size, self.maxdigits*2+1], dtype=np.int64)*(self.vocab_size-1)
        self.iscreated = [False for i in range(size)]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if not self.iscreated[idx]:
            ndigits = ((self.maxdigits-self.mindigits+1)*idx)//self.size + self.mindigits
            seq = [np.random.randint(2)*np.random.randint(1,10) for _ in range(ndigits)]
            
            pos = np.random.randint(self.maxdigits-ndigits+1)
            for i in range(pos, pos+ndigits):
                self.inputs[idx][i] = seq[i-pos]
            self.inputs[idx][pos+ndigits] = 10 #delim
            tidx = pos+ndigits+1
            for s in seq:
                if s > 0:
                    self.targets[idx][tidx] = s
                    tidx = tidx+1         
            self.iscreated[idx] = True
        return self.inputs[idx], self.targets[idx]

def printseq(x,y):
    tokenmap = ['0','1','2','3','4','5','6','7','8','9','_',' ','S','E','M','C']
    print("input:")
    xseq = np.argmax(x,-1)
    print('\t' + ' '.join([tokenmap[n] for n in xseq]))
    print("target:")
    print('\t' + ' '.join([tokenmap[n] for n in y]))

def printseq2(x,y):
    tokenmap = ['0','1','2','3','4','5','6','7','8','9','_',' ','S','E','M','C']
    print("input:")
    print('\t' + ' '.join([tokenmap[n] for n in x]))
    print("target:")
    print('\t' + ' '.join([tokenmap[n] for n in y]))

if __name__ == '__main__':
    
    #dataset = NSPDatasetAE2(copy,5,1, numbers=2)
    #dataset = NSPDatasetS2S(fib,12,2)
    #dataset = StringDataset('copy', 5, 1, size=256)
    dataset = ReductionDatasetAE2(12,8,size=512)
    loader = DataLoader(dataset, batch_size=4)
    for i in range(10):
        idx = np.random.randint(0,len(dataset))
        x,y = dataset.__getitem__(idx)
        printseq2(x,y)
        # print(np.argmax(x,-1))
        # print(y)