import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class StackRNN(nn.Module):

  def __init__(self, idim, hdim, odim, stsize=64):
    super(StackRNN, self).__init__()
    self.size = hdim
    self.stsize = stsize
    self.i2h = nn.Linear(idim, hdim)
    self.h2h = nn.Linear(hdim, hdim)
    self.h2o = nn.Linear(hdim, odim)
    self.h2a = nn.Linear(hdim, 3*stsize)
    self.h2s = nn.Linear(hdim, stsize)
    self.s2h0 = nn.Linear(stsize, hdim)
    self.s2h1 = nn.Linear(stsize, hdim)
    self.drop = nn.Dropout(0.1)
    

  def forward(self, input, hidden, stack):
    st0 = stack[:,0]
    st1 = stack[:,1]
    hidden = torch.sigmoid(self.i2h(input) + self.h2h(hidden) + self.s2h0(st0) + self.s2h1(st1))
    act  = torch.softmax(self.h2a(hidden).view(-1,3,self.stsize),dim=-2)
    top = torch.sigmoid(self.h2s(hidden)).view(-1,1,self.stsize)
    push = torch.cat((top, stack[:,:-1]),dim=-2)
    empty = torch.FloatTensor(input.size(0),1,self.stsize).fill_(-1)
    if torch.cuda.is_available():
      empty = empty.cuda()
    pop = torch.cat((stack[:,1:], empty),dim=-2)
    stack = push*act[:,0][:,None,:] + pop*act[:,1][:,None,:] + stack*act[:,2][:,None,:]
    out = self.h2o(self.drop(hidden))
    return out, hidden, stack
    
  def initHiddenAndStack(self, bsize, stackdepth):
    hidden = torch.FloatTensor(bsize,self.size).fill_(0)
    stack = torch.FloatTensor(bsize,stackdepth, self.stsize).fill_(0)
    if torch.cuda.is_available():
      hidden = hidden.cuda()
      stack = stack.cuda()
    return hidden, stack


  
class StackRNNAE(nn.Module):
  def __init__(self, d_model=256, vocab_size=16, nhead=4, num_layers=2, mem_size=64):
    super().__init__()

    self.stackrnn = StackRNN(d_model, d_model, d_model, stsize=mem_size)
    self.d_model=d_model
    self.vocab_size = vocab_size
    assert d_model%2 == 0
    self.embedding = nn.Embedding(vocab_size, d_model)

    self.fc = nn.Linear(d_model, vocab_size)
  #Batch-first in (N,S), batch-first out (N,C,S)
  def forward(self, inputs):
    hidden, stack = self.stackrnn.initHiddenAndStack(inputs.size(0), inputs.size(1))
    embedded = self.embedding(inputs.permute(1,0)) #S,N,C
    outputs = []
    for input in embedded:
      out, hidden, stack = self.stackrnn(input, hidden, stack)
      outputs.append(out)
    # SXN,C -> S,N,C
    stacked = torch.stack(outputs,dim=0)
    output = self.fc(F.relu(stacked)).permute(1,2,0)
    return output