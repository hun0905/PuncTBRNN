from nltk.util import pr
import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
import torch.nn.functional as F
class bi_LSTM(nn.Module):
    def __init__(self,vocab_len,vector_len,hidden_size,num_layers,punc_num):#4979 , 256 , 512 , 2 , 4  batchc size 128
        super(bi_LSTM,self).__init__() #繼承父類nn.Module 的特性
        self.vocab_len = vocab_len
        self.vector_len = vector_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.punc_num = punc_num
        self.embedding = nn.Embedding(self.vocab_len,self.vector_len) #將原始的one-hot 映射到低維度向量表達，降低特徵維度
        self.lstm = nn.LSTM(self.vector_len,self.hidden_size,self.num_layers,bidirectional = True,batch_first = True)
        self.linaer = nn.Linear(hidden_size*2,self.punc_num) #hidden_size*2 因為雙向 , punc_num 是標點的種類加1 ,因為要含' '也就是無標點
    
    def forward(self,input,input_length):
        input = self.embedding(input)
    
        input_length = input_length.view(-1)
        pack_input = pack_padded_sequence(input,input_length.cpu(),batch_first=True)
        #print(input.size())
        pack_out,(h_n,c_n) = self.lstm(pack_input)
        length = input.size(1)
        #print(h_n.size()) # [num_layers*2(bi),batch_szie,vector_len]
        #fbhn = (h_n[-2,:,:]+h_n[-1,:,:]).unsqueeze(0) #最後一層的forward 和 backward lstm
        #print(pack_out.data.size()) #[seq_len*batch_size,hidden_size*2(bi)] 一般output 是 [batch_size,seq_len,hidden_size*2(bi)]
        #fbout = output[:, :, :self.hidden_dim]+ output[:, :, self.hidden_dim:] 
        #padded_input: padded batch of variable length sequences.;input_length: list of sequence lengths of each batch element (must be on the CPU if provided as a tensor).
        #input can be of size T x B x * where T is the length of the longest sequence (equal to lengths[0]), B is the batch size
        #and * is any number of dimensions (including 0). If batch_first is True, B x T x * input is expected.
        
        output,_ = pad_packed_sequence(pack_out,batch_first=True,total_length=length) 
        output = self.linaer(output) 
        return output
    @classmethod
    def load_model(cls, path):
        # Load to CPU
        package = torch.load(path, map_location=lambda storage, loc: storage)########
        model = cls.load_model_from_package(package)
        return model

    @classmethod
    def load_model_from_package(cls, package):
        model = cls(package['vocab_len'], package['vector_len'],
                    package['hidden_size'], package['num_layers'],
                    package['punc_num'])
        model.load_state_dict(package['state_dict'])
        return model
        
    def serialize(self,model, optimizer, epoch,train_loss,val_loss):
        package = {
            # hyper-parameter
            'vocab_len': model.vocab_len,
            'vector_len': model.vector_len,
            'hidden_size': model.hidden_size,
            'num_layers': model.num_layers,
            'punc_num': model.punc_num,
            # state
            'state_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict(),
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss':val_loss
        }
        return package
