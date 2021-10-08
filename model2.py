import math
from nltk.util import pr
from numpy.core.fromnumeric import argmax
import torch
import random
from torch import nn
from torch._C import device
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.modules import dropout
import numpy as np
from torch.nn.modules.sparse import Embedding
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class Encoder(nn.Module):
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Embedding):
            nn.init.xavier_uniform_(m.weight)
        elif isinstance(m,nn.GRU) or isinstance(m,nn.LSTM):
            for name, param in m.named_parameters():
	            if name.startswith("weight"):
		            nn.init.xavier_uniform_(param)
	            else:
	            	nn.init.zeros_(param)
        
    def __init__(self,input_size,embedding_size,hidden_size,output_size,num_layers,p,pre_trained):
        super(Encoder,self).__init__()
        self.pre_trained = pre_trained
        self.p = p
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.dropout = nn.Dropout(p)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        if pre_trained:
            self.embedding = nn.Embedding(input_size,embedding_size)
            weight = torch.load('/home/yunghuan/Desktop/PuncTBRNN/ChineseFastText.pth')
            
            self.embedding.load_state_dict({'weight':weight})
            self.embedding.weight.requires_grad = True
            #self.embedding = nn.Embedding.from_pretrained(weight)
        else:
            self.embedding = nn.Embedding(input_size,embedding_size)
            #print(self.embedding.weight[0][0].type())
        self.rnn = nn.GRU(embedding_size,hidden_size,num_layers,bidirectional = True)
        self.fc_hidden = nn.Linear(hidden_size*2,hidden_size*2) #because it is bidirection, so *2.
        self.output_size = output_size
        self.predict = nn.Linear(hidden_size*2,self.output_size)
        self.apply(self.weight_init)
    def forward(self,x,length):
        #print(x.size())
        length = length.view(-1)
        embedding = self.dropout( self.embedding(x) )
        #embedding = x
        
        pack_input = pack_padded_sequence(embedding,length.cpu(),batch_first=True)
        h_t,_ = self.rnn(pack_input)
        length = x.size(1)
       
        h_t ,_ = pad_packed_sequence(h_t,total_length=length,batch_first=True) 
        y_t = self.predict(h_t)
        return h_t,y_t
    def serialize(self,model, optimizer, epoch,train_loss,test_loss):
        package = {
            # hyper-parameter
            'input_size':model.input_size,
            'embedding_size':model.embedding_size,
            'encoder_hidden_size':model.hidden_size,
            'encoder_output_size':model.output_size,
            'encoder_num_layers':model.num_layers,
            'encoder_dropout':model.p,
            'pre_trained':model.pre_trained,
            # state
            'encoder_state':model.state_dict(),
            'optim_dict': optimizer.state_dict(),
            'epoch': epoch,
            'train_loss': train_loss,
            'test_loss':test_loss
        }
        return package
    @classmethod
    def load_model(cls, path):
        # Load to CPU
        package = torch.load(path, map_location=lambda storage, loc: storage)########
        model = cls.load_encoder_from_package(package)
        return model

    @classmethod
    def load_encoder_from_package(cls, package):
        encoder = cls(package['input_size'], package['embedding_size'],package['encoder_hidden_size'],package['encoder_output_size'],
                    package['encoder_num_layers'],package['encoder_dropout'],package['pre_trained'] )
        encoder.load_state_dict(package['encoder_state'])
        return encoder
    
class Decoder(nn.Module):
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Embedding):
            nn.init.xavier_uniform_(m.weight)
        elif isinstance(m,nn.GRU) or isinstance(m,nn.LSTM):
            for name, param in m.named_parameters():
	            if name.startswith("weight"):
		            nn.init.xavier_uniform_(param)
	            else:
	            	nn.init.zeros_(param)
        
    def __init__(self,hidden_size,output_size,num_layers,p):
        super(Decoder,self).__init__()
        self.p = p
        self.dropout = nn.Dropout(p)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.num_layers = num_layers
        #hidden_size*2+embedding_size : because we have encoder with forward and backwrad
        self.rnn_cell = nn.GRUCell(self.hidden_size*2,self.hidden_size)
        # hidden_size*3 : add hidden state from the encoder and take hidden from previous state from our deccoder
        self.va = nn.Linear(self.hidden_size,self.hidden_size*2)
        self.we = nn.Linear(self.hidden_size*2,1)
        self.wf_a = nn.Linear(self.hidden_size*2,self.hidden_size)
        self.wf_f = nn.Linear(self.hidden_size,self.hidden_size,bias=False)
        self.wf_h = nn.Linear(self.hidden_size,self.hidden_size)
        
        self.w_y = nn.Linear(self.hidden_size,self.output_size)
        
        #encoder_states ; hidden from encoder ; hidden : from decoder
    def forward(self,x_t,projected_context,context,state = None): #x :1, batch_size ; encoder_states :  seq_len , batch_size , 2*hidden_size  cell : num_layers , batch_size , hidden_size * 2
        #print(x_t.size())
        sequence_length = projected_context.size()[0]
        if state == None:
            h_a = torch.tanh(projected_context )
            #new_state = self.rnn_cell(dec_input)
        else:
            h_a = torch.tanh(projected_context+self.va( state ) )
            #new_state = self.rnn_cell(dec_input,state)
        
     
        alphas = self.we(h_a)
        alphas = torch.exp(alphas)
        alphas = torch.squeeze(alphas,2)
        a_sum =  torch.sum(alphas,dim=0).unsqueeze(0).repeat(sequence_length,1)
        alphas = alphas/a_sum
        alphas = torch.unsqueeze(alphas,2) #94 , 16
 
        weight_context = (context * alphas.repeat(1,1,self.hidden_size*2)).sum(axis=0) # (40,64,512) (40.64.1) *不是dot product
        new_state = self.rnn_cell(x_t,state)
        

        lfc = self.wf_a(weight_context)
        fw = torch.sigmoid( self.wf_f(lfc)+ self.wf_h(new_state) )
        
        hf_t = lfc*fw+new_state
        #z = self.w_y(new_state)
        z = self.w_y(hf_t)
        
        return z,new_state
    @classmethod
    def load_decoder_from_package(cls, package):
        decoder =cls( package['decoder_hidden_size'],package['decoder_output_size'],package['decoder_num_layers']
                            ,package['decoder_dropout'])
        decoder.load_state_dict(package['decoder_state'])
        return decoder
    
class Seq2Seq(nn.Module):
    def __init__(self,encoder,decoder,hidden_size):
        super(Seq2Seq,self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.hidden_size = hidden_size
        self.projected = nn.Linear(hidden_size*2,hidden_size*2)
        self.predict = nn.Linear(hidden_size*2,4)
        nn.init.xavier_uniform_(self.projected.weight)
        nn.init.zeros_(self.projected.bias)
    def forward(self,source,length,teacher_force_ratio=0.5):
        #my source and target : batch_size , seq_len 
        #print(source.size())
        source = source
        batch_size = source.shape[0]
        target_len = source.shape[1]
        
        target_vocab_size = self.encoder.output_size #number of classes
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        outputs = torch.zeros(batch_size,target_len,target_vocab_size).to(device) 
        context,enc = self.encoder(source,length) #st , ht
        context = context.transpose(0,1)
        projected_context = self.projected(context)
        new_state = None
        
        #print(enc.size()) #32 58 4
        for t in range(1,target_len):
            #send our encoder_states to our decoder at every time step
            output , new_state= self.decoder(context[t,:,:],projected_context,context,new_state)
            outputs[:,t,:] = output
            
        
        return outputs
    @classmethod
    def load_model(cls, path):
        # Load to CPU
        package = torch.load(path, map_location=lambda storage, loc: storage)########
        model = cls.load_model_from_package(package)
        return model

    @classmethod
    def load_model_from_package(cls, package):

        model = cls(package['encoder'],package['decoder'],package['hidden_size'])
        model.decoder = Decoder.load_decoder_from_package(package)
        model.encoder = Encoder.load_encoder_from_package(package)
        model.load_state_dict(package['state_dict'])
        
        return model
    
    def serialize(self,model, optimizer, epoch,train_loss,val_loss):
        package = {
            # hyper-parameter
            'decoder': model.decoder,
            'encoder': model.encoder,
            'input_size':model.encoder.input_size,
            'embedding_size':model.encoder.embedding_size,
            'encoder_hidden_size':model.encoder.hidden_size,
            'encoder_output_size':model.encoder.output_size,
            'pre_trained':model.encoder.pre_trained,
            'encoder_num_layers':model.encoder.num_layers,
            'encoder_dropout':model.encoder.p,
            'pre_trained':model.encoder.pre_trained,
            'decoder_hidden_size':model.decoder.hidden_size,
            'decoder_output_size':model.decoder.output_size,
            'decoder_num_layers':model.decoder.num_layers,
            'decoder_dropout':model.decoder.p,
            'hidden_size':model.hidden_size,
            # state
            'encoder_state':model.encoder.state_dict(),
            'decoder_state':model.decoder.state_dict(),
            'state_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict(),
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss':val_loss
        }
        return package