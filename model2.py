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
        #model的weight和bias初始化，weight 用 Glorot initialization  bias則初使為0
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
        if pre_trained:#使否用word pretrained vector
            self.embedding = nn.Embedding(input_size,embedding_size)
            #載入pretrain embedding ,這裡採用的是以FastText訓練wiki corpus所得到的pretrained embedding
            weight = torch.load('/home/yunghuan/Desktop/PuncTBRNN/ChineseFastText.pth')
            self.embedding.load_state_dict({'weight':weight})
            
            #設定載入的weight可以進行梯度更新
            self.embedding.weight.requires_grad = True
            
        else:
            self.embedding = nn.Embedding(input_size,embedding_size)
            
        self.rnn = nn.GRU(embedding_size,hidden_size,num_layers,bidirectional = True)
        self.fc_hidden = nn.Linear(hidden_size*2,hidden_size*2) #because it is bidirection, so *2.
        self.output_size = output_size
        self.predict = nn.Linear(hidden_size*2,self.output_size)
        self.apply(self.weight_init)
    def forward(self,x,length):
        #print(x.size())
        
        length = length.view(-1)#句子長度

        embedding = self.dropout( self.embedding(x) ) #單字的idx經過embedding後輸出word vector然後在經過dropout,但此處dropout預設是0,所以無效
        
        #pack_padded_sequence() 是用來壓縮序列的，而 pad_packed_sequence() 則是用來展開序列成原本形狀的
        #pack_padded_sequence : 將一個batch中不同長度的padded的句子包裝在一起，這裡會輸出維度（seqlen(max) , batch_size , *）
        pack_input = pack_padded_sequence(embedding,length.cpu(),batch_first=True)

        #將packed的句子放入rnn，h_t是每個time step的輸出在這裡也就是我們句子中每個單字的最後一層輸出,維度是（seq_len,batch_size,hidden_size*2）
        # *2是因為是bidirectional所以會把正向和逆向的最後一層的hidden_size拼在一起
        h_t,_ = self.rnn(pack_input)

        #句子長度
        length = x.size(1)
       
        h_t ,_ = pad_packed_sequence(h_t,total_length=length,batch_first=True) 
        return h_t
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
    def load_model(cls, path): #使用load_model可以載入已經訓練好的model
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
    def weight_init(m):#初始化，同encoder
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
       
        #這裡因為一次指處理一個字所以沒有使用GRU,GRUCell也就是指輸入一個time step就可（就是沒有Seq_len的維度,因為必為1），input的形式(input_size, hidden_size）
        self.rnn_cell = nn.GRUCell(self.hidden_size*2,self.hidden_size)
         
        #在attention的過程會經過許多線性轉換，詳細參考論文的用，以下維各種線性轉換
        self.va = nn.Linear(self.hidden_size,self.hidden_size*2)
        self.we = nn.Linear(self.hidden_size*2,1)
        self.wf_a = nn.Linear(self.hidden_size*2,self.hidden_size)
        self.wf_f = nn.Linear(self.hidden_size,self.hidden_size,bias=False)
        self.wf_h = nn.Linear(self.hidden_size,self.hidden_size)
        self.w_y = nn.Linear(self.hidden_size,self.output_size)

        #weight,bias初始化
        self.apply(self.weight_init)
    def forward(self,x_t,projected_context,context,state = None):
        sequence_length = projected_context.size()[0]
        
        #h_a (seq_len,batch_size,hidden_size*2)
        h_a = torch.tanh(projected_context+self.va( state ) ) #h_a 維度（）
        alphas = self.we(h_a) #alphas : (seq_len,batch_size,1)
        alphas = torch.exp(alphas)# alphas : (seq_len,batch_size,1)
        alphas = torch.squeeze(alphas,2) #alphas : (seq_len,batch_size)
        a_sum =  torch.sum(alphas,dim=0).unsqueeze(0).repeat(sequence_length,1) #a_sum : (seq_len,batch_size)
        alphas = alphas/a_sum #alphas : (seq_len,batch_size)
        alphas = torch.unsqueeze(alphas,2) #alphas : (seq_len,batch_size,1)
 
        weight_context = (context * alphas.repeat(1,1,self.hidden_size*2)).sum(axis=0) # (batch_size,hidden_size*2)
        new_state = self.rnn_cell(x_t,state) #new_state : (batch_size , hidden_size )
        

        lfc = self.wf_a(weight_context) # lfc : (batch_size,hidden_size)
        fw = torch.sigmoid( self.wf_f(lfc)+ self.wf_h(new_state) ) #fw : (batch_size,hidden_size)
        
        hf_t = lfc*fw+new_state #hf_t : (batch_size,hidden_size)
        #z = self.w_y(new_state)
        z = self.w_y(hf_t) #z : (batch_size,output_size)
        
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
        outputs = torch.zeros(batch_size,target_len,target_vocab_size).to(device) #用來存decoder的結果
        context = self.encoder(source,length) # context : (batch_size,seq_len,hidden_size*2)
        context = context.transpose(0,1) # context : (seq_len,batch_size,hidden_size*2)
        projected_context = self.projected(context) # projected_context : (seq_len,batch_size,hidden_size*2)
        new_state = torch.zeros( batch_size,self.hidden_size).cuda()
        for t in range(1,target_len):
            
            #從1開始(0必定無標點沒意義)將context(encoder的編碼)輸入，然後為了得到seq中每個字跟其他的的關聯，
            #所以會將context中的每個字跟 context（整個句子的編碼）和projected context些加權
            #計算，而new_state則是從GRUcell傳出的state,就是要在下一次調用decoder時可以讓其獲得
            #GRUcell再經過上次解每後的狀態改變

            #context[t,:,:]:  (batch_size,hidden_size*2)
            #context : (seq_len,batch_size,hidden_size*2)
            #projected_context : (seq_len,batch_size,hidden_size*2)
            output , new_state= self.decoder(context[t,:,:],projected_context,context,new_state)

            #output :(batch_size,output_size)
            #new_state : (batch_size , hidden_size )
            outputs[:,t,:] = output 
            
        
        return outputs
    @classmethod
    def load_model(cls, path): #載入model
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
    
    def serialize(self,model, optimizer, epoch,train_loss,val_loss): #儲存model所有需要的參數和state
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