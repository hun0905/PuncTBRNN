from torch.utils import data
from torch.utils.data import dataset
import torch
from sklearn.model_selection import KFold
import time
import numpy as np
from dataset import PuncDataset
from dataset import Collate
from model2 import Decoder,Encoder,Seq2Seq
from model import bi_LSTM
from torchvision.transforms import ToTensor
from torch.utils.data import  DataLoader,random_split,SubsetRandomSampler
from sklearn.metrics import precision_recall_fscore_support as score
from torch.optim.lr_scheduler import ExponentialLR
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

with_att = False
stage = 3
pretrained = True
K_fold = False
class Train():
    def __init__(self,dataset, model, criterion, optimizer,use_cuda,batch_size,epochs,scheduler,punc_path,save_path = 'punc_model',collate_fn = None,is_continue=False,\
                num_worker = 8,batch_size_times = 1,pin_memory = False,k = 10,with_l1= False,with_l2=False,l1_weight = 0,l2_weight=0,K_fold = False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        k = 10
        self.start = 0
        self.epochs = epochs
        self.dataset = dataset
        self.epoch = 0
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.use_cuda = use_cuda
        self.batch_size = batch_size
        self.scheduler = scheduler
        self.save_path = save_path
        self.collate_fn = collate_fn
        self.is_continue = is_continue
        self.num_worker = num_worker
        self.batch_size_times = batch_size_times
        self.pin_memory = pin_memory
        self.with_l1 = with_l1
        self.with_l2 = with_l2
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.k = k
        self.K_fold = K_fold
        with open(punc_path, encoding='utf-8') as file:
            self.punc2id = { i + 1 : word.strip()for i, word in enumerate(file) } 
        self.punc2id[0] = " " #沒有標點
        self.history_train_loss = []
        self.history_val_loss = []
        if is_continue:
            package = torch.load(save_path)
            self.model = self.model.load_model(save_path).cuda()
            self.optimizer.load_state_dict(package['optim_dict'])
            self.start = package['epoch']
            self.history_train_loss = package['train_loss']
            self.history_val_loss = package['val_loss']
        torch.manual_seed(42)
        self.splits=KFold(n_splits=k,shuffle=True,random_state=42)
    def prfs(self,train_trues,train_preds,total_loss):
        precision, recall, fscore, support = score(train_trues, train_preds)
        accuracy = accuracy_score(train_trues, train_preds)
        print("Multi-class accuracy: %.2f" % accuracy)
        SPLIT = "-"*(12*4+3)
        print(SPLIT)
        f = lambda x : round(x, 2)
        for (v, k) in sorted(self.punc2id.items(), key=lambda x:x[1]):
            if v >= len(precision): continue
            if k == " ":
                k = "  "
            print("Punctuation: {} Precision: {:.3f} Recall: {:.3f} F-Score: {:.3f}".format(k,precision[v],recall[v],fscore[v]))
        print(SPLIT)
        sklearn_accuracy = accuracy_score(train_trues, train_preds) 
        sklearn_precision = precision_score(train_trues, train_preds, average='micro')
        sklearn_recall = recall_score(train_trues, train_preds, average='micro')
        sklearn_f1 = f1_score(train_trues, train_preds, average='micro')
        print("[sklearn_metrics] Total Epoch:{} loss:{:.4f} accuracy:{:.4f} precision:{:.4f} recall:{:.4f} f1:{:.4f}".format(self.epoch+1, \
            total_loss, sklearn_accuracy, sklearn_precision, sklearn_recall, sklearn_f1))
    def train_epoch(self,data_loader):
        self.model.train()  #確保layers of model 在train mode
        total_loss = 0
        train_preds = []
        train_trues = []
        start = time.time()
        count = 0
        for  i,(data) in enumerate(data_loader):
            input ,length, label = data
            if  self.use_cuda:
                input = input.cuda()#換成可傳入gpu的型態
                label = label.cuda()
                length = length.cuda()
                input = input.to(self.device)
                label = label.to(self.device)
                length = length.to(self.device)
            if with_att:
                outputs = self.model(input,length)        
            else:
                outputs = self.model(input,length)
            outputs = outputs.view(-1, outputs.size(-1))
            if self.use_cuda:
                outputs = outputs.to(self.device)
            label = label.view(-1)
            loss = self.criterion(outputs, label)

            loss_with_reg = loss
            if self.use_cuda:
                loss_with_reg = loss_with_reg.to(self.device)
            if self.with_l1:
                l1 = 0
                l1 += sum ( [p.abs().sum() for p in self.model.encoder.parameters()] )
                l1 += sum ( [p.abs().sum() for p in self.model.decoder.parameters()] )
                l1 += sum ( [p.abs().sum() for p in self.model.projected.parameters()] )
                l1_penalty = self.l1_weight *l1
                loss_with_reg += l1_penalty
            if self.with_l2:
                l2 = 1e-3
                l2 += sum ( [(p**2).sum() for p in self.model.encoder.parameters()] )
                l2 += sum ( [(p**2).sum() for p in self.model.decoder.parameters()] )
                l2 += sum ( [(p**2).sum() for p in self.model.projected.parameters()] )
                l2_penalty = self.l2_weight *l2
                loss_with_reg += l2_penalty
            loss_with_reg.backward()
            clipping_value = 2 # arbitrary value of your choosing
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), clipping_value)
            if (i+1) % self.batch_size_times == 0 or (i+1) == len(data_loader):
                self.optimizer.step()
                self.optimizer.zero_grad()
            total_loss += loss.item()
            train_outputs = outputs.argmax(dim=1)
            train_preds.extend(train_outputs.detach().cpu().numpy())
            train_trues.extend(label.detach().cpu().numpy())
        self.scheduler.step()
        print('train: ','\n')
        self.prfs(train_trues,train_preds,total_loss)
        return total_loss/(i+1)
    def val_epoch(self,data_loader):
        val_loss = 0
        self.model.eval()#告訴model不要學新東西
        for i,(data) in enumerate(data_loader):
            val_preds = []
            val_trues = []
            input , length , label = data
            if  self.use_cuda:
                input = input.cuda()#換成可傳入gpu的型態
                label = label.cuda()
                length = length.cuda()
                input = input.to(self.device)
                label = label.to(self.device)
                length = length.to(self.device)
            if with_att:
                outputs = self.model(input,length)
            else:
                outputs = self.model(input,length)
            outputs = outputs.view(-1, outputs.size(-1))
            if self.use_cuda:
                outputs = outputs.to(self.device)
            label = label.view(-1)
            loss = self.criterion(outputs, label.view(-1))
            val_loss += loss.item()
            val_outputs = outputs.argmax(dim=1)

            val_preds.extend(val_outputs.detach().cpu().numpy())
            val_trues.extend(label.detach().cpu().numpy())
        print("validation: ",'\n')
        self.prfs(val_trues,val_preds,val_loss)
        return val_loss/(i+1)
    def train(self):
        for fold, (train_idx,val_idx) in enumerate(self.splits.split(np.arange(len(self.dataset)))):
            train_sampler = SubsetRandomSampler(train_idx)
            val_sampler = SubsetRandomSampler(val_idx)
            train_loader = DataLoader(self.dataset, batch_size=self.batch_size, sampler=train_sampler,collate_fn=self.collate_fn,num_workers=self.num_worker,pin_memory=self.pin_memory)
            val_loader = DataLoader(self.dataset, batch_size=self.batch_size, sampler=val_sampler,collate_fn=self.collate_fn,num_workers=self.num_worker,pin_memory=self.pin_memory)
            #device = "cpu"
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(device,'\n\n\n')
            for epoch in range(self.start,self.epochs):
                self.epoch = epoch
                train_loss=self.train_epoch(train_loader)
                val_loss=self.val_epoch(val_loader)
                print(f"Epoch:{self.epoch + 1} ; {self.epochs} average Training Loss:{train_loss} ; average Test Loss:{val_loss} ")
                self.history_train_loss.append(train_loss)
                self.history_val_loss.append(val_loss)
                torch.save( self.model.serialize(self.model,self.optimizer,epoch,self.history_train_loss,self.history_val_loss) ,self.save_path)    
            if self.K_fold == False:
                break
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
def main():
    language = 'zh'
    if language == 'Eng':
        vocab_path = '/home/yunghuan/NLP_Dataset/English/data_Eng/punc_vocab'
        punc_path = '/home/yunghuan/NLP_Dataset/English/data_Eng/punc_vocab'
        data_path = '/home/yunghuan/NLP_Dataset/English/data_Eng/train.txt'
        embedding_size = 27180
    else:
        vocab_path = '/home/yunghuan/NLP_Dataset/Chinese/data_Ch/vocab.txt'
        punc_path = '/home/yunghuan/NLP_Dataset/Chinese/data_Ch/punc.txt'
        data_path = '/home/yunghuan/NLP_Dataset/Chinese/data_Ch/train.txt'
        embedding_size = 35518
    dataset = PuncDataset(data_path,vocab_path,punc_path)
    collate_fn =Collate()
    hidden_dim = 256
    if with_att:
        encoder = Encoder(embedding_size,300,hidden_dim,3,1,0.,pretrained)
        decoder = Decoder(hidden_dim,3,1,0.0)
        model = Seq2Seq(encoder,decoder,hidden_dim)
        save_path = 'model/final_attn.pth.tar'
    else:
        save_path = 'model/final_noattn.pth.tar'
        model = bi_LSTM(embedding_size,hidden_dim,hidden_dim,2,4)
    use_cuda = True
    if use_cuda:
        model = model.cuda()
    print('parameters_count:',count_parameters(model))
    criterion = torch.nn.CrossEntropyLoss(ignore_index=4)
    optimizer = torch.optim.Adam(model.parameters(),lr = 9e-4,weight_decay=0.0)#0.00002
    scheduler = ExponentialLR(optimizer, gamma=0.5)
    TBRNN = Train(dataset,model,criterion,optimizer,use_cuda,2,30,scheduler,punc_path,save_path,collate_fn,True,batch_size_times=8,)
    TBRNN.train()
if __name__ == '__main__':
    main()

