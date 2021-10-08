import torch
from torch._C import _llvm_enabled
from torch.utils.data import dataset
from model2 import bi_LSTM
from model5 import Encoder,Decoder,Seq2Seq
from dataset import NoPuncDataset
from gensim.models import word2vec
from gensim.models import Word2Vec
from torch.utils.data import Dataset, DataLoader
vocab_path = 'data/vocabulary'
with open(vocab_path, encoding='utf-8',errors='ignore') as file:
    vocab = { word.strip(): i + 3 for i, word in enumerate(file) } 
def punc_vocab(punc_vocab_path):
    with open(punc_vocab_path, encoding='utf-8') as file:
        punc_vocab = { i + 1 : word.strip()for i, word in enumerate(file) } 
    punc_vocab[0] = " " #沒有標點
    return punc_vocab

def add_punc(seq, predict, class2punc):    
    txt_with_punc = ""
    for i, word in enumerate(seq.split()):
        punc = class2punc[predict[i]][0]
        txt_with_punc += word + " " if punc == " " else punc + " " + word + " "
    punc = class2punc[predict[i + 1]][0]
    txt_with_punc += punc
    print(txt_with_punc)
    return txt_with_punc

def add_punctuation(use_cuda = True):
    model = Seq2Seq.load_model('model/final_attn.pth.tar')
    model.eval()
    if use_cuda:
        model = model.cuda()
    demo_result = open('data_after_process/dem0_result.txt','w')
    dataset = NoPuncDataset('data_after_process/demo.txt',\
                            'data/vocabulary',\
                            'data/punc_vocab')
    data_loader = DataLoader(dataset, batch_size=1)
    class2punc = punc_vocab('data/punc_vocab')
    
    with torch.no_grad():
        for i,(seq_id,seq) in enumerate(data_loader):
            lengths = torch.LongTensor([ seq_id.size()[1] ])
            if use_cuda:
                seq_id = seq_id.cuda()
            output = model(seq_id)
            output = torch.argmax( output.squeeze(1) , 1)
            output = output.data.cpu().numpy().tolist()
            out_text = add_punc(seq[0],output,class2punc)
            demo_result.write(out_text)
            demo_result.write('\n')
        
def main():
    #add_punc('/home/yunghuan/Desktop/PunctuationPredict_Eng_without_att/data_after_process/demo.txt')
    add_punctuation()
if __name__ == '__main__':
    main()
