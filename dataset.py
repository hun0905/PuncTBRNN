import torch
from torch.functional import Tensor
from torch.nn.utils.rnn import pad_sequence
import torch.utils.data as data 
import nltk
from gensim.models import word2vec
from gensim.models import Word2Vec
class PuncDataset(data.Dataset):
    def __init__(self, file_path, vocab_path, punc_vocab_path):
        self.seqs = open(file_path, encoding='utf8',errors='ignore').readlines()
        #製作vocab 的對照表
        #file = Word2Vec.load('/home/yunghuan/Desktop/PunctuationPredict_Eng_without_att/w2v_all.model')
        with open(vocab_path, encoding='utf-8',errors='ignore') as file:
            vocab = { word.strip(): i + 3 for i, word in enumerate(file) } 
        vocab['<PAD>'] = 0 
        vocab['<UNK>'] = 1 #UNK is unknown word
        vocab['<END>'] = 2 #END is the END of sequence
        self.vocab = vocab

        with open(punc_vocab_path, encoding='utf-8') as file:
            punc_vocab = { word.strip(): i + 1 for i, word in enumerate(file) } 
        punc_vocab[" "] = 0 #沒有標點
        #punc_vocab["END"] = 1
        self.punc_vocab = punc_vocab
        self.seqs.sort(key=lambda x: len(x.split()), reverse=True) #將句子由長到短來排列
        
    def __len__(self):
        return len(self.seqs)
    def __getitem__(self, index):
        seq = self.seqs[index].strip('\n')
        input = []
        label = []
        punc = ' '
        for token in seq.split():
            if token in self.punc_vocab:
                punc = token
                #label.append(self.punc_vocab[token])
            else:
                input.append(self.vocab.get(token, self.vocab["<UNK>"]))
                label.append(self.punc_vocab[punc])
                punc = ' '
        input.append(self.vocab["<END>"])
        label.append(self.punc_vocab[punc])
        length = [len(input)]
        input = torch.LongTensor(input)
        label = torch.LongTensor(label)
        #print(label)
        return input,label
class NoPuncDataset(data.Dataset):
    def __init__(self, file_path, vocab_path, punc_vocab_path):
        self.seqs = open(file_path, encoding='utf8',errors='ignore').readlines()
        #製作vocab 的對照表
        #file = Word2Vec.load('/home/yunghuan/Desktop/PunctuationPredict_Eng_without_att/w2v_all.model')
        with open(vocab_path, encoding='utf-8',errors='ignore') as file:
            vocab = { word.strip(): i + 3 for i, word in enumerate(file) } 
        vocab['<PAD>'] = 0 
        vocab['<UNK>'] = 1 #UNK is unknown word
        vocab['<END>'] = 2 #END is the END of sequence
        self.vocab = vocab

        
    def __len__(self):
        return len(self.seqs)
    def __getitem__(self, index):
        seq = self.seqs[index].strip('\n')
        seq_id = self._process(seq)
      
        seq_id = torch.LongTensor(seq_id)
        return seq_id,seq
    def _process(self,seq):
        """Convert txt sequence to word-id-seq."""
        input = []
        for word in seq.split():
            input.append(self.vocab.get(word, self.vocab["<UNK>"]))
        input.append(self.vocab["<END>"])
        input = torch.LongTensor(input)
        return input

class Collate(object):
    def __init__(self):
        pass
    def __call__(self, batch):
        batch.sort(key=lambda x: len(x[0]), reverse=True)
        input_seqs, label_seqs = zip(*batch)
        lengths = [len(seq) for seq in input_seqs]
        #max_length = 100
        max_length = max(lengths)
        #print(max_length)
        input_padded = []
        label_padded = []
        for i, (input, label) in enumerate(zip(input_seqs, label_seqs)):
            n = max_length-lengths[i]
        
            input_padded.append( list( nltk.pad_sequence(input,n+1,pad_right=True,right_pad_symbol=0) )  ) 
            label_padded.append( list( nltk.pad_sequence(label,n+1,pad_right=True,right_pad_symbol=0) )  ) #4 is end
            #print(torch.IntTensor(lengths))
        input_padded = torch.tensor(input_padded)
        label_padded = torch.tensor(label_padded)
        return input_padded, torch.IntTensor(lengths), label_padded

