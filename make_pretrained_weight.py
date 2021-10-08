import gensim
import torch
file =  open('/home/yunghuan/NLP_Dataset/Chinese/data_Ch/vocab.txt','r', encoding='utf-8',errors='ignore').readlines()
word2idx = { word.strip(): i + 3 for i, word in enumerate(file) } 
idx2word = { i + 3: word.strip() for i, word in enumerate(file) }
word2idx['<PAD>'] = 0 
word2idx['<UNK>'] = 1 #UNK is unknown word
word2idx['<END>'] = 2 #END is the END of sequenc
idx2word[0] = '<PAD>'
idx2word[1] = '<UNK>'
idx2word[2] = '<END>'
wv = open('/home/yunghuan/NLP_Dataset/Chinese/pretrained_vectors/zh_wiki_fasttext_300.txt','r'\
        ,encoding='utf-8',errors='ignore').readlines()[1:]

#wv = gensim.models.KeyedVectors.load_word2vec_format('/home/yunghuan/Desktop/Punctuation_test_now/PuncBiLstm_924/pretrained_vectors/zh_wiki_fasttext_300.txt')
#print(wv[0].split())

vocab_size = len(file)+3
embed_size = 300
weight = torch.zeros(vocab_size,embed_size)
for i in wv:
    line = i.split()
    try:
        index = word2idx[line[0] ]
    except:
        continue
    weight[index,:] = torch.FloatTensor(list(map(float,line[1:]))).cpu()
weight[0,:] = torch.ones(300)
weight[1,:] = torch.zeros(300)
weight[2,:] = torch.full((300,),0.5)
torch.save(weight,'/home/yunghuan/Desktop/PuncTBRNN/ChineseFastText.pth')