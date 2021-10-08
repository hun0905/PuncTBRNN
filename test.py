from torch.utils.data import dataset
from dataset import PuncDataset
import torch
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support as score
from model5 import Seq2Seq
from model2 import bi_LSTM
import numpy as np
from add_punc import punc_vocab
def test(data_path,vocab_path,punc_path,model_path,use_cuda):
    dataset = PuncDataset(data_path,vocab_path,punc_path)
    model = Seq2Seq.load_model(model_path)
    print(model)
    model.eval()
    if use_cuda:
        model = model.cuda()
    labels = np.array([])
    predicts = np.array([])
    for i, (word_id, label_id) in enumerate(dataset):
        input_lengths = torch.LongTensor([len(word_id)])
        input = word_id.unsqueeze(0)
        if use_cuda:
            input,input_lengths = input.cuda(),input_lengths.cuda()
        result = model(input,input_lengths)
        result = result.view(-1, result.size(-1))
        _, predict = torch.max(result, 1)
        #print(label_id)
        predict = predict.data.cpu().numpy()
        labels = np.append(labels, label_id)
        predicts = np.append(predicts, predict)
    punc2id = punc_vocab(punc_path)
    precision, recall, fscore, support = score(labels, predicts)
    accuracy = accuracy_score(labels, predicts)
    print("Multi-class accuracy: %.2f" % accuracy)
    SPLIT = "-"*(12*4+3)
    print(SPLIT)

    
    f = lambda x : round(x, 2)
    for (v, k) in sorted(punc2id.items(), key=lambda x:x[1]):
        if v >= len(precision): continue
        if k == " ":
            k = "  "
        print("Punctuation: {} Precision: {:.3f} Recall: {:.3f} F-Score: {:.3f}".format(k,precision[v],recall[v],fscore[v]))
    print(SPLIT)

    all_precision = sum( [precision[i]*support[i]/sum(support[1:]) for i in range(1,len(punc2id))] )
    all_recall = sum( [recall[i]*support[i]/sum(support[1:]) for i in range(1,len(punc2id))] )
    all_fscore = sum( [fscore[i]*support[i]/sum(support[1:]) for i in range(1,len(punc2id))] )
    print("OverAll(punc):  Precision: {:.3f} Recall: {:.3f} F-Score: {:.3f}".format(all_precision,all_recall,all_fscore))
def main():
    data_path = 'data_train_test_Ch/test.txt'
    vocab_path = 'data_train_test_Ch/vocab.txt'
    punc_path = 'data_train_test_Ch/punc.txt'
    model_path = 'model/final_attn.pth.tar'
    test(data_path,vocab_path,punc_path,model_path,True)
if __name__ == '__main__':
    main()