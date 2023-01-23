import pandas as pd
import numpy as np
import json
import dill
from datasets import load_dataset
import ast
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')

def test_sent_len():
    '''dics2 = load_file("noniid_voc/client2/train2.txt")
    dics1 = load_file("noniid_voc/client2/train1.txt")
    print("load finish")
    lens1 = get_sentence_len(dics1)
    lens2 = get_sentence_len(dics2)
    print("noniidvoc file1 average sentence length{}".format(sum(lens1)/len(lens1)))
    print("noniidvoc file2 average sentence length{}".format(sum(lens2)/len(lens2)))'''
    
    '''dics2 = load_file("noniid_num/client2/train2.txt")
    dics1 = load_file("noniid_num/client2/train1.txt")
    print("load finish")
    lens1 = get_sentence_len(dics1)
    lens2 = get_sentence_len(dics2)
    print("noniidnum file1 average sentence length{}".format(sum(lens1)/len(lens1)))
    print("noniidnum file2 average sentence length{}".format(sum(lens2)/len(lens2)))'''
    
    '''dics2 = load_file("client2/train2.txt")
    dics1 = load_file("client2/train1.txt")
    lens1 = get_sentence_len(dics1)
    lens2 = get_sentence_len(dics2)
    print("noniid file1 average sentence length{}".format(sum(lens1)/len(lens1)))
    print("noniid file2 average sentence length{}".format(sum(lens2)/len(lens2)))'''
    
    dics2 = load_file("noniid_len/client2/train2.txt")
    dics1 = load_file("noniid_len/client2/train1.txt")
    print("load finish")
    lens1 = get_sentence_len(dics1)
    lens2 = get_sentence_len(dics2)
    print("noniidlen file1 average sentence length{}".format(sum(lens1)/len(lens1)))
    print("noniidlen file2 average sentence length{}".format(sum(lens2)/len(lens2)))
    
    '''dics2 = load_file("client2/val2.txt")
    dics1 = load_file("client2/val1.txt")
    lens1 = get_sentence_len(dics1)
    lens2 = get_sentence_len(dics2)
    print("noniid file1 average sentence length{}".format(sum(lens1)/len(lens1)))
    print("noniid file2 average sentence length{}".format(sum(lens2)/len(lens2)))'''
    
    '''#dics2 = load_file("noniid_len/client2/val2.txt")
    dics1 = load_file("noniid_voc/client2/val1.txt")
    print("load finish")
    lens1 = get_sentence_len(dics1)
    #lens2 = get_sentence_len(dics2)
    print("noniidvoc file1 average sentence length{}".format(sum(lens1)/len(lens1)))
    #print("noniidvoc file2 average sentence length{}".format(sum(lens2)/len(lens2)))'''

def test_voc():
    '''dics2 = load_file("client2/train2.txt")
    dics1 = load_file("client2/train1.txt")
    print("load finish")
    len1 = test_vocab(dics1)
    len2 = test_vocab(dics2)'''
    
    '''dics2 = load_file("noniid_voc/client2/train2.txt")
    dics1 = load_file("noniid_voc/client2/train1.txt")
    print("load finish")
    len1 = test_vocab(dics1)
    len2 = test_vocab(dics2)'''
    
    '''dics2 = load_file("noniid_num/client2/train2.txt")
    dics1 = load_file("noniid_num/client2/train1.txt")
    print("load finish")
    len1 = test_vocab(dics1)
    len2 = test_vocab(dics2)'''
    
    dics2 = load_file("noniid_len/client2/train2.txt")
    dics1 = load_file("noniid_len/client2/train1.txt")
    print("load finish")
    len1 = test_vocab(dics1)
    len2 = test_vocab(dics2)
    
    '''dics2 = load_file("client2/val2.txt")
    dics1 = load_file("client2/val1.txt")
    print("load finish")
    len1 = test_vocab(dics1)
    len2 = test_vocab(dics2)'''
    
    '''dics2 = load_file("noniid_len/client2/val2.txt")
    #dics1 = load_file("noniid_voc/client2/val1.txt")
    print("load finish")
    #len1 = test_vocab(dics1)
    len2 = test_vocab(dics2)'''
    
    
def load_file(file):
    dics = []
    with open(file,'r') as f:
        for line in f.readlines():
            #print(type(line))
            dic = json.loads(line)
            dics.append(dic)
            
    return dics

def get_sentence_len(dics):
    ave_lens = []  
    count = 1  
    for dic in dics:
        total_len = 0
        texts = dic['article_text']
        for sent in texts:        
            sent_token = word_tokenize(sent)                  
            total_len += len(sent_token)
        #print(count)
        if len(texts) == 0:
            ave_lens.append(0)
        else:
            ave_len = total_len / len(texts)
            ave_lens.append(ave_len)
        #print(ave_len)
        count += 1
    print(len(ave_lens))
    return ave_lens

def get_vocab(dics): 
    vocs = []
    vocs_lens = []
    count = 1  
    for dic in dics:
        texts = dic['article_text']
        voc = []
        for sent in texts:        
            sent_token = word_tokenize(sent)
            for tok in sent_token:
                if tok not in voc:
                    voc.append(tok)
        #print(count)
        #vocs.append(voc)  
        vocs_lens.append(len(voc)) 
        count += 1
    print(len(vocs_lens))
    return vocs, vocs_lens

def test_vocab(dics): 
    voc = []
    count = 0 
    for dic in dics:
        texts = dic['article_text']
        for sent in texts:        
            sent_token = word_tokenize(sent)
            for tok in sent_token:
                if tok not in voc:
                    #voc.append(tok)
                    count += 1
        #print(count)
        
    print(count)
    return count

def noniid_partition(num, rank, file, ifval = True):
    num_files = num
    with open(file) as in_file:
        lines = in_file.readlines()
        lines_per_file = len(lines) // num_files
        for n in range(num_files):
            if ifval:
                with open('noniid_len/client{}/val{}.txt'.format(num, n+1), 'w') as out_file:
                    for i in rank[n * lines_per_file:(n+1) * lines_per_file]: 
                        out_file.write(lines[i])
            else:
                with open('noniid_len/client{}/train{}.txt'.format(num, n+1), 'w') as out_file:
                    for i in rank[n * lines_per_file:(n+1) * lines_per_file]:
                        out_file.write(lines[i])
                        
def noniid_partition_num(num):
    num_files = 0
    for n in range (num):
        num_files += n+1
    
    print(num_files)
           
    with open('val.txt') as in_file:
        lines = in_file.readlines()
        range1 = [0]
        for n in range (num):
            range1.append(range1[n] + len(lines) * (n+1) // num_files)
        
        for n in range(num):
            with open('noniid_num/client{}/val{}.txt'.format(num, n+1), 'w') as out_file:
                for i in range(range1[n], range1[n+1]):
                    out_file.write(lines[i])
                    
    with open('train.txt') as in_file:
        lines = in_file.readlines()
        range1 = [0]
        for n in range (num):
            range1.append(range1[n] + len(lines) * (n+1) // num_files)
            
        for n in range(num):
            with open('noniid_num/client{}/train{}.txt'.format(num, n+1), 'w') as out_file:
                for i in range(range1[n], range1[n+1]):
                    out_file.write(lines[i])
 
if __name__ == "__main__":
    
    dfile = "train.txt"
    dics = load_file(dfile)
    sent_lens = get_sentence_len(dics)
    vocs, vocs_lens = get_vocab(dics)
    print("load finish")
    
    '''tem = np.divide(np.mat(sent_lens),np.mat(vocs_lens))
    tem = np.divide(np.mat(tem),np.mat(vocs_lens))'''
    
    tem = np.divide(np.mat(vocs_lens),np.mat(sent_lens))
    tem = np.divide(np.mat(tem),np.mat(sent_lens))
    tem = np.divide(np.mat(tem),np.mat(sent_lens))
    tem = np.divide(np.mat(tem),np.mat(sent_lens))
    
    tem = np.matrix.tolist(tem)
    rank = np.argsort(tem[0])
    
    noniid_partition(2, rank, dfile, ifval = False)
    
    print("partition finish")
    
    '''sent_lens = get_sentence_len(dics)
    rank = np.argsort(sent_lens)
    noniid_partition(2, rank, dfile, ifval = False)'''
    
    '''vocs, vocs_lens = get_vocab(dics)
    rank = np.argsort(vocs_lens)
    noniid_partition(2, rank, dfile, ifval = False)'''
       
    #noniid_partition_num(2)
    
    test_sent_len()
    
    test_voc()
