
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import json
import time
import random
import numpy as np
import pandas as pd
import math
import gc
import psutil
from collections import Counter
import warnings
warnings.filterwarnings('ignore')
import sys
# from sklearn.externals import joblib
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import time
from torch.autograd import Variable
from glob import glob
from sys import getsizeof
import os
import torch.nn as nn
from torch.optim import lr_scheduler
from torch import optim
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
import shutil
from torchvision import transforms
from torchvision import models
from torchtext import data, datasets
from nltk import ngrams
from torchtext.vocab import GloVe, Vectors
from collections import defaultdict
data_path = '/kaggle/input/spooky-author-identification/'
import xgboost as xgb
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from nltk import word_tokenize
from nltk.corpus import stopwords
import nltk
stop_words = stopwords.words('english')
from torch.nn import utils as nn_utils

print (torch.cuda.is_available())

def read_data():
    df_train = pd.read_csv(data_path + 'train.zip')
    df_test = pd.read_csv(data_path + 'test.zip')
    df_sub = pd.read_csv(data_path + 'sample_submission.zip')
    return df_train, df_test, df_sub
df_train, df_test, df_sub = read_data()
print (df_train.shape, df_test.shape, df_sub.shape)
print (df_train.head(3), '\n\n', df_test.head(3), '\n\n', df_sub.head(3))
# 处理数据
lbl_enc = preprocessing.LabelEncoder()
y = lbl_enc.fit_transform(df_train.author.values)
xtrain, xvalid, ytrain, yvalid = train_test_split(df_train.text.values, y, 
                                                  stratify = y, 
                                                  random_state = 2020, 
                                                  test_size = 0.1, shuffle = True)
xtest = df_test.text.values
print (xtrain.shape)
print (xvalid.shape)
print (xtest.shape)

embeddings_word_set = set()
def sent2vec_add_word(s):
    words = str(s).lower()
    words = word_tokenize(words)
    words = [w for w in words if not w in stop_words]
    words = [w for w in words if w.isalpha()]
    for w in words:
        embeddings_word_set.add(str.encode(w))
    return 0

temp = [sent2vec_add_word(x) for x in xtrain]
del temp
gc.collect()
temp = [sent2vec_add_word(x) for x in xvalid]
del temp
gc.collect()
temp = [sent2vec_add_word(x) for x in xtest]
del temp
gc.collect()

# %% [code]
# load the GloVe vectors in a dictionary:
embeddings_index = {}
f = open(data_path + '../glove840b300dtxt/glove.840B.300d.txt', 'rb')
index = 0
pre_time = time.time()
for line in f: # tqdm(f):
    index += 1
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    if word in embeddings_word_set:
        embeddings_index[word] = coefs
    if index % 500000 == 0:
        print ('index: {:}, time:{:}'.format(index, time.time() - pre_time))
f.close()
print('Found %s word vectors.' % len(embeddings_index))

# %% [code]
import gc
del df_train, df_test, df_sub
gc.collect()

# %% [code]
def sent2vec(s):
    words = str(s).lower()
    words = word_tokenize(words)
    words = [w for w in words if not w in stop_words]
    words = [w for w in words if w.isalpha()]
    M = []
    for w in words:
        try:
            torch_tmp = list(embeddings_index[str.encode(w)])
            M.append(torch_tmp)
        except:
            continue
    return M

xtrain_glove = [sent2vec(x) for x in xtrain]
del xtrain
gc.collect()
xvalid_glove = [sent2vec(x) for x in xvalid]
del xvalid
gc.collect()
xtest_glove = [sent2vec(x) for x in xtest]
del xtest
gc.collect()

print (psutil.virtual_memory().percent)
print (psutil.virtual_memory().total / 1024 / 1024 / 1024)
del embeddings_index
gc.collect()
print (psutil.virtual_memory().percent)
print (psutil.virtual_memory().total / 1024 / 1024 / 1024)

# %% [code]
word_vector_size = len(xtrain_glove[0][0])
label_size = 3
print ('word_vector_size: {:}, label_size: {:}'.format(word_vector_size, label_size))

xtrain_lengths = torch.LongTensor([len(x) for x in xtrain_glove]).cuda()
print (xtrain_lengths.max())
max_length = 100
xtrain_troch = torch.zeros((len(xtrain_glove), max_length, word_vector_size)).long().cuda()
for idx in range(len(xtrain_glove)):
    seqlen = min(xtrain_lengths[idx], max_length)
    seq = xtrain_glove[idx]
    xtrain_troch[idx, :seqlen] = torch.LongTensor(seq[: seqlen, :])

print (typ(xtrain_troch), xtrain_troch.size())
xtrain_lengths, seq_idx = xtrain_lengths.sort(0, descending = False)
xtrain_troch = xtrain_troch[seq_idx]

