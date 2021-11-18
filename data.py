import gc,torch
from sklearn.feature_extraction.text import CountVectorizer
import math,os
import numpy as np
import nltk
from nltk.corpus import stopwords
from utils import load_obj_pkl5,load_obj,save_obj
from nltk import word_tokenize
nltk.download('punkt')


def load_data(data,dtype,cur_dir,skipgram_embeddings):

  data = data.lower()
  dtype = dtype.lower()
  dir ='/content/data_'+data+'/'+dtype
  os.chdir(cur_dir+dir)

  data_preprocessed=load_obj_pkl5("data_preprocessed_"+data+"_"+dtype)
  data_preprocessed_labels=load_obj_pkl5("data_preprocessed_labels_"+data+"_"+dtype)
  if skipgram_embeddings: embeddings=load_obj_pkl5("generated_embeddings_"+data+"_"+dtype)
  else: embeddings=load_obj_pkl5("embeddings_"+data+"_"+dtype)
  os.chdir(cur_dir)
  return data_preprocessed,data_preprocessed_labels,embeddings,data
  

def get_data_label_vocab_for_large_data(data,lables,max_features):
  min_df=0
  preprossed_data = data
  train_label = lables
  vectorizer = CountVectorizer(min_df=min_df,max_features=max_features,dtype=np.float32)
  count_vec = vectorizer.fit_transform(preprossed_data)
  vocab = vectorizer.vocabulary_
  id_vocab = dict(map(reversed, vocab.items()))
  print("train_input_vec_shape : ",count_vec.shape,count_vec.__class__.__name__,len(vocab))
  # print(vocab)
  train_label = np.asarray(train_label)

  return count_vec,train_label,id_vocab

def get_data_label_vocab_normal(data,lables,max_features):
  min_df=0
  preprossed_data = data
  train_label = lables
  vectorizer = CountVectorizer(min_df=min_df,max_features=max_features,dtype=np.float32)
  train_vec = vectorizer.fit_transform(preprossed_data).toarray()
  vocab = vectorizer.vocabulary_
  id_vocab = dict(map(reversed, vocab.items()))
  print("train_input_vec_shape : ",train_vec.shape,train_vec.__class__.__name__,len(vocab))
  # print(vocab)
  
  tensor_train_w = (torch.from_numpy(np.array(train_vec)).float())
  train_label = np.asarray(train_label)

  return tensor_train_w,train_label,id_vocab