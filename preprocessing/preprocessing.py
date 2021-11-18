## Preprocessing.py
"""
Preprocessing Steps :- 
1. Convert to lowercase
2. Remove punctuations with empty space 
3. Remove digits
4. Apply lemmatization
5. Remove Stopwords
6. Remove words that do not have word embeddings
7. Remove words that have length < 3.
*** NOT APPLYING STEMMING (Instead applying lemmatization for better results)

Generating Preprocessed Docs Steps:
1. Apply Preprocessing Steps
2. Remove Documents that have length < 3
3. Apply CountVectorizer Data transform { with min_df =3 i.e. words that appear in less than 3 documents are removed }
4. Remove documents that are empty after countvectorization
5. Remove words in documents that are not in the vocab generated after countvectorization.
6. export the vocab words for which we have existing embeddings(Google Word2vec, GloVE etc.)
7. export the preprocessed data with its label.
8. (optional) Learn new embeddings using the preprocessed corpus 
"""
from preprocessing_imports import *
from preprocessing_utils import *
from preprocessing_data import get_bbc_data
import gensim.downloader as api

# download and extract the GoogleNews-vectors-negative300.bin (https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz)
word2vec_model = models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

def get_data(data_to_get,dtype):
  data_to_get = data_to_get.lower()
  if data_to_get == 'bbc':
    corpus,labels = get_bbc_data(dtype)
  return corpus,labels

## Driver - Data Generation
all_data_to_get = ['bbc'] 
dtypes = ['short'] # short text 

## Settings :- 
min_doc_len = 3
min_word_length = 3

generate_skipgram_embeddings = 1 # Yes

home_dir = ''
for data_to_get in all_data_to_get:

  if data_to_get == 'bbc': max_features = 2000
  else: max_features = 4000
  
  for dtype in dtypes:
    os.chdir(home_dir+"/content")
    docs,labels = get_data(data_to_get,dtype)
    os.makedirs(home_dir+'/content/data_'+data_to_get+"/"+dtype,exist_ok=True)
    os.chdir(home_dir+'/content/data_'+data_to_get+"/"+dtype)
    with open(data_to_get+'_'+dtype+".txt", "a") as f:
      f.write(data_to_get+" - "+dtype)
      f.write("\n\n")
      f.write('**'*50)
      f.write("\n\n")
      f.write(str(len(docs))+', '+str(len(labels))+'\n')
      f.write('(labels,count): '+str(list(zip(*np.unique(labels, return_counts=True)))))
      data_preprocessed,data_preprocessed_labels,embeddings,vocab = docs_labels_preprocessing(docs,labels,word2vec_model,min_doc_len,min_word_length,max_features)
      f.write('\n\nlen of - \n  data_preprocessed: '+str(len(data_preprocessed))+'\n  data_preprocessed_labels: '+str(len(data_preprocessed_labels))+'\n  vocab:  '+str(len(vocab))+'\n  embeddings : '+str(len(embeddings))+'\n')
      len_docs = [len(d.split(" ")) for d in data_preprocessed]
      f.write('\n min,mean,max docs len: '+str(np.min(len_docs))+', '+str(np.mean(len_docs).round(2))+', '+str(np.max(len_docs))+'\n\n')
      
      save_obj(data_preprocessed,'data_preprocessed'+'_'+data_to_get+'_'+dtype)
      save_obj(data_preprocessed_labels,'data_preprocessed_labels'+'_'+data_to_get+'_'+dtype)   
      save_obj(vocab,'vocab'+'_'+data_to_get+'_'+dtype)
      save_obj(embeddings,'embeddings'+'_'+data_to_get+'_'+dtype)
      f.write('**'*50)

if generate_skipgram_embeddings:
  ## Skipgram embeddings generation 

  gensim_prep_doc = [word_tokenize(d) for d in data_preprocessed]
  model = gensim.models.Word2Vec(gensim_prep_doc, min_count=2, sg=1, size=300,iter=50, negative=10, window=4,workers=16)
  vocab = list(model.wv.vocab)
  # model.save(data_to_get+"_"+dtype+"_generated_word2vec.model")

  generated_embeddings = {}
  for v in list(model.wv.vocab):
    vec = model.wv.__getitem__(v)
    generated_embeddings[v] =vec

  save_obj(generated_embeddings,'generated_embeddings_'+data_to_get+"_"+dtype)
  # save_obj(vocab,'generated_vocab_'+data_to_get+"_"+dtype)

os.chdir(home_dir+'/content')
zipped_pickle_filename = data_to_get
os.system('zip -r '+zipped_pickle_filename+'_'+str(max_features)+'_.zip '+'/content/data_'+data_to_get+"/"+dtype)
