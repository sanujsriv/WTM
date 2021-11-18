## Preprocessing_data.py

from preprocessing_utils import *
from preprocessing_imports import *

## Example for bbc

def get_bbc_data(dtype='full'):
  if not os.path.isfile('bbc.zip'):
    ## Raw dataset public link
    os.system('wget -N http://mlg.ucd.ie/files/datasets/bbc.zip')
    os.system('wget -N http://mlg.ucd.ie/files/datasets/bbc-fulltext.zip')
    os.system('unzip bbc.zip')
    os.system('unzip bbc-fulltext.zip')

  # BBC Docs -
  corpus = []
  subfolders = [f.path for f in os.scandir(os.getcwd()+'/bbc') if f.is_dir()]
  subfolders = sorted(subfolders)
  for s in subfolders:
    files_list = sorted(glob.glob(s+"/*.txt")) 
    for file in files_list:
      with open(file, "rb") as f:
        content = f.readlines()
        if dtype == 'short':
          content = [content[0],content[2]] # headlines and abstracts 
        content = [x.strip().lower().decode('ISO-8859-1') for x in content] 
        corpus.append(' '.join(content).strip())

  # BBC_labels -
  with open("bbc.classes", "r") as f:
    content = f.readlines()
    content = [x.strip()[::-1][0] for x in content] 
    labels = content[4:]
    label_dict = {'0':'business','1':'entertainment','2':'politics','3':'sport','4':'tech'}
  for l in range(len(labels)):
    labels[l] = label_dict[labels[l]]
  return corpus,labels