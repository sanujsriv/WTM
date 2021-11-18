## Preprocessing_data.py

from preprocessing_utils import *
from preprocessing_imports import *

def get_bbc_data(dtype='full'):
  if not os.path.isfile('bbc.zip'):
    os.system('wget -N https://www.dropbox.com/s/vunli21d312x55g/bbc.zip')
    os.system('wget -N https://www.dropbox.com/s/h6y9zfdb76gl4uz/bbc-fulltext.zip')
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


def get_searchsnippet_data():
  if not os.path.isfile('data-web-snippets.tar.gz'):
    os.system('wget http://jwebpro.sourceforge.net/data-web-snippets.tar.gz')
    os.system('tar -xvzf data-web-snippets.tar.gz')
  corpus = []
  labels=[]
  with open("/content/data-web-snippets/train.txt", "r") as f:
    content = f.readlines()
    content = [x.split('\n')[0] for x in content]
    labels_train = [x.split(' ')[-1] for x in content]
    content = [' '.join(x.split(' ')[:-1]) for x in content]
    corpus.extend(content)
    labels.extend(labels_train)

  with open("/content/data-web-snippets/test.txt", "r") as f:
    content = f.readlines()
    content = [x.split('\n')[0] for x in content]
    labels_test = [x.split(' ')[-1] for x in content]
    content = [' '.join(x.split(' ')[:-1]) for x in content]
    corpus.extend(content)
    labels.extend(labels_test)
  return corpus,labels


def get_AGNews_data_120k():
  # try : os.mkdir(data_to_get)
  # except FileExistsError:pass
  # os.chdir(data_to_get)
  if not os.path.isfile('ag_news_csv.tar.gz'):
    os.system('wget -N -c https://www.dropbox.com/s/tyzue51quuo5y79/ag_news_csv.tar.gz')
    os.system('tar -xvzf ag_news_csv.tar.gz')
  
  home_dir = '' # your home_dir here
  os.chdir(home_dir+'/ag_news_csv')
  dict_labels = {'1':'World', '2': 'Sports', '3': 'Business', '4': 'Sci/Tech'}
  labels = []
  corpus = []
  # files = ['train.csv','test.csv']
  files = ['train.csv']
  count = 0
  for f in files:
    with open(f, newline='') as csvfile:
        reader = csv.reader(csvfile)
        # next(reader)
        for row in reader:
          count = count+1
          labels.append(row[0])
          corpus.append(row[-1])
        # corpus = corpus[1:]
        # labels = labels[1:]
  labels = [dict_labels[l] for l in labels]
  os.chdir('..')
  return corpus,labels

def get_yahoo_answers(dtype):
  gdrive_fileid = """0Bz8a_Dbh9Qhbd2JNdDBsQUdocVU"""
  gdrive_filename = """yahoo_answers_csv.tar.gz"""
 
  home_dir = '' # your home_dir here
  
  dir= home_dir+'/yahoo_answers_csv'
  if not os.path.exists(dir) and not os.path.isfile(gdrive_filename):
    # yahoo_answers_csv.tar.gz 
    # Download Link - https://drive.google.com/drive/u/0/folders/0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M?resourcekey=0-TLwzfR2O-D2aPitmn5o9VQ%2F
    # !wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=0Bz8a_Dbh9Qhbd2JNdDBsQUdocVU' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=0Bz8a_Dbh9Qhbd2JNdDBsQUdocVU" -O yahoo_answers_csv.tar.gz && rm -rf /tmp/cookies.txt
    os.system("tar -xvzf yahoo_answers_csv.tar.gz")
  # os.makedirs(home_dir+dir,exist_ok=True)
  os.chdir(dir)
  dict_labels = {'1':'Society & Culture', '2': 'Science & Mathematics', 
                '3': 'Health', '4': 'Education & Reference', '5' : 'Computers & Internet',
                '6': 'Sports', '7': 'Business & Finance', '8': 'Entertainment & Music',
                '9': 'Family & Relationships' , '10':'Politics & Government'}

  labels = []
  corpus = []
  files = ['train.csv','test.csv']
  # files = ['train.csv']
  count = 0
  for f in files:
    with open(f, newline='') as csvfile:
        reader = csv.reader(csvfile)
  #       # next(reader)
        if dtype == 'short':
          for row in reader:
            count = count+1
            labels.append(dict_labels[row[0]])
            # corpus.append(' '.join(t for t in row[1:]))
            corpus.append(' '.join(t for t in row[1:3])) # SHORT 
        elif dtype == "full": 
          for row in reader:
            count = count+1
            labels.append(dict_labels[row[0]])
            corpus.append(' '.join(t for t in row[1:]))
      
  os.chdir(home_dir)
  return corpus,labels
