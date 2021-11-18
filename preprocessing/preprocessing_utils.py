## Preprocessing_utils.py

from preprocessing_imports import *

def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def load_obj_pkl5(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle5.load(f)

wnl = WordNetLemmatizer()
stopwords = ['la','wa','will','fa','ha','pa','co','v','said','cant','better','well','going','will','would','know','dont','get','like','think','im',"also","said","a", "able", "about", "above", "abst", "accordance", "according", "accordingly", "across", "act", "actually", "added", "adj", "affected", "affecting", "affects", "after", "afterwards", "again", "against", "ah", "all", "almost", "alone", "along", "already", "also", "although", "always", "am", "among", "amongst", "an", "and", "announce", "another", "any", "anybody", "anyhow", "anymore", "anyone", "anything", "anyway", "anyways", "anywhere", "apparently", "approximately", "are", "aren", "arent", "arise", "around", "as", "aside", "ask", "asking", "at", "auth", "available", "away", "awfully", "b", "back", "be", "became", "because", "become", "becomes", "becoming", "been", "before", "beforehand", "begin", "beginning", "beginnings", "begins", "behind", "being", "believe", "below", "beside", "besides", "between", "beyond", "biol", "both", "brief", "briefly", "but", "by", "c", "ca", "came", "can", "cannot", "can't", "cause", "causes", "certain", "certainly", "co", "com", "come", "comes", "contain", "containing", "contains", "could", "couldnt", "d", "date", "did", "didn't", "different", "do", "does", "doesn't", "doing", "done", "don't", "down", "downwards", "due", "during", "e", "each", "ed", "edu", "effect", "eg", "eight", "eighty", "either", "else", "elsewhere", "end", "ending", "enough", "especially", "et", "et-al", "etc", "even", "ever", "every", "everybody", "everyone", "everything", "everywhere", "ex", "except", "f", "far", "few", "ff", "fifth", "first", "five", "fix", "followed", "following", "follows", "for", "former", "formerly", "forth", "found", "four", "from", "further", "furthermore", "g", "gave", "get", "gets", "getting", "give", "given", "gives", "giving", "go", "goes", "gone", "got", "gotten", "h", "had", "happens", "hardly", "has", "hasn't", "have", "haven't", "having", "he", "hed", "hence", "her", "here", "hereafter", "hereby", "herein", "heres", "hereupon", "hers", "herself", "hes", "hi", "hid", "him", "himself", "his", "hither", "home", "how", "howbeit", "however", "hundred", "i", "id", "ie", "if", "i'll", "im", "immediate", "immediately", "importance", "important", "in", "inc", "indeed", "index", "information", "instead", "into", "invention", "inward", "is", "isn't", "it", "itd", "it'll", "its", "itself", "i've", "j", "just", "k", "keep  keeps", "kept", "kg", "km", "know", "known", "knows", "l", "largely", "last", "lately", "later", "latter", "latterly", "least", "less", "lest", "let", "lets", "like", "liked", "likely", "line", "little", "'ll", "look", "looking", "looks", "ltd", "m", "made", "mainly", "make", "makes", "many", "may", "maybe", "me", "mean", "means", "meantime", "meanwhile", "merely", "mg", "might", "million", "miss", "ml", "more", "moreover", "most", "mostly", "mr", "mrs", "much", "mug", "must", "my", "myself", "n", "na", "name", "namely", "nay", "nd", "near", "nearly", "necessarily", "necessary", "need", "needs", "neither", "never", "nevertheless", "new", "next", "nine", "ninety", "no", "nobody", "non", "none", "nonetheless", "noone", "nor", "normally", "nos", "not", "noted", "nothing", "now", "nowhere", "o", "obtain", "obtained", "obviously", "of", "off", "often", "oh", "ok", "okay", "old", "omitted", "on", "once", "one", "ones", "only", "onto", "or", "ord", "other", "others", "otherwise", "ought", "our", "ours", "ourselves", "out", "outside", "over", "overall", "owing", "own", "p", "page", "pages", "part", "particular", "particularly", "past", "per", "perhaps", "placed", "please", "plus", "poorly", "possible", "possibly", "potentially", "pp", "predominantly", "present", "previously", "primarily", "probably", "promptly", "proud", "provides", "put", "q", "que", "quickly", "quite", "qv", "r", "ran", "rather", "rd", "re", "readily", "really", "recent", "recently", "ref", "refs", "regarding", "regardless", "regards", "related", "relatively", "research", "respectively", "resulted", "resulting", "results", "right", "run", "s", "said", "same", "saw", "say", "saying", "says", "sec", "section", "see", "seeing", "seem", "seemed", "seeming", "seems", "seen", "self", "selves", "sent", "seven", "several", "shall", "she", "shed", "she'll", "shes", "should", "shouldn't", "show", "showed", "shown", "showns", "shows", "significant", "significantly", "similar", "similarly", "since", "six", "slightly", "so", "some", "somebody", "somehow", "someone", "somethan", "something", "sometime", "sometimes", "somewhat", "somewhere", "soon", "sorry", "specifically", "specified", "specify", "specifying", "still", "stop", "strongly", "sub", "substantially", "successfully", "such", "sufficiently", "suggest", "sup", "sure    t", "take", "taken", "taking", "tell", "tends", "th", "than", "thank", "thanks", "thanx", "that", "that'll", "thats", "that've", "the", "their", "theirs", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "thered", "therefore", "therein", "there'll", "thereof", "therere", "theres", "thereto", "thereupon", "there've", "these", "they", "theyd", "they'll", "theyre", "they've", "think", "this", "those", "thou", "though", "thoughh", "thousand", "throug", "through", "throughout", "thru", "thus", "til", "tip", "to", "together", "too", "took", "toward", "towards", "tried", "tries", "truly", "try", "trying", "ts", "twice", "two", "u", "un", "under", "unfortunately", "unless", "unlike", "unlikely", "until", "unto", "up", "upon", "ups", "us", "use", "used", "useful", "usefully", "usefulness", "uses", "using", "usually", "v", "value", "various", "'ve", "very", "via", "viz", "vol", "vols", "vs", "w", "want", "wants", "was", "wasnt", "way", "we", "wed", "welcome", "we'll", "went", "were", "werent", "we've", "what", "whatever", "what'll", "whats", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "wheres", "whereupon", "wherever", "whether", "which", "while", "whim", "whither", "who", "whod", "whoever", "whole", "who'll", "whom", "whomever", "whos", "whose", "why", "widely", "willing", "wish", "with", "within", "without", "wont", "words", "world", "would", "wouldnt", "www", "x", "y", "yes", "yet", "you", "youd", "you'll", "your", "youre", "yours", "yourself", "yourselves", "you've", "z", "zero"]
 
def preprocess_data(doc,word2vec_model,my_punctuation,min_word_length=0):
    doc = doc.lower()
    doc = doc.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation))) 
    doc = word_tokenize(doc)    
    doc = filter(lambda x: x not in my_punctuation, doc)  
    doc = filter(lambda x:not x.isdigit(), doc)
    doc = [wnl.lemmatize(w.lower()) for w in doc]
    doc = filter(lambda x:x not in stopwords, doc)
    doc = filter(lambda x: x in word2vec_model.vocab or x in ".",doc)
    doc = ' '.join(e for e in doc if len(e)>=min_word_length) # if min_word_length is not 3 then filter it out
    return doc

def vocab_filtered_data(doc,vocab):
  doc = word_tokenize(doc)
  doc = filter(lambda x: x in vocab, doc) 
  doc = ' '.join(e for e in doc)
  return doc

def docs_labels_preprocessing(docs,labels,word2vec_model,min_doc_len,min_word_length,max_features):
  data_preprocessed = []
  data_preprocessed_labels = []
  embeddings = {}

  for i in range(len(docs)):
    doc = preprocess_data(docs[i],word2vec_model,string.punctuation,min_word_length)
    data_preprocessed.append(doc)
    data_preprocessed_labels.append(labels[i])
 
  vectorizer = CountVectorizer(max_features=max_features,dtype=np.float32)
  train_vec = vectorizer.fit_transform(data_preprocessed).toarray()
  vocab = vectorizer.vocabulary_

  nonzeros_indexes = np.where(train_vec.any(1))[0]
  data_preprocessed = [data_preprocessed[i] for i in nonzeros_indexes]
  data_preprocessed_labels = [data_preprocessed_labels[i] for i in nonzeros_indexes]

  for i in range(len(data_preprocessed)):
    data_preprocessed[i] = vocab_filtered_data(data_preprocessed[i],vocab) 

  data_preprocessed_f = [data_preprocessed[i] for i in range(len(data_preprocessed)) if len(data_preprocessed[i].split(' '))>=min_doc_len]
  data_preprocessed_labels_f = [data_preprocessed_labels[i] for i in range(len(data_preprocessed)) if len(data_preprocessed[i].split(' '))>=min_doc_len]
  
  vectorizer_f = CountVectorizer(dtype=np.float32)
  vectorizer_f.fit_transform(data_preprocessed_f)
  vocab_f = vectorizer.vocabulary_

  for f in vocab:
    embeddings[f] = word2vec_model[f]
  return data_preprocessed_f,data_preprocessed_labels_f,embeddings,vocab_f
 
 
def embeddingsVectors_to_txt(embeddings,d_data):
  os.chdir(home_dir+'/content')
  with open("embeddings_"+d_data+"_"+str(len(embeddings))+".txt","w") as f:
    for a in embeddings.keys():
      f.write(str(a)+" ")
      vector= embeddings[a]
      str_vector = ' '.join([str(elem) for elem in vector])
      f.write(str_vector+'\n')
# embeddingsVectors_to_txt(embeddings,data_to_get)