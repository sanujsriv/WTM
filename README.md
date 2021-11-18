# WTM - A Word Embedding Topic Model for Robust Inference of Topics and Visualization
This is the code for Word Embedding Topic Model (WTM).

WTM is a robust topic model that uses word embeddings to solve the data sparsity problem encountered while detecting topics and generating
visualization of short texts. 

# Environment
Tested on :-

**Local Machine -**
<ul>
  <li>Python 3.8.5</li>
  <li>Pytorch 1.7.1+cu110</li>
</ul>

**Colab -**
<ul>
  <li>Python 3.7.10</li>
  <li>Pytorch 1.9.0+cu102</li>
</ul>

# Preprocessing
The scripts used for preprocessing the data can be found in the folder "preprocessing".
## Datasets
The preprocessed example dataset can be found in the folder named "content". It follows the directory structure - content/data_{dataname}/short/**.pkl**. 
## Data Preprocessing
If you want to run WTM on your own dataset you can follow the script for preprocessing i.e. **preprocessing.py** or use your own. It is recommended to follow the preprocessing steps given in the **preprocessing.py**

## Generating Embeddings
WTM can also run on generated embeddings. It uses *skipgram* technique by default to generate embeddings.The script for generating the embeddings could be found in the **preprocessing.py**. You can either follow that or generate your own embeddings.

# Running WTM
You can directly pass *bbc* to run WTM on bbc dataset.
```  
python3 main.py --dataset bbc --num_topics 10 -e 1000 -drop 0.2
```
<Run WTM on bbc dataset with 10 topic for 1000 epochs with 0.2 dropout rate> <br/>

## Run WTM with Generated embeddings
To run WTM with generated embeddings you must place the newly generated embeddings file (pickle file) to the data content directory and then run the model with parameter *skipgram_embeddings 1* as -
```  
python3 main.py --dataset bbc --num_topics 10 -e 1000 --skigram_embeddings 1 -drop 0.2
```
## Run WTM to generate visualization
To generate the visualization produced by WTM do -
```  
python3 main.py --dataset bbc --num_topics 10 -e 1000 --skigram_embeddings 1 --visualize True -drop 0.2
```

# Visualizations
Here is an example visualization produced by WTM for searchsnippet - <br/>
![ssnip_vis](/visualizations/searchsnippet_WTM.png)
![ssnip_label](/visualizations/searchsnippet_WTM_label.png)

# Citation 
```
@inproceedings{kumar2021word,
  title={A Word Embedding Topic Model for Robust Inference of Topics and Visualization},
  author={Kumar, Sanuj and Le, Tuan},
  booktitle={The First International Conference on AI-ML-Systems},
  pages={1--7},
  year={2021}
}
```
