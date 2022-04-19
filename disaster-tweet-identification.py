#!/usr/bin/env python
# coding: utf-8

# # Disaster Tweet Identification

# In[1]:


import re
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import re
import nltk
from nltk.corpus import stopwords

import warnings
warnings.filterwarnings("ignore")
from collections import defaultdict


# In[2]:


data = pd.read_csv("./datas/disaster_tweets.csv")
data.head()


# In[3]:


data.tail()


# In[4]:


data.info()


# In[5]:


data.describe().T


# In[15]:


data["keyword"].unique()


# ## Data Cleaning and Text Processing

# In[6]:


nltk.download("stopwords")

URL_PATTERN = '((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*'
stopwords = stopwords.words("english")


# `sub()` function searches for the pattern in the string and replaces the matched strings with the replacement *(repl)*.if couldn't find a match, it returns the original string.

# In[7]:


def clean_text(text):
    
    # remove stopwords
    remove_stopwords = ' '.join([word for word in text.split() if word not in stopwords])
    # remove URL
    remove_url = re.sub(URL_PATTERN,'',remove_stopwords)
    # remove punctuation
    remove_punctuation = re.sub(r'[^\w\s]','',remove_url)
    
    return remove_punctuation.lower()


# In[8]:


data['cleaned_text'] = data['text'].apply(lambda x : clean_text(x))


# In[9]:


print(f"Before text cleaning: \n{data.text[100]}")
print("\n")
print(f"After cleaning the text: \n{data.cleaned_text[100]}")


# ### Corpus and Word Frequency Dictionaries

# In[10]:


def create_freq_dict(string):
    freqency_dictionary = defaultdict(int)
    
    for word in string.split():
        if word not in freqency_dictionary:
            freqency_dictionary[word] = 1
        else:
            freqency_dictionary[word] += 1
            
    
    return freqency_dictionary


# In[11]:


positive_corpus = ' '.join(text for text in data[data["target"] == 1]["cleaned_text"])
negative_corpus = ' '.join(text for text in data[data["target"] == 0]["cleaned_text"])

positive_frequency_dictionary = create_freq_dict(positive_corpus)
negative_frequency_dictionary = create_freq_dict(negative_corpus)


# In[12]:


def frequency(freqency_dictionary,text):
    frequency = 0
    
    for word in text.split():
        frequency += freqency_dictionary[word]
        
    return frequency


# In[13]:


data["positive_freq"] = data["cleaned_text"].apply(lambda text : frequency(positive_frequency_dictionary,text))
data["negative_freq"] = data["cleaned_text"].apply(lambda text : frequency(negative_frequency_dictionary,text))

data.head()


# ## Splitting the data to train set and validation set

# In[16]:


def split_the_data(features,labels,split_size):
    
    train_size = int(len(features) * split_size)
    
    data = list(zip(features,labels))
    shuffle_data = random.sample(data,len(data))
    
    shuffle_labels = [label for feature,label in shuffle_data]
    shuffle_features = [feature for feature,label in shuffle_data]
    
    X_train = np.array(shuffle_features[:train_size])
    y_train = np.array(shuffle_labels[:train_size]).reshape((len(shuffle_labels[:train_size]),1))
    
    X_test = np.array(shuffle_features[train_size:])
    y_test = np.array(shuffle_labels[train_size:]).reshape((len(shuffle_labels[train_size:]),1))
    
    return X_train,X_test,y_train,y_test


# In[ ]:




