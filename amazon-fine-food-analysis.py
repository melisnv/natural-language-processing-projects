#!/usr/bin/env python
# coding: utf-8

# # Amazon Fine Food Analysis

# In[1]:


import pandas as pd
import numpy as np
import string
import spacy
from wordcloud import WordCloud
import re
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_curve, roc_auc_score

import warnings
warnings.filterwarnings("ignore")


# In[2]:


data = pd.read_csv("./datas/Reviews.csv")
data.head()


# In[3]:


len(data)


# In[4]:


data = data[:10000]


# In[5]:


data.info()


# In[6]:


data.describe()


# In[7]:


data['Score'].hist()
plt.title("Distribution of Scores")
plt.xlabel("Scores")
plt.ylabel("Number of Reviews");


# In[8]:


new_df = data[['Text','Score']]
new_df.head()


# In[9]:


new_df['Score'] = np.where(new_df['Score'] > 3, 1, 0)

new_df.drop_duplicates(inplace=True)
new_df.head()


# In[10]:


new_df = new_df.groupby('Score').sample(n = 1000, random_state = 1)


# In[11]:


new_df = new_df.sample(frac = 1).reset_index(drop = True)


# In[12]:


new_df['Text'] = new_df['Text'].str.lower()


# In[13]:


new_df.head()


# -----
# # Processing Data Before EDA
# 
# 
# ## Lemmatization and Stopwords Removal

# In[14]:


# Loading the model
nlp = spacy.load("en_core_web_sm")

# Lemmatization with stopwords removal
new_df['Lemmatized_Text'] = new_df['Text'].apply(lambda x: ' '.join([token.lemma_ for token in list(nlp(x)) if (token.is_stop==False)]))


# In[15]:


new_df.head()


# #### Removing HTML tags, punctuations and numerical vals

# In[16]:


new_df['Lemmatized_Text'] = new_df['Lemmatized_Text'].replace(to_replace ='<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});',
                                                             value= ' ', regex=True)


# ####  Punctuation removal

# In[17]:


new_df['Lemmatized_Text'] = new_df['Lemmatized_Text'].apply(lambda x : re.sub('[%s]' %re.escape(string.punctuation),'',x))


# #### Number removal

# In[18]:


new_df['Lemmatized_Text'] = new_df['Lemmatized_Text'].apply(lambda x : re.sub('\w*\d\w*','', x))


# In[19]:


new_df['Lemmatized_Text'][:5]


# In[20]:


new_df['Text'] = [i for i in new_df['Lemmatized_Text'].str.split()]
new_df['Text'] = new_df['Text'].apply(' '.join)
new_df['Text']


# In[21]:


new_df.drop('Lemmatized_Text', axis=1, inplace=True)
new_df.head()


# ----
# ### Most frequent words
# 
#     Creating a document term matrix, which holds the frequency of words present in the corpus. TF-IDF vectorizer will be used to make this matrix. In case of positive words which are more than one word length, n-grams will be used.

# In[22]:


tfidf = TfidfVectorizer(analyzer='word',ngram_range=(2,3))
data_p = tfidf.fit_transform(new_df[new_df['Score'] == 1]['Text']) # positive reviews
dtm_p = pd.DataFrame(data_p.toarray(), columns=tfidf.get_feature_names())
dtm_p.index = new_df[new_df['Score'] == 1].index
dtm_p


# ### Top 10 positive reviews

# In[23]:


dtm_p.sum().nlargest(10).plot.bar()
plt.title("Top 10 most frequent words in postive reviews")
plt.xlabel("Words")
plt.ylabel("Occurence")
print(list(dtm_p.sum().nlargest(10).index))


# ### Top 10 negative reviews

# In[24]:


tfidf = TfidfVectorizer(analyzer='word',ngram_range=(4,4))
data_n = tfidf.fit_transform(new_df[new_df['Score'] == 0]['Text']) # negative reviews
dtm_n = pd.DataFrame(data_n.toarray(), columns=tfidf.get_feature_names())
dtm_n.index = new_df[new_df['Score'] == 0].index
dtm_n


# In[25]:


dtm_n.sum().nlargest(10).plot.bar()
plt.title("Top 10 most frequent words in negative reviews")
plt.xlabel("Words")
plt.ylabel("Occurence")
print(list(dtm_n.sum().nlargest(10).index))


# In[26]:


top_positive_words = list(dtm_p.sum().nlargest(10).index)
print(top_positive_words)

for top_words in top_positive_words:
    positive_sentences_with_topwords = []
    
    for pos_sent in new_df[new_df['Score'] == 1]['Text']:
        if top_words in pos_sent:
            positive_sentences_with_topwords.append(pos_sent.replace(top_words,''))
            
    df = pd.DataFrame(positive_sentences_with_topwords,columns=['Txt'])
    
    tfidf = TfidfVectorizer(analyzer='word', ngram_range=(2,3))
    data_pcw = tfidf.fit_transform(df['Txt'])
    dtm_pcw = pd.DataFrame(data_pcw.toarray(), columns=tfidf.get_feature_names())
    dtm_pcw.index = df.index
    
    lst = list(dtm_pcw.sum().nlargest(15).index)
    print(top_words,' : ', lst,'\n')


# #### Character count distribution in Text column

# In[27]:


arr_length = [i for i in range(2000)]

char_count = [len(i) for i in new_df['Text']]
plt.scatter(arr_length,char_count);


# In[28]:


np.percentile(char_count,98.48)


# Majority of the sentences have a character count `less than 916 characters` (98.48% of time).

# ### Word Cloud for Positive Sentences

# In[29]:


def generate_word_cloud(text):
    
    stop_words = nlp.Defaults.stop_words
    
    wordcloud = WordCloud(stopwords=stop_words,background_color='white',collocation_threshold=3)
    wordcloud.generate(text)
    plt.figure(figsize=(15,7))
    plt.axis('off')
    plt.imshow(wordcloud,interpolation='bilinear')
    return plt.show()


# In[30]:


positive_sentences = new_df.loc[new_df.Score == 1].Text
text = " ".join(review for review in positive_sentences.astype(str))

generate_word_cloud(text)


# In[31]:


negative_sentences = new_df.loc[new_df.Score == 0].Text
text = " ".join(review for review in negative_sentences.astype(str))

generate_word_cloud(text)


# ----
# # Training and Testing

# In[32]:


def get_vector(x):
    
    """
    A function to convert sentence into vector using spaCy's word2vec vectorizer.
    """
    
    doc = nlp(x)
    vect = doc.vector
    
    return vect


# In[33]:


new_df['Vector'] = new_df['Text'].apply(lambda x: get_vector(x))
new_df.head()


# In[34]:


X = new_df['Vector'].to_numpy()
X = X.reshape(-1,1)
X.shape


# In[35]:


X = np.concatenate(np.concatenate(X,axis=0),axis=0).reshape(-1,300)
X.shape


# # Split the Data

# In[42]:


y = new_df.Score[:640]

# splitting the data into training and testing with 75:25 split ratio
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25)


# In[45]:


print("Shape of training data X: ",X_train.shape)
print("Shape of testing data X: ",X_test.shape)


# ## Training a Logistic Regression Model

# In[46]:


# load model and set parameters
model = LogisticRegression(C=1, solver='liblinear', random_state=1)

# fitting the model
model.fit(X_train,y_train)


# In[48]:


print("Training Accuracy: ",model.score(X_train,y_train))
print("Testing Accuracy: ",model.score(X_test, y_test))


# In[50]:


y_prediction = model.predict(X_test)

print("Precision score on test data: ",precision_score(y_test,y_prediction))
print("Recall score on test data: ",recall_score(y_test,y_prediction))
print("F1 score on test data: ",f1_score(y_test,y_prediction))


# In[51]:


fpr, tpr, _ = roc_curve(y_test,  y_prediction)
auc = roc_auc_score(y_test, y_prediction)
plt.plot(fpr,tpr,label="auc="+str(auc))
plt.legend(loc=4)
plt.show()


# In[53]:


confusion_matrix(y_test, y_prediction) # confusion matrix

