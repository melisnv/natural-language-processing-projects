# libraries
import numpy as np
import pandas as pd
import csv
import nltk
import sys
import nltk
from nltk import pos_tag
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')


def read_data():
    # Read the contents of the file
    df = pd.read_csv('./datas/SEM-2012-SharedTask-CD-SCO-training-simple.v2.txt', delimiter='\t',
                     names=['file_name', 'sentence_num', 'word_number', 'word', 'coreference_label'])
    return df


# lemmatizing
def lemmatization_feature(tokens):
    '''
    This function applies lemmatization to tokens

    :returns: a list with lemmatized tokens
    '''
    wnl = WordNetLemmatizer()
    lemmas = []

    for t in tokens:
        lemmas.append(wnl.lemmatize(t))

    return lemmas


def stemming_feature(tokens):
    '''
    This function applies stemming to tokens

    :returns: a list with stemmized tokens
    '''
    ps = PorterStemmer()
    stemm = []

    for t in tokens:
        stemm.append(ps.stem(t))

    return stemm


def lowercasing(tokens):
    '''
    This function checks whether tokens are capitalized or not
    :param tokens: the list of tokenized data
    :type tokens: list

    :returns: provides list which 0 (not capitalized) and 1 (capitalized) for tokens
    '''

    lowercase = []

    for t in tokens:
        lowercase.append(t.lower())

    return lowercase


def pos_tagging(tokens):
    # POS tagging
    pos_tag = tokens.apply(lambda x: nltk.pos_tag([x])[0][1]).tolist()

    return pos_tag


def find_syn_ant(word, pos, syn_ant):
    if pos == '':
        return []
    synsets = wordnet.synsets(word, pos=pos)
    if syn_ant == 'syn':
        return [syn.lemmas()[0].name() for syn in synsets]
    elif syn_ant == 'ant':
        return [ant.lemmas()[0].name() for syn in synsets for ant in syn.lemmas()[0].antonyms()]
    else:
        return []


def change_context(df, syn_ant):
    def get_wordnet_pos(treebank_tag):
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return ''

    df["syn_ant"] = df["word"].apply(
        lambda x: (lambda pos: find_syn_ant(x, pos, syn_ant))(get_wordnet_pos(nltk.pos_tag([x])[0][1])))
    return df


def save_data(df):
    return df.to_csv("preprocessed_data.csv", index=False)


df = read_data()

# apply processing steps
df = change_context(df, 'syn')
df["lemma"] = lemmatization_feature(df["word"])
df["lowercase"] = lowercasing(df["word"])
df["stem"] = stemming_feature(df["word"])
df["pos_tag"] = pos_tagging(df["word"])

# save processed data
save_data(df)

