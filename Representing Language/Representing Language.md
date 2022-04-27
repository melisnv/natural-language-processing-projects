# Week 3 : Representing Language

### Distributional Hypothesis

The fact that, for example, not every adjectives occurs with every noun can be used as a measure of meaning difference. For it is not merely that different members of the one class have different selections of members of the other class with which they are actually found. More than that: if we consider words or morphemes A and B to be more different than A and C , then we will often find that
the distributions of A and B are more different than the distributions of A and C . In other words, difference in meaning correlates with difference in distribution.

### Co-occurrence Counts

We can model the distributional hypothesis using co-occurrence counts to capture in which contexts a word appears. Context is modeled using a window over the words. If we collect co-occurrence counts over thousands of sentences, the vectors for `enjoy` and `like` will have very similar vector representations.
However, real data has a huge vocabulary and vectors become very large and contain many zeros. Therefore, we need to apply dimensionality reduction.

## Word Embeddings

Train a neural network that learns to predict a word given the surrounding context and predict the surrounding context given a word. The network will learn to represent similar words (words which occur in similar contexts) with similar representations/vectors/embeddings. Similar words are expected to be close to each other in vector space. Many variants of word embeddings have been developed and have boosted the performance of almost all NLP models. The problem is that there is only one vector per word but words can have multiple meanings.

To represent the complexity of a typical 50,000 word English vocabulary requires hundreds of features. Designing all those features by hand, and assigning accurate coordinates to all those words, would be a lot of work! Instead we can let the computer create the feature space for us by supplying a machine learning algorithm with a large amount of text, such as all of Wikipedia, or a huge collection of news articles. The algorithm discovers statistical relationships between words by looking at what other words they co-occur with. It uses this information to create word representations in a semantic feature space of its own design. These representations are called word embeddings. A typical embedding might use a 300 dimensional space, so each word would be represented by 300 numbers. The most significant application of word embeddings is to encode words for use as input to complex neural networks that try to understand the meanings of entire sentences, or even paragraphs. One such class of networks are called transformer neural networks. Two famous transformer networks are BERT from Google and GPT3 from OpenAI. BERT now handles many Google searches.

## Contextualized Language Models

Traditionally, the task of a language model was to predict next word (auto-completion) and to calculate probability of a sequence.

Unigram model: P(Lisa sings X) = P(X|sings) × P(sings|Lisa) × P(Lisa|<BOS>)
Bigram model: P(X|Lisa sings)× P(sings |<BOS> Lisa)

Words are mapped into token ids which are mapped into vectors using weight matrix `W`. W is initialized randomly but fixed which means the same token is mapped into the same vector. We apply the softmax to the output vector to obtain a probability distribution over all tokens in the vocabulary.

## RNN Architectures

The architecture for calculating the hidden representations can be way more complex. For example, multiple stacked LSTM layers with dropout. If you want to familiarize yourself with LSTMs: train a very basic LSTM language model. Output the intermediate variables, play around with the parameters and make sure you understand what is happening.

### Language Modelling as Pre-training

Language modelling is a task that processes `only the raw text`, no annotations are required. Next word prediction requires both `semantic` and `syntactic knowledge`. Semantic knowledge is the aspect of language knowledge that involves word meanings/vocabulary. Syntactic language knowledge is the knowledge of how words can be combined in meaningful sentences, phrases, or utterances.

# Transformers as Language Models

The transformer architecture works much better for machine translation than LSTMs. Attention is a learned vector that was developed for encoder-decoder tasks. For each decoding step, the attention vector indicates which parts of the input representation are most relevant. The encoder yields a representation of the input sentence in language 1. The decoder tries to consecutively generate the output tokens in language 2. Not all tokens in the input sentence are equally relevant for generating the correct output token. The `attention vector a` indicates a `weight` for each token in the input. The attention vector is different at each decoding `step t`.

Bigger training datasets generally lead to better models. In order to improve efficiency during training we can let go of the sequence constraints since this makes it easier to parallelize computation within sentences.

## Self-Attention

Model relationships between words in a sequence. Each word in a sequence can attend to every other word in the same sequence. Self-attention is a mechanism to attend to token representations in the same
sequence (not only in the encoder).Each input xi is multiplied with three different weight vectors:
`query: qi = WQx1 key: ki = WKx1 value: vi = WVx1`
The attention weights are calculated by combining queries and keys for each element. Every element of an input sequence is considered as a query and every other element as a key. The attention between element i and j is calculated by combining query i with keys: `scoreij = qi · kj`

## Multi-Head Attention

Attention depends on the perspective. If the network learns multiple attention heads (= multiple sets of Q, K, V matrices), it might learn to attend to different aspects of language (e.g. local semantics, grammatical gender agreement, coreference resolution, number agreement, semantic roles, ...). It is not yet clear, if the attention heads actually work that way, but they do have an important role in the good performance of transformers.

## Training Objective

`Masked Language Modelling` (MLM) using 15% of the words in the training data for learning. The model should learn to predict these words. It can attend to any other word in the sentence.

“The training data generator chooses 15% of the token positions at random for prediction. If the i-th token is chosen, we replace the i-th token with (1) the [MASK] token 80% of the time (2) a random token 10% of the time (3) the unchanged i-th token 10% of the time.”. This is for `robustness`.

## Feature Extraction vs Fine-tuning

BERTbase returns a different sentence representation for every layer which is a concatenation of the token representations in this layer. A token is represented differently depending on the context in which it occurs. A token is represented differently in every layer.

We can extract sentence representations to use for classification by xperimenting with different layers and take the best one and by averaging/summing/concatenating over (a subset of) layers.
To have a fixed number of dimensions independent of the sentence length we can averaging/summing over the token representations and can only take the CLS token.

The CLS token can be interpreted as a compressed sentence representation. It can be used for directly fine-tuning BERT for another task. It is often sufficient to add a single feed-forward layer to BERT.
During fine-tuning, the parameters of this layer are optimized, but the parameters in the pre-trained BERT model might also be subject to small changes. `Fine-tuning` often requires only a few training epochs and standard hardware. The pre-trained models were trained for weeks with a training cost of several million dollars. Fine-tuning is also known as transfer learning.

## References

- Jacob Eisenstein (2018) - Natural Language Processing Notes (https://github.com/jacobeisenstein/gt-nlp-class/blob/master/notes/eisenstein-nlp-notes.pdf) <br/>
- Vrije Universiteit Amsterdam NLP Lecture Notes
