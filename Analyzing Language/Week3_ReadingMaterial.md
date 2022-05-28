![]()

![]()

## Transformer's Architecture

_I would like to **drink** a very hot tall decaf half-soy (...) white chocolate **mocha**._<br>

In this sentence not all elements are equally important to remember._(Selective `attention`)_ `Attention` is a learned vector that was developed for _encoder-decoder_ tasks. For each decoding step, attention vector indicates which parts of the input representation are most relevant. The `attention vector` **a** indicates a weight for each token in the input.

![Attention Vector a](https://firebasestorage.googleapis.com/v0/b/birthday-react-6eca4.appspot.com/o/Natural-Language-Processing%2FWeek%202%20%3A%20Classifying%20Language%2Fvectora.jpg?alt=media&token=220ebc48-66e2-4223-99da-a3a7a6ffff06)<br>

`Self Attention` is a mechanism to attend to token representations in the same sequence _(not only in the encoder)_. Each input x_i is multiplied with three different weight vectors: <br>

_query :_ q*i = W^Q_x1 \_key :* k*i = W^K_x1 \_value :* v_i = W^V_x1 <br>

The attention weights are calculated by combining queries and keys for each element. Every element of an input sequence is considered as a query and every other element as a key. The attention between element _i_ and _j_ is calculated by combining query _i_ with keys:<br>

score_ij = q_i \* k_j `(Attention for i with respect to j)` <br>

Then **softmax function** applied to the score in order to make it normalize.<br>

If the network şearns `multi attention heads` _(multiple sets of Q,K,V matrices)_,it might learn to attend to different aspects of language (e.g. local semantics, grammatical gender agreement, coreference resolution, number agreement, semantic roles,...)<br>

![BERT](https://firebasestorage.googleapis.com/v0/b/birthday-react-6eca4.appspot.com/o/Natural-Language-Processing%2FWeek%202%20%3A%20Classifying%20Language%2Fembed.jpg?alt=media&token=2101d2ef-2ca8-4f6f-aa29-c6797e95021b)

`Sentence Embedding` tells us whether this token belongs to the first sentence or second sentence.

### Fine-tuning(transfer learning) BERT

The `CLS` token can be interpreted as a compressed sentence representation. It can be used for directly fine-tuning BERT for another task. It if often sufficient to add a **single feed-forward layer** to BERT.During fine-tuning, the parameters of this layer are optimized, but the parameters in the pre-trained BERT model might also be subject to small changes.

# Vector Semantics and Embeddings (chapter 6)

Words that occur in similar contexts tend to have similar meanings. This link between similarity in how words are distributed and similarity in what they mean is called the `distributional hypothesis`. Words which are synonyms (like _oculist_ and _eye-doctor_) tended to occur in the same environment (e.g., near words like _eye_ or _examined_). In this chapter we introduce `vector semantics`, which instantiates this linguistic hypothesis by learning representations of the meaning of words, called embeddings, directly from their distributions in texts. These representations are used in every natural language processing application that makes use of meaning, and the static embeddings we introduce here underlie the more powerful dynamic or **contextualized embeddings** like `BERT`.

### 6.1 Lexical Semantics

**mouse** (N)<br>

1. any of numerous small rodents...<br>
2. a hand-operated device that controls a cursor...<br>

Here the form _mouse_ is the `lemma`, also called the _citation form_. The form citation form mouse would also be the **lemma** for the word _mice_; dictionaries don’t have separate definitions for inflected forms like mice. Similarly sing is the lemma for sing, sang, sung. The specific forms wordform sung or carpets or sing or mice are called `wordforms`.<br>

One common kind of relatedness between words is if they belong to the same _semantic field_. A `semantic field` is a set of words which cover a particular semantic domain and bear structured relations with each other. For example, words might be related by being in the semantic field of hospitals (surgeon, scalpel, nurse, anesthetic, hospital), restaurants (waiter, menu, plate, food, chef), or houses (door, roof, kitchen, family, bed).<br>

_Positive_ or _negative_ evaluation language is called `sentiment`, and word sentiment plays a role in important tasks like sentiment analysis, stance detection, and applications of NLP to the language of politics and consumer reviews.

### 6.2 Vector Semantics

_Vectors semantics_ is the standard way to represent word meaning in NLP. To define the meaning of a word by its _distribution_ in language use, meaning its neighboring words or grammatical environments. Their idea was that two words that occur in very similar distributions (whose neighboring words are similar) have similar meanings. The idea of vector semantics is to represent a word as a point in a multidimensional semantic space that is derived (in ways we’ll see) from the distributions of word neighbors. _Vectors_ for representing words are called `embeddings`. By representing words as embeddings, classifiers can assign sentiment as long as it sees some words with similar meanings. And as we’ll see, vector semantic models can be learned automatically from text without supervision.

### 6.3 Words and Vectors

Vector or distributional models of meaning are generally based on a co-occurrence matrix, a way of representing how often words co-occur. We’ll look at two popular matrices: _the term-document matrix_ and _the term-term matrix_. In a `term-document matrix`, each row represents a word in the vocabulary and each column represents a document from some collection of documents. Term-document matrices were originally defined as a means of finding similar documents for the task of document **information retrieval**. Two documents that are similar will tend to have similar words, and if two documents have similar words their column vectors will tend to be similar. _Information retrieval_ (IR) is the task of finding the document d from the D documents in some collection that best matches a query q.<br>

For documents, we saw that similar documents had similar vectors, because similar documents tend to have similar words. This same principle applies to words: similar words have similar vectors because they tend to occur in similar documents. The term-document matrix thus lets us represent the meaning of a word by the documents it tends to occur in.<br>

To measure similarity between two target words v and w, we need a metric that takes two vectors. By far the most common similarity metric is the _cosine_ of the angle between the vectors.

## Transfer Learning with Pretrained Language Models and Contextual Embeddings (chapter 11)

First, we’ll introduce the idea of `contextual embeddings`: representations for words in context. Second, we’ll introduce in this chapter the idea of `pretraining` and `fine-tuning`. We call pretraining the process of learning some sort of representation of meaning for words or sentences by processing very large amounts of text. We’ll call these pretrained models pretrained language models, since they can take the form of the transformer language models. We call `fine-tuning` the process of taking the representations from these pretrained models, and further training the model, often via an added neural net classifier, to perform some downstream task like named entity tagging or question answering or coreference. The intuition is that the pretraining phase learns a language model that instantiates a rich representations of word meaning, that thus enables the model to more easily learn (_‘be fine-tuned to’_) the requirements of a downstream language understanding task.

In this chapter we’ll introduce a second paradigm, called the `bidirectional transformer encoder`, and the method of `masked language modeling`, introduced with the BERT model.

## 11.1 Bidirectional Transformer Encoders

When applied to sequence classification and labeling problems causal models have obvious shortcomings since they are based on an incremental, left-to-right processing of their inputs. If we want to assign the correct named-entity tag to each word in a sentence, or other sophisticated
linguistic labels, we’ll want to be able to take into account information from the right context as we process each element.<br>

As can be seen, the hidden state computation at each point in time is based solely on the current and earlier elements of the input, ignoring potentially useful information located to the right of each tagging decision.

![Birdirectional vs feed-forward model](https://firebasestorage.googleapis.com/v0/b/birthday-react-6eca4.appspot.com/o/Natural-Language-Processing%2FWeek%202%20%3A%20Classifying%20Language%2Fselfattention.jpg?alt=media&token=74f4c88c-eac1-4819-9335-a09387be5213)

### 11.2 Training Bidirectional Encoders

Fortunately, the traditional learning objective suggests an approach that can be used to train bidirectional encoders. Instead of trying to predict the next word, the model learns to cloze task perform a fill-in-the-blank task, technically called the cloze task.

That is, given an input sequence with one or more elements missing, the learning task is to predict the missing elements. More precisely, during training the model is deprived of one or more elements of an input sequence and must generate a probability distribution over the vocabulary for each of the missing items. We then use the cross-entropy loss from each of the model’s predictions to drive the learning process. This approach can be generalized to any of a variety of methods that corrupt the training input and then asks the model to recover the original input. Examples of the kinds of manipulations that have been used include masks, substitutions, reorderings, deletions, and extraneous insertions into the training text.

### 11.2.1 Masking Words

The original approach to training bidirectional encoders is called `Masked Language Modeling` (_MLM_). MLM uses unannotated text from a large corpus. Here, the model is presented with a series of sentences from the training corpus where a random sample of tokens from each training sequence is selected for use in the learning task. Once chosen, a token is used in one of three ways:<br>

• It is replaced with the unique vocabulary token `[MASK]`.<br>
• It is replaced with another token from the vocabulary, randomly sampled
based on token unigram probabilities.<br>
• It is left unchanged.<br>

![Encoder](https://firebasestorage.googleapis.com/v0/b/birthday-react-6eca4.appspot.com/o/Natural-Language-Processing%2FWeek%202%20%3A%20Classifying%20Language%2Fencoder.jpg?alt=media&token=ae94c45d-d2ce-43e8-a5e8-c067f73a4ae0)

Here, long, thanks and the have been sampled from the training sequence, with the first two masked and the replaced with the randomly sampled token _apricot_. The resulting embeddings are passed through a stack of bidirectional transformer blocks. To produce a probability distribution over the vocabulary for each of the masked tokens, the output vector from the final transformer layer for each of the masked tokens is multiplied by a learned set of classification weights and then through a _softmax_ to yield the required predictions over the vocabulary.

### 11.2.2 Masking Spans

A span is a contiguous sequence of one or more words selected from a training text, prior to subword tokenization. In span-based masking, a set of randomly selected spans from a training sequence are chosen. Here the span selected is and thanks for which spans from position 3 to 5. The total loss associated with the masked token thanks is the sum of the cross-entropy loss generated from the prediction of thanks from the output y_4, plus the cross-entropy loss from the prediction of thanks from the output vectors for y_2, y_6 and the embedding for position 4 in the span.

![Masking](https://firebasestorage.googleapis.com/v0/b/birthday-react-6eca4.appspot.com/o/Natural-Language-Processing%2FWeek%202%20%3A%20Classifying%20Language%2Fmask.jpg?alt=media&token=5a0d355f-e7c1-41ff-b7ef-c94bea0df34f)

### 11.2.3 Next Sentence Prediction

In this task, the model is presented with pairs of sentences and is asked to predict whether each pair consists of an actual pair of adjacent sentences from the training corpus or a pair of unrelated sentences. In BERT, 50% of the training pairs consisted of positive pairs, and in the other 50% the second sentence of a pair was randomly selected from elsewhere in the corpus. The NSP loss is based on how well the model can distinguish true pairs from random pairs. To facilitate NSP training, BERT introduces two new tokens to the input representation (tokens that will prove useful for fine-tuning as well). After tokenizing the input with the subword model, the token `[CLS]` is prepended to the input sentence pair, and the token `[SEP]` is placed between the sentences and after the final token of the second sentence. Finally, embeddings representing the first and second segments of the input are added to the word and positional embeddings to allow the model to more easily distinguish the input sentences. During training, the output vector from the final layer associated with the `[CLS]` token represents the next sentence prediction.

### 11.2.5 Contextual Embeddings

These `contextual embeddings` can be used as a contextual representation of the meaning of the input token for any task requiring the meaning of word. Contextual embeddings are thus vectors representing some aspect of the meaning of a token in context. Where static embeddings represent the meaning of word types (vocabulary entries), contextual embeddings represent the meaning of word tokens: instances of a particular word type in a particular context. Contextual embeddings can thus by used for tasks like measuring the semantic similarity of two words in context, and are useful in linguistic tasks that require models of word meaning.

## 11.3 Transfer Learning through Fine-Tuning

The power of pretrained language models lies in their ability to extract generalizations from large amounts of text—generalizations that are useful for myriad downstream applications. To make practical use of these generalizations, we need to create interfaces from these models to downstream applications through a process finetuning called fine-tuning. Fine-tuning facilitates the creation of applications on top of pretrained models through the addition of a small set of application-specific parameters. The fine-tuning process consists of using labeled data from the application to train these additional application-specific parameters. Typically, this training will either freeze or make only minimal adjustments to the pretrained language model parameters.

### 11.3.1 Sequence Classification

Sequence classification applications often represent an input sequence with a single consolidated representation. With RNNs, we used the hidden layer associated with the final input element to stand for the entire sequence. A similar approach is used with transformers. An additional vector is added to the model to stand for the entire sequence. This vector is sometimes called the sentence embedding since it refers to the entire sequence.
