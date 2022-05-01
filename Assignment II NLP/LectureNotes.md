# Vector Semantics and Embeddings

## Chapter 6

Words that occur in similar contexts tend to have similar meanings. This link between similarity in how words are distributed and similarity distributional in what they mean is called the `distributional hypothesis`. The hypothesis was first formulated in the 1950s by linguists like Joos (1950), Harris (1954), and Firth (1957), who noticed that words which are synonyms (like oculist and eye-doctor) tended to occur in the same environment (e.g., near words like eye or examined) with the amount of meaning difference between two words `“corresponding roughly to the amount of difference in their environments"`(Harris, 1954, 157).

`Vector semantics`, which instantiates this linguistic hypothesis by learning representations of the meaning of words, called embeddings, directly from their distributions in texts. These representations are used in every natural language processing application that makes use of meaning, and the static embeddings we introduce here underlie the more powerful dynamic or contextualized embeddings like `BERT`. Finding such self-supervised ways to learn representations of the input, instead of creating representations by hand via feature engineering, is an important focus of NLP research (Bengio et al., 2013).

## Lexical Semantics

A model of word meaning should allow us to draw inferences to address meaning-related tasks like question-answering or dialogue. One important component of word meaning is the relationship between
word senses. For example when one word has a sense whose meaning is identical to a sense of another word, or nearly identical, we say the two senses of those two words are `synonyms`. Synonyms include such pairs as _couch/sofa_ _vomit/throw up_ _filbert/hazelnut_ _car/automobile_. A more formal definition of synonymy (between words rather than senses) is that two words are synonymous if they are substitutable for one another in any sentence without changing the truth conditions of the sentence, the situations in which the sentence would be true. We often say in this case that the two words have the same `propositional meaning`.

The notion of word `similarity` is very useful in larger semantic tasks. Knowing how similar two words are can help in computing how similar the meaning of two phrases or sentences are, a very important component of tasks like question answering, paraphrasing, and summarization.

### Connotation

Words have affective meanings or `connotations`. The word connotation is an aspects of a word’s meaning that are related to a writer or reader’s emotions, sentiment, opinions, or evaluations. For example some words have positive connotations (happy) while others have negative connotations (sad). Even words whose meanings are similar in other ways can vary in connotation; consider the difference in connotations between _fake, knockoff, forgery_, on the one hand, and _copy, replica, reproduction_ on the other, or **innocent** _(positive connotation)_ and **naive** _(negative connotation)_. Some words describe positive evaluation _(great, love)_ and others negative evaluation (terrible, hate). Positive or negative evaluation language is called `sentiment`, and word sentiment plays a role in important tasks like sentiment analysis, stance detection, and applications of NLP to the language of politics and consumer reviews.

Early work on affective meaning (Osgood et al., 1957) found that words varied along three important dimensions of affective meaning: <br>
**valence:** the pleasantness of the stimulus<br>
**arousal:** the intensity of emotion provoked by the stimulus<br>
**dominance:** the degree of control exerted by the stimulus<br>

Thus words like happy or satisfied are high on valence, while unhappy or annoyed are low on valence. Excited is high on arousal, while calm is low on arousal. Controlling is high on dominance, while awed or influenced are low on dominance. Each word is thus represented by three numbers, corresponding to its value on each of the three dimensions. Osgood et al. (1957) noticed that in using these 3 numbers to represent the meaning of a word, the model was representing each word as a point in a threedimensional space, a vector whose three dimensions corresponded to the word’s rating on the three scales. This revolutionary idea that word meaning could be represented as a point in space (e.g., that part of the meaning of **heartbreak** can be represented as the point [2:45;5:65;3:58]) was the first expression of the vector semantics models that we introduce next.

## Vector Semantics

Vectors semantics is the standard way to represent word meaning in NLP, helping us model many of the aspects of word meaning. The roots of the model lie in the 1950s when two big ideas converged: Osgood’s 1957 idea mentioned above to use a point in three-dimensional space to represent the connotation of a word, and the proposal by linguists like Joos (1950), Harris (1954), and Firth (1957) to define the meaning of a word by its distribution in language use, meaning its neighboring words or grammatical environments. Their idea was that two words that occur in very **similar distributions** (whose neighboring words are similar) have **similar meanings**.

The idea of vector semantics is to represent a word as a point in a multidimensional semantic space that is derived from the distributions of word neighbors. Vectors for representing words are called `embeddings`. The word `“embedding”` derives from its mathematical sense as a mapping from one
space or structure to another. By representing words as embeddings, classifiers can assign sentiment as long as it sees some words with similar meanings. The two most commonly used models; **tf-idf model**, an important baseline, the meaning of a word is defined by a simple function of the counts of nearby words. This method results in very long vectors that are sparse, i.e. mostly zeros (since most words simply never occur in the context of others). The **word2vec model** family for constructing short, dense vectors that have useful semantic properties. The **cosine**, the standard way to use embeddings to compute semantic similarity, between two words, two sentences, or two documents, an important tool in practical applications like question answering, summarization, or automatic essay grading.

## Words and Vectors

In a **term-document matrix**, each row represents a word in the vocabulary and each column represents a document from some collection of documents. Term-document matrices were originally defined as a means of finding similar documents for the task of document **information retrieval**. Two documents that are similar will tend to have similar words, and if two documents have similar words their column vectors will tend to be similar. **Information retrieval (IR)** is the task of finding the document d from the D
documents in some collection that best matches a query q. For IR, therefore there is a need to represent a query by a vector of length |V|, and need a way to compare two vectors to find how similar they are.

## Cosine for Measuring Similarity

To measure similarity between two target words _v_ and _w_, a metric that takes two vectors (of the same dimensionality, either both with words as dimensions, hence of length |V|, or both with documents as dimensions as documents, of length |D|) needed and gives a measure of their similarity. By far the most common similarity metric is the **cosine** of the angle between the vectors.

## TF-IDF: Weighing Terms in the Vector

Raw frequency is very skewed and not very discriminative. If we want to know what kinds of contexts are shared by cherry and strawberry but not by digital and information, we’re not going to get good discrimination from words like the, it, or they, which occur frequently with all sorts of words and aren’t informative about any particular word.

## Pointwise Mutual Information (PMI)

PPMI draws on the intuition that the best way to weigh the association between two words is to ask how much more the two words co-occur in our corpus than we would have a priori expected them to appear by chance. It is a measure of how often two events x and y occur, compared with what we would expect if they were independent. The numerator tells us how often we observed the two words together (assuming we compute probability by using the MLE). The denominator tells us how often we would expect the two words to co-occur assuming they each occurred independently; recall that the probability of two independent events both occurring is just the product of the probabilities of the two events. Thus, the ratio gives us an estimate of how much more the two words co-occur than we expect by chance. PMI is a useful tool whenever we need to find words that are strongly associated.

### Applications of the tf-idf or PPMI vector models

In summary, the vector semantics model we’ve described so far represents a target word as a vector with dimensions corresponding either to the documents in a large collection (the term-document matrix) or to the counts of words in some neighboring window (the term-term matrix). The values in each dimension are counts, weighted by tf-idf (for term-document matrices) or PPMI (for term-term matrices), and the vectors are sparse (since most values are zero). The model computes the similarity between two words x and y by taking the cosine of their tf-idf or PPMI vectors; high cosine, high similarity. This entire model is sometimes referred to as the **tf-idf model** or the **PPMI model**, after the weighting function.

## Word2vec

It turns out that dense vectors work better in every NLP task than sparse vectors. While we don’t completely understand all the reasons for this, we have some intuitions. Representing words as 300-dimensional dense vectors requires our classifiers to learn far fewer weights than if we represented words as 50,000-dimensional vectors, and the smaller parameter space possibly helps with generalization and avoiding overfitting. Dense vectors may also do a better job of capturing synonymy. For example, in a sparse vector representation, dimensions for synonyms like car and automobile dimension are distinct and unrelated; sparse vectors may thus fail to capture the similarity between a word with car as a neighbor and a word with automobile as a neighbor.

One method for computing embeddings: **skip-gram with negative sampling**, sometimes called **SGNS**. The skip-gram algorithm is one of two algorithms in a software package called **word2vec**. Word2vec embeddings are static embeddings, meaning that the method learns one fixed embedding for each word in the vocabulary. The **intuition of word2vec** is that instead of counting how often each word w occurs near, say, apricot, we’ll instead train a classifier on a binary prediction task: _“Is word w likely to show up near apricot?”_ We don’t actually care about this prediction task; instead we’ll take the learned classifier weights as the word embeddings.
we can just use running text as implicitly supervised training data for such a classifier; a word c that occurs near the target word apricot acts as gold ‘correct answer’ to the question “Is word c likely to show up near apricot?” This method, often called self-supervision, avoids the need for any sort of hand-labeled supervision signal. This idea was first proposed in the task of neural language modeling, when Bengio et al. (2003) and Collobert et al. (2011) showed that a neural language model (a neural network that learned to predict the next word from prior words) could just use the next word in running text as its supervision signal, and could be used to learn an embedding representation for each word as part of doing this prediction task.

### The Classifier

The intuition of the skipgram model is to base this probability on embedding similarity: a word is likely to occur near the target if its embedding vector is similar to the target embedding. To compute similarity between these dense embeddings, we rely on the intuition that two vectors are similar if they have a high **dot product** (after all, cosine is just a normalized dot product).
Similarity(w,c) ~ c.w

## Learning skip-gram Embeddings

The learning algorithm for skip-gram embeddings takes as input a corpus of text, and a chosen vocabulary size N. It begins by assigning a random embedding vector for each of the N vocabulary words, and then proceeds to iteratively shift the embedding of each word w to be more like the embeddings of words that occur nearby in texts, and less like the embeddings of words that don’t occur nearby.

For training a binary classifier we also need negative examples. In fact skipgram with negative sampling **(SGNS)** uses more negative examples than positive examples (with the ratio between them set by a parameter k). So for each of these (w,cpos) training instances we’ll create k negative samples, each consisting of the target w plus a **‘noise word’** cneg. A noise word is a random word from the lexicon, constrained not to be the target word w. We’ll have 2 negative examples in the negative training set, for each positive example w,cpos.

![Intuition of one step of gradient descent.](https://firebasestorage.googleapis.com/v0/b/birthday-react-6eca4.appspot.com/o/Natural-Language-Processing%2FWeek%201%20%3A%20Analyzing%20Language%2Fembedding.jpg?alt=media&token=068e0f99-bce4-4262-a91c-19eda9950058)

The skip-gram model tries to shift embeddings so the target embeddings (here for _apricot_) are closer to (have a higher dot product with) context embeddings for nearby words (here _jam_) and further from (lower dot product with) context embeddings for noise words that don’t occur nearby (here _Tolstoy_ and _matrix_).

## Visualizing Embeddings

The simplest way to visualize the meaning of a word _w_ embedded in a space is to list the most similar words to _w_ by sorting the vectors for all words in the vocabulary by their **cosine** with the vector for _w_. Yet another visualization method is to use a _clustering algorithm_ to show a _hierarchical representation_ of which words are similar to others in the embedding space.

## Evaluating Vector Models

The most important evaluation metric for vector models is extrinsic evaluation on tasks, i.e., using vectors in an NLP task and seeing whether this improves performance over some other model. Nonetheless it is useful to have intrinsic evaluations. The most common metric is to test their performance on similarity, computing the correlation between an algorithm’s word similarity scores and word similarity ratings assigned by humans.

All embedding algorithms suffer from inherent variability. For example because of randomness in the initialization and the random negative sampling, algorithms like word2vec may produce different results even from the same dataset, and individual documents in a collection may strongly impact the resulting embeddings (Tian et al. 2016, Hellrich and Hahn 2016, Antoniak and Mimno 2018). When embeddings are used to study word associations in particular corpora, therefore, it is best practice to train multiple embeddings with bootstrap sampling over documents and average the results (Antoniak and Mimno, 2018).

## Summary

- In vector semantics, a word is modeled as a vector—a point in high-dimensional space, also called an embedding. _Static embeddings_, in each each word is mapped to a fixed embedding.

- Vector semantic models fall into two classes: **sparse** and **dense**. In _sparse_ models each dimension corresponds to a word in the vocabulary V and cells are functions of co-occurrence counts. The term-document matrix has a row for each word (term) in the vocabulary and a column for each document. The word-context or term-term matrix has a row for each (target) word in the vocabulary and a column for each context term in the vocabulary. Two sparse weightings are common: the tf-idf weighting which weights each cell by its term frequency and inverse document frequency, and PPMI (pointwise positive mutual information), which is most common for for word-context matrices.

- _Dense_ vector models have dimensionality 50–1000. **Word2vec** algorithms like skip-gram are a popular way to compute dense embeddings. **Skip-gram** trains a _logistic regression classifier_ to compute the probability that two words are ‘likely to occur nearby in text’. This probability is computed from the dot product between the embeddings for the two words.

- Skip-gram uses stochastic gradient descent to train the classifier, by learning embeddings that have a high dot product with embeddings of words that occur nearby and a low dot product with noise words.

# Transfer Learning with Pretrained Language Models and Contextual Embeddings

## Chapter 11

The idea of contextual embeddings: representations for words in context. The methods of Chapter 6 like word2vec or GloVe learned a single vector embedding for each unique word w in the vocabulary. By contrast, with contextual embeddings, such as those learned by popular methods like BERT or GPT (Radford et al., 2019) or their descendants, each word w will be represented by a different vector each time it appears in a different context.

The idea of **pretraining** and **fine-tuning**. We call **pretraining** the process of learning some sort of representation of meaning for words or sentences by processing very large amounts of text. We’ll call these pretrained models _pretrained language models_, since they can take the form of the transformer language models introduced in Chapter 9. We call **fine-tuning** the process of taking the representations from these pretrained models, and further training the model, often via an added neural net classifier, to perform some downstream task like named entity tagging or question answering or coreference. The intuition is that the pretraining phase learns a language model that instantiates a rich
representations of word meaning, that thus enables the model to more easily learn (‘be fine-tuned to’) the requirements of a downstream language understanding task.

The **pretrain-finetune paradigm** is an instance of what is called transfer learning in machine learning: the method of acquiring knowledge from one task or domain, and then applying it (transferring it) to solve a new task. There are two common paradigms for pretrained language models. One is the causal or **left-to-right transformer model** we introduced in Chapter 9. In this chapter we’ll introduce a second paradigm, called the bidirectional transformer encoder, and the method of masked language modeling, introduced with the **BERT** model (Devlin et al., 2019) that allows the model to see entire texts at a time, including both the right and left context.

## Bidirectional Transformer Encoders

Bidirectional encoders overcome limitation by allowing the self-attention mechanism to range over the entire input. The focus of bidirectional encoders is on computing contextualized representations of the tokens in an input sequence that are generally useful across a range of downstream applications. Therefore, bidirectional encoders use self-attention to map sequences of input embeddings (x1,...,xn) to sequences of output embeddings the same length (y1,...,yn), where the output vectors have been contextualized using information from the entire input sequence.

This contextualization is accomplished through the use of the same self-attention mechanism used in causal models. As with these models, the first step is to generate a set of key, query and value embeddings for each element of the input vector x through the use of learned weight matrices WQ, WK, and WV. These weights project each input vector xi into its specific role as a key, query, or value.

`qi = WQxi, ki = WKxi, vi = WVxi`

The output vector yi corresponding to each input element xi is a weighted sum of all the input value vectors v. The a weights are computed via a softmax over the comparison scores between every element of an input sequence considered as a query and every other element as a key, where the comparison scores are computed using dot products. All of the other elements of the transformer architecture remain the same for bidirectional encoder models. Inputs to the model are segmented using subword tokenization and are combined with positional embeddings before being passed through a series of standard transformer blocks consisting of self-attention and feedforward layers augmented with residual connections and layer normalization.

A fundamental issue with transformers is that the size of the input layer dictates the complexity of model. Both the time and memory requirements in a transformer grow quadratically with the length of the input. It’s necessary, therefore, to set a fixed input length that is long enough to provide sufficient context for the model to function and yet still be computationally tractable. For **BERT**, a fixed input
size of 512 subword tokens was used.

## Training Bidirectional Encoders

We trained causal transformer language models in Chapter 9 by making them iteratively predict the next word in a text. But eliminating the causal mask makes the guess-the-next-word language modeling task trivial since the answer is now directly available from the context, so we’re in need of a new training scheme. Fortunately, the traditional learning objective suggests an approach that can be used to train bidirectional encoders. Instead of trying to predict the next word, the model learns to perform a fill-in-the-blank task, technically called the **cloze task**.

During training the model is deprived of one or more elements of an input sequence and must generate a probability distribution over the vocabulary for each of the missing items. We then use the cross-entropy loss from each of the model’s predictions to drive the learning process. This approach can be generalized to any of a variety of methods that corrupt the training input and then asks the model to recover the original input. Examples of the kinds of manipulations that have been used include masks, substitutions, reorderings, deletions, and extraneous insertions into the training text.

## Masking Words

The original approach to training bidirectional encoders is called **Masked Language Modeling (MLM)**. MLM uses unannotated text from a large corpus. The model is presented with a series of sentences from the training corpus where a random sample of tokens from each training sequence is selected for use in the learning task. In **BERT**, 15% of the input tokens in a training sequence are sampled for learning.
Of these, 80% are replaced with [MASK], 10% are replaced with randomly selected tokens, and the remaining 10% are left unchanged. The MLM training objective is to predict the original inputs for each of the masked tokens using a bidirectional encoder of the kind described in the last section. The cross-entropy loss from these predictions drives the training process for all the parameters in the model. Note that all of the input tokens play a role in the selfattention process, but only the sampled tokens are used for learning.
More specifically, the original input sequence is first tokenized using a subword model. The sampled items which drive the learning process are chosen from among the set of tokenized inputs. Word embeddings for all of the tokens in the input are retrieved from the word embedding matrix and then combined with positional embeddings to form the input to the transformer.

![Masked language model training.](https://firebasestorage.googleapis.com/v0/b/birthday-react-6eca4.appspot.com/o/Natural-Language-Processing%2FWeek%201%20%3A%20Analyzing%20Language%2Fmml.jpg?alt=media&token=2f8207c5-5149-448a-846a-316543613c86)

Here, long, thanks and the have been sampled from the training sequence, with the first two masked and the
replaced with the randomly sampled token apricot. The resulting embeddings are passed through a stack of bidirectional transformer blocks. To produce a probability distribution over the vocabulary for each of the masked tokens, the output vector from the final transformer layer for each of the masked tokens is multiplied by a learned set of classification weights **W** and then through a **softmax** to yield the required predictions over the vocabulary. yi = softmax(WV hi)

With a predicted probability distribution for each masked item, we can use crossentropy to compute the loss for each masked item—the negative log probability assigned to the actual masked word.

## Next Sentence Prediction

The focus of masked-based learning is on predicting words from surrounding contexts with the goal of producing effective word-level representations. However, an important class of applications involves determining the relationship between pairs of sentences. These includes tasks like paraphrase detection (detecting if two sentences have similar meanings), entailment (detecting if the meanings of two sentences
entail or contradict each other) or discourse coherence (deciding if two neighboring sentences form a coherent discourse).

To capture the kind of knowledge required for applications such as these, **BERT** introduced a second learning objective called _Next Sentence Prediction_ (_NSP_). In this task, the model is presented with pairs of sentences and is asked to predict whether each pair consists of an actual pair of adjacent sentences from the training corpus or a pair of unrelated sentences. In **BERT**, 50% of the training pairs consisted of positive pairs, and in the other 50% the second sentence of a pair was randomly selected from elsewhere in the corpus. The NSP loss is based on how well the model can distinguish true pairs from random pairs.

To facilitate NSP training, BERT introduces two new tokens to the input representation (tokens that will prove useful for fine-tuning as well). After tokenizing the input with the subword model, the token **[CLS]** is prepended to the input sentence pair, and the token **[SEP]** is placed between the sentences and after the final token of the second sentence. Finally, embeddings representing the first and second segments of the input are added to the word and positional embeddings to allow the model to more easily distinguish the input sentences.

## Transfer Learning through Fine-Tuning

The power of pretrained language models lies in their ability to extract generalizations from large amounts of text—generalizations that are useful for myriad downstream applications. To make practical use of these generalizations, we need to create interfaces from these models to downstream applications through a process called **fine-tuning**. **Fine-tuning** facilitates the creation of applications on top of pretrained models through the addition of a small set of application-specific parameters. The fine-tuning _process consists of using **labeled data** from the application to train these additional application-specific parameters_. Typically, this training will either freeze or make only minimal adjustments to the pretrained language model parameters.

During fine-tuning, pairs of labeled sentences from the supervised training data are presented to the model. As with sequence classification, the output vector associated with the prepended **[CLS]** token represents the model’s view of the input pair. And as with NSP training, the two inputs are separated by the a **[SEP]** token. To perform classification, the **[CLS]** vector is multiplied by a set of learning classification weights and passed through a softmax to generate label predictions, which are then used to update the weights.

## Summary

• Bidirectional encoders can be used to generate contextualized representations of input embeddings using the entire input context.
• Pretrained language models based on bidirectional encoders can be learned using a masked language model objective where a model is trained to guess the missing information from an input.
• Pretrained language models can be fine-tuned for specific applications by adding lightweight classifier layers on top of the outputs of the pretrained model.

## References

- Daniel Jurafsky & James H. Martin. Speech and Language Processing January 12, 2022. <br/>
- Vrije Universiteit Amsterdam Natural Language Processing Technology Course Notes
