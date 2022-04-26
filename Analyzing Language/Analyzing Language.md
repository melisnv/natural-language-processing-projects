# Week 1 : Analyzing Language

Natural language processing is the set of methods for making human language accessible to computers. Natural language processing is focused on the design and analysis of computational algorithms and representations for processing natural human language. The goal of natural language processing is to provide new computational capabilities around human language: for example, extracting information from texts, translating between languages, answering questions, holding a conversation, taking instructions, and so on.

The study of computer systems is also relevant to natural language processing. Large datasets of unlabeled text can be processed more quickly by parallelization techniques like MapReduce (Dean and Ghemawat, 2008; Lin and Dyer, 2010); high-volume data sources such as social media can be summarized efficiently by approximate streaming and sketching techniques (Goyal et al., 2009). When deep neural networks are implemented in production systems, it is possible to eke out speed gains using techniques such as reduced-precision arithmetic (Wu et al., 2016). Many classical natural language processing algorithms are not naturally suited to graphics processing unit (GPU) parallelization, suggesting directions for further research at the intersection of natural language processing and computing hardware.

Many natural language processing problems can be written mathematically in the form of optimization,<br/>
![alt text](https://firebasestorage.googleapis.com/v0/b/birthday-react-6eca4.appspot.com/o/Natural-Language-Processing%2FWeek%201%20%3A%20Analyzing%20Language%2Fformula.jpg?alt=media&token=825a75bb-b1b3-44d8-bf51-2cb832e269df)

where,

- x is the input, which is an element of a set X;
- y is the output, which is an element of a set Y(x);
- &#968; is a scoring function (also called the model), which maps from the set X Y to the real numbers;
- &theta; is a vector of parameters for ;
- &upsilon; is the predicted output, which is chosen to maximize the scoring function.

This basic structure can be applied to a huge range of problems. For example, the input x might be a social media post, and the output y might be a labeling of the emotional sentiment expressed by the author, or x might be a news article and y might be a structured record of the events that the article describes.
The <b>search</b> module is responsible for computing the argmax of the function &#968;. In other words, it finds the output &upsilon; that gets the best score with respect to the input x. This is easy when the search space &upsilon;(x) is small enough to enumerate, or when the scoring function has a convenient decomposition into parts. The <b>learning module</b> is responsible for finding the parameters &theta;. This is typically (but not always) done by processing a large dataset of labeled examples.

The division of natural language processing into separate modules for search and learning makes it possible to reuse generic algorithms across many tasks and models. Much of the work of natural language processing can be focused on the design of the model &#968; identifying and formalizing the linguistic phenomena that are relevant to the task at hand — while reaping the benefits of decades of progress in search, optimization, and learning.

### N-grams

N-grams are sequences of n tokens. N-grams are extracted iteratively from a sequence. N-gram statistics are more informative because they provide information about common phrases in the data and capture compounds (e.g. New York State).
We can extract n-grams over word classes to abstract from lexical aspects :

### How can we find patterns in language?

Calculating frequency effects:<br/>
○ Zipf’s law, types, tokens, lemmas, morphemes, subwords, word classes<br/>
Analyzing sequences:<br/>
○ n-grams, POS tags, named entities<br/>
Capturing meaning<br/>
Analyzing hierarchical structure

### Linguistic Pre-processing

Linguistic analysis steps which used to be huge research challenges can now
be handled by a single Python library (spacy, nltk, stanza) and are
considered to be pre-processing for more challenging tasks.

#### Linguistic Pipelines

Analysis steps are often performed as a consecutive pipeline.<br/>
○ For named entity recognition, we rely on good POS-Tags.<br/>
○ For semantic role labeling, we rely on a good syntactic analysis.

#### Error Propagation

These dependencies can lead to error propagation. An error in an earlier processing level leads to errors later in the pipeline.

### Dataset Analysis

“Once you get a qualitative sense it is also a good idea to write some simple code to search/filter/sort by whatever you can think of (e.g. type of label, size of annotations, number of annotations, etc.) and visualize their distributions and the outliers along any axis. The outliers especially almost always uncover some bugs in data quality or preprocessing.”

#### What are annotations?

Metadata providing linguistic information about the content. The labels are annotated by linguistic experts. Annotated corpora are very important resources for natural language processing. They are often not available for languages that are of less economic interest.

## References

- Jacob Eisenstein (2018) - Natural Language Processing Notes (https://github.com/jacobeisenstein/gt-nlp-class/blob/master/notes/eisenstein-nlp-notes.pdf) <br/>
- Vrije Universiteit Amsterdam NLP Lecture Notes
