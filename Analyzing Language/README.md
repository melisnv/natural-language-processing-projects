# Week 1 : Analyzing Language

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
