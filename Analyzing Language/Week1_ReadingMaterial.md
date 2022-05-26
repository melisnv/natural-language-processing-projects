`Semantic ambiguity :` _(Anlamsal belirsizlik)_ Dilsel bir ifade, en azından bağlam dışında söylendiğinde, birden çok duyuya sahip olduğunda anlamsal belirsizlik gösterir. Sözcük belirsizliği, sözcük veya biçimbirim düzeyinde ortaya çıkan anlamsal belirsizliğin alt türüdür. Anlamsal belirsizlik, belirli bir kelimenin iki farklı yorumu olduğu zamandır. İki farklı yorum, o tek kelimenin iki farklı yorumudur.

Consider the newspaper headline, **Gandhi stoned in rally in India**. There are two different interpretations of the verb stoned. One interpretation is people threw stones at Gandhi at the rally in India. Another interpretation is, Gandhi was using illicit drugs at the rally in India. So those are two different interpretations of the word stoned, and they give rise to two different interpretations of what someone is saying when they say, Gandhi stoned in rally in India.

`Lexical ambiguity` is the subtype of semantic ambiguity which occurs at the level of words or morphemes.

`Syntactic ambiguity : `_(Söz dizimsel belirsizlik)_ Yapısal belirsizlik, sözdizimsel belirsizlik, belirsiz cümle yapısı nedeniyle bir cümlenin birden fazla şekilde yorumlanabildiği bir durumdur.

**Police can’t stop gambling.** There are two different ways of understanding this sentence: Again, two different correct interpretations, two different possible interpretations. One is a saying that, there’s gambling taking place and the police just can’t stop it or the sentence can be understood as saying, the police themselves are gambling and they just can’t stop.Two different interpretations of the sentence police can’t stop gambling.

# Word Segmentation

`Tokenization` is splitting a sentence into its parts(tokens). `types` are different tokens.

_I have seen it time and time and time again._

This sentence has 11 tokens(with **.** included), 8 types(with **.** included, but **time** counted 1 and **and** counted 1).

## Word Classes

`Function word`, dilbilimde, işlev sözcükleri, sözcüksel anlamı çok az olan veya anlamı belirsiz olan ve bir cümle içindeki diğer sözcükler arasındaki dilbilgisel ilişkileri ifade eden ya da konuşmacının tutumunu ya da ruh halini belirten sözcüklerdir.
_**Function words** are words that have a grammatical purpose. Function words include pronouns, determiners, and conjunctions. These include words such as he, the, those, and the words and or but._

`Lemmatization`, is to lemmatize the words “_cats_,” “_cat's_”, and “_cats_'” means taking away the suffixes “s,” “'s,” and “s'” to bring out the root word “**cat**”.

A `morpheme` is the smallest meaningful lexical item in a language. The field of linguistic study dedicated to morphemes is called morphology.

"_Un-drink-able_"
Prefix morphome - root morphome - suffix morphome

`Subwords` solve the out of vocabulary problem, and help to reduce the number of model parameters to a large extent. _Subword_ is in between word and character. It is not too fine-grained while able to handle unseen word and rare word. For example, we can split “subword” to “sub” and “word”. In other word we use two vector (i.e. “sub” and “word”) to represent “subword”.

`N-grams : ` Hesaplamalı dilbilim ve olasılık alanlarında, n-gram, belirli bir metin veya konuşma örneğindeki n öğenin bitişik bir dizisidir. Öğeler uygulamaya göre fonemler, heceler, harfler, kelimeler veya baz çiftleri olabilir. N-gramlar tipik olarak bir metin veya konuşma topluluğundan toplanır.

`Part of speech : ` There are eight parts of speech in the English language: noun, pronoun, verb, adjective, adverb, preposition, conjunction, and interjection. The part of speech indicates how the word functions in meaning as well as grammatically within the sentence

# Named Entity Recognation

![NER](https://firebasestorage.googleapis.com/v0/b/birthday-react-6eca4.appspot.com/o/Natural-Language-Processing%2FWeek%201%20%3A%20Analyzing%20Language%2Fnet.jpg?alt=media&token=830cb265-77b2-42cf-a24d-c9811183de72)

# BIO-Tagging

The **IOB** format (short for inside, outside, beginning) is a common tagging format for tagging tokens in a chunking task in computational linguistics (ex. _named-entity recognition_). The **I-** prefix before a tag indicates that the tag is inside a chunk. An **O** tag indicates that a token belongs to no chunk. The **B-** prefix before a tag indicates that the tag is the beginning of a chunk that immediately follows another chunk without **O** tags between them. It is used only in that case: when a chunk comes after an **O** tag, the first token of the chunk takes the **I-** prefix.<br>

Alex `I-PER`<br>
is `O`<br>
going `O`<br>
to `O`<br>
Los `I-LOC`<br>
Angeles `I-LOC`<br>
in `O`<br>
California `I-LOC`<br>

The same example after filtering out stop words:

Alex `I-PER`<br>
going `O`<br>
Los `I-LOC`<br>
Angeles `I-LOC`<br>
California `B-LOC`<br>

Notice how "_California_" now has the "**B-**" prefix, because it immediately follows another `LOC` chunk.

# Parsing

![Parsing](https://firebasestorage.googleapis.com/v0/b/birthday-react-6eca4.appspot.com/o/Natural-Language-Processing%2FWeek%201%20%3A%20Analyzing%20Language%2Fparsing.jpg?alt=media&token=33587a30-37a6-4088-9d48-ebd011d9258a)

# The Bender Rule

Work on languages other than English is often considered “language specific” and thus reviewed as less important than equivalent work on English. Reviewers for NLP conferences often mistake the state of the art on a given task as the state of the art for English on that task, and if a paper doesn’t compare to that then they can’t tell whether it’s “worthy”.

`Do state the name of the language that is being studied, even if it's English. Acknowledging that we are working on a particular language foregrounds the possibility that the techniques may in fact be language specific. Conversely, neglecting to state that the particular data used were in, say, English, gives [a] false veneer of language-independence to the work.`

Eventually, the statement of the Bender Rule coalesced to “_always name the language you’re working on_”.

The second concern is that models trained on running text will pick up biases from that text, based on how the authors of the text view and talk about the world (e.g. Bolukbasi et al 2016[30], Speer 2017[26:1]).It is suggested that all NLP systems should be accompanied by detailed information about the training data, including the specific language varieties involved, the curation rationale (how was the data chosen and why?), demographic information about the speakers and annotators, and more(Bender & Friedman, 2018[31]). This information alone won’t solve the problems of bias, of course, but it opens up the possibility of addressing them.

# Reading Comprehension Questions

## Einstein's Introduction

- _How can Zipf’s law be summarized and why is it relevant for NLP?_
  According to Zipf's law, the frequency of a given word is dependent on the inverse of it's rank. The distribution over words resembles a **power law** _(Zipf's Law)_ and there will be few words that are very frequent and a long tail of words that are rare.

- _Why is the compositionality of language a problem for NLP?_
  Units such as words can combine by the very same principles to create larger phases.
  The meaning of a word is constructed from the constituent part -- the principle of **compositionality**. This principle can be applied to larger units: phrases, sentences and beyond. One of the great strengths of the compositional view of meaning is that it provides a roadmap for understanding entire texts and dialogues through a single analytic lens, grounding out in the smallest parts of individual words.

The principle of **compositionality** states that in a meaningful sentence, if the lexical parts are taken out of the sentence, what remains will be the rules of composition. Take, for example, the sentence _"Socrates was a man"_. Once the meaningful lexical items are taken away—"Socrates" and "man"—what is left is the pseudo-sentence, _"S was a M"_. The task becomes a matter of describing what the connection is between S and M.

-_Which ethical aspects need to be considered in NLP?_
Natural Language Processing raises some particularly salient issues around ethics, fairness and accountability:

`Access`<br>
Who is natural language processing designed to serve ? Whose language is translated _from_, and whose language is translated _to_. <br>

`Bias`<br>
Does language technology learn to replicate social biases from the text corpora, and does it reinforce these biases as seemingly objective computational conclusions?<br>

`Labor`<br>
Whose text and speech comprise the datasets that power natural language processing, and who performs the annotations? Are the benefits of this technology shared with all the people whose work makes it possible ?<br>

`Privacy and internet freedom`<br>
What is the impact of large-scale text processing on the right to free and private communication? What is the potential role of nlp in regimes of censorship or surveillance?<br>

-_How can the distributional perspective on language be summarized?_
The **distributional hypothesis** suggests that the more semantically similar two words are, the more distributionally similar they will be in turn, and thus the more that they will tend to occur in similar linguistic contexts.

# Essentials of Linguistics

## Words and Morphemes

A _word_ is a free form that has a meaning. There are other forms that have meaning and some of them seem to be smaller than whole words. A _morpheme_ is the smallest form that has meaning. Some morphemes are free: they can appear in isolation.<br>

_walked_, _baked_, _cleaned_, _kicked_, _kissed_.<br>

This **–ed** unit appears consistently in this form and consistently has this meaning, but it never appears in isolation: it’s always attached at the end of a word. It’s a `bound morpheme`.<br>

Many words are made up of meaningful pieces called morphemes. In English, the most common bound morphemes are suffixes and prefixes, which can be affixed to words to derive new words, or can convey grammatical information via inflection.

## Combining Words

`Function Words : ` There are also several smaller categories of words called closed-class categories because the language does not usually add new words to these categories. They’re the **function words** or **non-lexical categories** that do a lot of grammatical work in a sentence but don’t necessarily have obvious semantic content.

# Speech and Language Processing

## Tokenization and Lemmatization (chapter 2.2 - 2.4)

Normalizing text means converting it to a more convenient, standard form. For example, most of what we are going to do with language relies on first separating out or **tokenizing** words from running text, the task of tokenization.

Another part of text normalization is **lemmatization**, the task of determining that two words have the same root, despite their surface differences. For example, the words sang, sung, and sings are forms of the verb sing. The word sing is the common lemma of these words, and a lemmatizer maps from all of these to sing. _Lemmatization_ is essential for processing morphologically complex languages

**Stemming** refers to a simpler version of lemmatization in which we mainly just strip suffixes from the end of the word. Text normalization also includes sentence segmentation: breaking up a text into individual sentences, using cues like segmentation periods or exclamation points.

# Words (chapter 2.2)

_He stepped out into the hall, was delighted to encounter a water brother._<br>
This sentence has 13 words if we don’t count punctuation marks as words, 15 if we count punctuation. Whether we treat period (“.”), comma (“,”), and so on as words depends on the task. Punctuation is critical for finding boundaries of things (commas, periods, colons) and for identifying some aspects of meaning (question marks, exclamation marks, quotation marks). For some tasks, like part-of-speech tagging or parsing or speech synthesis, we sometimes treat punctuation marks as if they were separate words.<br>

Are capitalized tokens like `They` and uncapitalized tokens like `they` the same word? These are lumped together in some tasks (_speech recognition_), while for **part of-speech** or **named-entity tagging**, capitalization is a useful feature and is retained.<br>

A `lemma` is a set of lexical forms having the same stem, the same major part-of-speech, and the same word sense. The word form is the full inflected or derived form of the word.<br>

**Types** are the number of distinct words in a corpus; if the set of words in the vocabulary is V, the number of types is the vocabulary size `|V|`. **Tokens** are the total number N of running words. If we ignore punctuation, the following Brown sentence has 16 tokens and 14 types:<br>

_They picnicked by the pool, then lay back on the grass and looked at the stars._<br>

The larger the corpora we look at, the more word types
we find, and in fact this relationship between the number of types `|V|` and number of tokens N is called **Herdan’s Law** or **Heaps' Law**.

# Text Normalization (chapter 2.4)

Before almost any natural language processing of a text, the text has to be normalized. At least three tasks are commonly applied as part of any normalization process:<br>

1. Tokenizing (segmenting) words<br>
2. Normalizing word formats<br>
3. Segmenting sentences<br>

## 2.4.2 Word Tokenization

While the Unix command sequence just removed all the numbers and punctuation, for most NLP applications we’ll need to keep these in our **tokenization**. We often want to break off _punctuation_ as a separate token; commas are a useful piece of information for parsers, periods help indicate sentence boundaries. But we’ll often want to keep the punctuation that occurs word internally, in examples like _m.p.h._, _Ph.D._, _AT&T_, and _cap’n_. Special characters and numbers will need to be kept in prices _($45.55)_ and dates _(01/02/06)_; we don’t want to segment that price into separate tokens of “45” and “55”. And there are URLs (*http://www.stanford.edu*), Twitter hashtags (_#nlproc_), or email addresses (*someone@cs.colorado.edu*).<br>

Depending on the application, tokenization algorithms may also tokenize multiword expressions like _New York_ or _rock 'n' roll_ as a single token, which requires a multiword expression dictionary of some sort. `Tokenization` is thus intimately tied up with `named entity recognition`, the task of detecting names, dates, and organizations

## 2.4.3 Byte-Pair Encoding for Tokenization

There is a third option to tokenizing text. Instead of defining tokens as words (whether delimited by spaces or more complex algorithms), or as characters (as in Chinese), we can use our data to automatically tell us what the tokens should be. This is especially useful in dealing with unknown words, an important problem in language processing. As we will see in the next chapter, NLP algorithms often learn some facts about language from one corpus (a training corpus) and then use these facts to make decisions about a separate test corpus and its language. Thus if our training corpus contains, say the words low, new, newer, but not lower, then if the word lower appears in our test corpus, our system will not know what to do with it.

To deal with this unknown word problem, modern tokenizers often automatically induce sets of tokens that include tokens smaller than words, called subwords. **Subwords** can be arbitrary substrings, or they can be meaning-bearing units like the _morphemes_ **-est** or **-er**. (_A morpheme is the smallest meaning-bearing unit of a language; for example the word unlikeliest has the morphemes un-, likely, and -est._) In modern tokenization schemes, most tokens are words, but some tokens are frequently occurring morphemes or other subwords like -er. Every unseen word like lower can thus be represented by some sequence of known subword units, such as _low_ and _er_, or even as a sequence of individual letters if necessary.

Most tokenization schemes have two parts: a **token learner**, and a **token segmenter**. The token learner takes a raw training corpus (sometimes roughly preseparated into words, for example by whitespace) and induces a vocabulary, a set of tokens. The token segmenter takes a raw test sentence and segments it into the tokens in the vocabulary. Three algorithms are widely used: `byte-pair encoding`, `unigram language modeling`, and `WordPiece`.

The **BPE** token learner begins with a vocabulary that is just the set of all individual characters. It then examines the training corpus, chooses the two symbols that are most frequently adjacent (say ‘A’, ‘B’), adds a new merged symbol ‘AB’ to the vocabulary, and replaces every adjacent ’A’ ’B’ in the corpus with the new ‘AB’. It continues to count and merge, creating new longer and longer character strings, until k merges have been done creating k novel tokens; k is thus a parameter of the algorithm. The resulting vocabulary consists of the original set of characters plus k new symbols. The algorithm is usually run inside words (not merging across word boundaries), so the input corpus is first white-space-separated to give a set of strings, each corresponding to the characters of a word, plus a special end-of-word symbol , and its counts. Let’s see its operation on the following tiny input corpus of 18 word tokens with counts for each word (the word low appears 5 times, the word newer 6 times, and so on), which would have a starting vocabulary of 11 letters:

# 2.4.4 Word Normalization, Lemmatization and Stemming

Word normalization is the task of putting words/tokens in a standard format, choosing a single normal form for words with multiple forms like USA and US or uh-huh and uhhuh. This standardization may be valuable, despite the spelling information that is lost in the normalization process. For information retrieval or information extraction about the US, we might want to see information from documents whether they mention the US or the USA.

**Lemmatization** is the task of determining that two words have the same root, despite their surface differences. The words _am_, _are_, and _is_ have the shared lemma be; the words dinner and dinners both have the lemma dinner. Lemmatizing each of these forms to the same lemma will let us find all mentions of words in Russian like Moscow. The lemmatized form of a sentence like _He is reading detective stories_ would thus be _He be read detective story_.

# Sequence Labeling for Parts of Speech and Named Entities (chapter 8)

`Parts of speech` (also known as _POS_) and named entities are useful clues to sentence structure and meaning. Knowing whether a word is a noun or a verb tells us about likely neighboring words and syntactic structure , making part-of-speech tagging a key aspect of parsing. Knowing if a named entity like Washington is a name of a person, a place, or a university is important to many natural language processing tasks like question answering, stance detection, or information extraction.

The task of part-of-speech tagging, taking a sequence
of words and assigning each word a part of speech like NOUN or VERB, and the task of `named entity recognition` _(NER)_, assigning words or phrases tags like _PERSON_, _LOCATION_, or _ORGANIZATION_.

![POS](https://firebasestorage.googleapis.com/v0/b/birthday-react-6eca4.appspot.com/o/Natural-Language-Processing%2FWeek%201%20%3A%20Analyzing%20Language%2Fpos.jpg?alt=media&token=dba04fc9-0c44-445c-889f-22f1aa7c1f13)

Tagging is a disambiguation task; words are **ambiguous** and they have more than one possible part-of-speech—and the goal is to find the correct tag for the situation. For example, book can be a **verb** (_`book` that flight_) or a **noun** (_hand me that `book`_).

This idea suggests a useful _baseline_: given an ambiguous word, choose the tag which is most **frequent** in the training corpus. `Most Frequent Class Baseline`: Always compare a classifier against a baseline at least as good as the most frequent class baseline (assigning each token to the class it occurred in most often in the training set).

Part of speech tagging can tell us that words like Janet, Stanford University, and Colorado are all proper nouns; being a proper noun is a grammatical property of these words. But viewed from a semantic perspective, these proper nouns refer to different kinds of entities: Janet is a **person**, Stanford University is an **organization**, and Colorado is a **location**.

A named entity is, roughly speaking, anything that can be referred to with a proper name: a person, a location, an organization. The task of named entity recognition (NER) is to find spans of text that constitute proper names and tag the type of recognition (NER) the entity. Four entity tags are most common: PER (person), LOC (location), ORG
(organization), or GPE (geo-political entity). However, the term named entity is commonly extended to include things that aren’t entities per se, including dates, times, and other kinds of temporal expressions, and even numerical expressions like prices.

Named entity tagging is a useful first step in lots of natural language processing tasks. In sentiment analysis we might want to know a consumer’s sentiment toward a
particular entity. Entities are a useful first stage in question answering, or for linking text to information in structured knowledge sources like Wikipedia. And named
entity tagging is also central to tasks involving building semantic representations, like extracting events and the relationship between participants. Unlike part-of-speech tagging, where there is no segmentation problem since each word gets one tag, the task of named entity recognition is to find and label spans of text, and is difficult partly because of the ambiguity of segmentation.

The standard approach to sequence labeling for a span-recognition problem like NER is `BIO tagging` (_Ramshaw and Marcus, 1995_). This is a method that allows us to treat NER like a _word-by-word sequence labeling task_, via tags that capture both the boundary and the named entity type. Figure 8.7 shows the same excerpt represented with BIO tagging, as well as variants called IO tagging and BIOES tagging. In `BIO tagging` we label any token that _begins a span of interest_ with the label **B**, tokens that _occur inside a span_ are tagged with an **I**, and any tokens _outside of any span of interest_ are labeled **O**. While there is only one O tag, we’ll have distinct B and I tags for each named entity class.

![BIO-Tagging](https://firebasestorage.googleapis.com/v0/b/birthday-react-6eca4.appspot.com/o/Natural-Language-Processing%2FWeek%201%20%3A%20Analyzing%20Language%2Fbio.jpg?alt=media&token=61a60318-75ef-4451-951e-b12709813465)
