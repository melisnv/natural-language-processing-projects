![]()

![]()

# Hate Speech: The Phenomenon

`Hate speech` (HS)is communication that disparages a person or a group on the basis of some characteristic such as race, color, ethnicity, gender, sexual orientation, nationality, religion, or other characteristics. Related concepts are toxicity, cyberbullying, offensive language, online
harassment, discriminatory language, abusive language, misogyny Growing phenomenon is pervasive on social media platforms and more niche platforms (4chan and 8chan) that are used to spread HS, aggression, etc. (partly) due to possibility to post content anonymously.<br>

**Definition**: disputed, subjective and multi-interpretable concept <br>
• Prone to personal bias and culture-dependent (HS for some is not HS
for others)<br>
• **Racial bias:** when racial background of the author is added, annotators
are significantly less likely to label an African American English tweet as
offensive; text alone does not determine offensiveness: community
context is very important.<br>

Complex phenomenon: lack of a consensus on the definition and common annotation guidelines →low/moderate inter-annotator agreement →scarcity of high-quality training data<br>
• **Recent datasets:** offensive vocabulary and keywords evolve fast (neologisms); users adapt lexical choices or introduce minor<br>
**misspellings:** countermeasure against identification.(Nefret dili detect edilmesin diye yapılıyor yanlış yazım hatası etc.)<br>
• Limitations on datasets distribution:<br>
• Sharing data is an important aspect of open science but also poses ethical and legal risks<br>
• Concern that whilst it aimed to support academic research, it may enabled individuals to radicalise by making otherwise banned, e.g.,extremist material, available<br>
• The problem of data access and sharing remains unresolved in the field of hate speech detection<br>

### Data Availability Limitations: Source of Data

The `lack of diversity` in where data is collected is a serious limitation:<br>
• Linguistic practices vary across platforms: good for Twitter could be worse for long texts<br>
• The demographics of users on different platforms vary considerably. Twitter users are usually younger and wealthier than offline populations<br>
• Platforms have different norms and so host different types and amounts of abuse: aggressive forms of abuse, such as direct threats,are likely to be taken down, while more niche platforms are more likely to contain explicit abuse.<br>

### Data Availability Limitations: Size of Datasets

The size of the training datasets varies considerably from 469 posts to 17 million._Differences in size partly reflect different annotation approaches._<br>
• Smaller datasets are problematic because they contain too little linguistic variation and increase the likelihood of overfitting<br>
• Large training datasets which have been poorly sampled, annotated with theoretically problematic categories or inexpertly and unthoughtfully annotated, could still lead to the development of poor classification systems<br>
Imbalanced class distribution is common HS datasets (< 10% HS).<br>

### Data Collection and Annotation Strategies (Example)

LiLaH (The Linguistic Landscape of Hate Speech in Social Media) dataset contains Facebook comments to online news articles on LGBT and migrant topics in Dutch. It's collected using keywords (e.g., gay, biseksueel / immigrant, islam, moslim). They trained the annotators. Detailed annotation guidelines are: <br>
• **type**: violence, threat, offensive language, inappropriate speech<br>
**target**: against whom hate speech is directed
• fine-grained categories<br>
• _the context and the author’s intent should be considered_<br>

Then Demographics Analysis of Hateful Content Creators are applied. Bu nefret dilini kullananlar ile ilgili bir istatistik çalışması yapıldı ve yaşı büyük erkeklerin daha çok nefret dili kullandığı, kadınların 26 yaşından sonra oranının azaldığı görüldü.<br>

`OLID dataset` (Zampieri et al., 2019) (≈14,000 tweets)
• Examples collected from Twitter using keywords: ‘you are’, ‘she is’, ‘gun control’, ‘antifa’, ‘liberals’, ...<br>
• Annotation using `crowdsourcing` (≈3 annotators; majority vote)<br>
• Fine-grained three-layer annotation scheme:<br>
• Whether the language is offensive or not<br>
• Whether offensive language is targeted or untargeted<br>
• Whether offensive language is targeted towards individual, group, or other (organization, situation, event, issue)<br>

### Annotation Layers (Human Rationales)

HateXplain dataset (Mathew et al., 2021): word and phrase level span annotations that capture human rationales for the labeling <br>
Data collected from Twitter and Gab (≈20,000 posts)<br>
Annotation using `crowdsourcing` (_3 annotators; majority vote_) from three different
perspectives:
• **Type**: hate, offensive, normal
• **Target**: based on race, religion, gender, sexual orientation: African, Islam, Jewish, LGBTQ, women, refugee, Arab, Caucasian, Hispanic, Asian (10 target groups)
• `Rationales`: portions of the post on which the labelling decision is based _(human-level explanations)_

![Hate Speech](https://firebasestorage.googleapis.com/v0/b/birthday-react-6eca4.appspot.com/o/Natural-Language-Processing%2Fhate.jpg?alt=media&token=82ce9609-5008-4442-b7c3-9010ad934681)

`Counterspeech` is a tactic of countering hate speech or misinformation by presenting an alternative narrative rather than with censorship of the offending speech. It also means responding to hate speech with empathy and challenging the hate narratives, rather than responding with more hate speech directed in the opposite direction. According to advocates, counterspeech is more likely to result in _deradicalization_ and peaceful resolution of conflict. `Counterspeech` is the effective strategies in order to limit the hate content online.

## Automatic Hate Speech Detection

Discussion of automatic hate speech detection goes back approx. two decades (Spertus, 1997). Recently renewed interest (hot topic). Performance increased significantly by the large amount of available social media data and with Deep learning approaches.

#### Hate Speech (Type) Detection as a Classification Problem

`Binary classification:` the task of classifying textual content into hate or non-hate speech classes (the most common framing of the task)<br>
`Multi-class classification:` each sample is labeled with exactly one class out of multiple classes, e.g., fine-grained types (e.g., threat,offense, violence) or the severity of hate speech<br>
`Multi-label classification:` comments fulfilling different predefined criteria at the same time (e.g., toxic, severe toxic, insult, threat, obscene, identity hate)<br>

#### Shared Tasks

• _Toxic comment classification challenge (2017)_ (Wikipedia comments, English): different types of of toxicity: toxic, severe toxic, insult, threat, obscene, identity hate
• _HatEval (2019) (Twitter, English and Spanish):_ hate speech against immigrants and women (hate speech or not), aggressive behavior (aggressive or not), target (individual, group)
• _OffensEval (2019, 2020) (Twitter, multiple languages):_ type (offensive or not) and target (targeted or untargeted; what is the target: individual, group, other) of offensive content (2020: 145 teams, all-time record)

## Evaluation Metrics

• `Precision`: of all messages predicted as hateful, how many are actually hateful<br>
• `Recall`: how many of actually hateful messages our model predicted as hateful<br>
• `F1-score`: harmonic mean of precision and recall;<br>
**macro-averaged**: arithmetic mean of all the per-class F1-scores <br>

Precision, recall and F1-score for each class is used by some scientist to show how the performance is.<br>
Area under the curve (`AUC`): way to summarize a precision-recall curve<br>

**Accuracy** is not used in evaluation metrics of Hate Speech Detection because the datasets are usally very imbalanced in terms of classes and the uaccuracy is used when there is even distribution.
<br>

## Approaches

• `Lexicon-based approaches` rely on using external resources, e.g.,
offensive word lists: ≈70%–75% F1-score*<br>
• `Conventional machine learning approaches:` SVM with word, character n-grams, other: ≈70%–78%<br>
• `Deep neural networks:` CNN, LSTM, BiLSTM: ≈75%–79%;<br>
• `Transformer-based approaches:` BERT, RoBERTa >80%; ≈82% <br>
• Ensemble learning: ≈85%<br>
Each approach has advantages and disadvantages<br>
• Performance is depends on the language, training data, how the task is framed <br>
• cross-domain drop is about 10%<br>
• high results considering the inter-annotator agreement<br>
*Results for the binary hate speech detection task (hate speech VS. non-hate speech)<br>

## Lexicon-Based Approaches

Various ways of exploiting lexicon information. Often used as a baseline. E.g., given a reference dictionary/lexicon of profanities, abusive terms, slurs, etc., if a message contains one or more of the terms in the dictionary, then it is labeled as hateful (Caselli et al., 2021). ‘+’ * Variety of available lexical resources (e.g, dictionaries, lexicons of offensive words) in different languages; multilingual resources:<br>
• HurtLex (Bassignana et al., 2018): multilingual lexicon of hateful words in 53 languages (1,157 unique lemmas)<br>
• POW (De Smedt et al., 2020): multilingual fine-grained lexicons for hate speech in 5 languages (10,000+ expressions for Dutch)<br>
‘+’ Explainability (used by many organizations)<br>
‘–’ Implicit hate speech: hate speech that is not explicitly expressed by means of profanities, slurs or insults<br>
‘–’ Offensive words in a non-hateful content<br>
*Only some of the advantages and disadvantages for each approach are presented<br>

## Conventional Machine Learning Approaches

Stylometric and emotion-based approach (Markov et al., 2021)<br>
NRC emotion lexicon (Mohammad and Turney, 2013): 14,182 emotion words and their associations with eight emotions and two sentiments.<br>
E.g., illness<br>
![](https://firebasestorage.googleapis.com/v0/b/birthday-react-6eca4.appspot.com/o/Natural-Language-Processing%2Fillnes.jpg?alt=media&token=8406fa12-8499-42ee-ba15-2938f0f853b2)

This is a stylometric and emotion-based approach. Features are `Part-of-speech (POS)` tags such as morpho-syntactic patterns, `Function words (FWs)` like very informative stylometric patterns, and `Emotion-based features` like emotion-conveying words, their frequency, and emotion associations from the NRC emotion lexicon.
![](https://firebasestorage.googleapis.com/v0/b/birthday-react-6eca4.appspot.com/o/Natural-Language-Processing%2Fmental.jpg?alt=media&token=19bc2ed1-477a-4b17-994e-4e7160da73a2)

• Hate speech lexicon features<br>
• Word and character n-grams<br>

- SVM + tf-idf weightning scheme<br>

They realized that the function words are very relevant in the hate speech. Therefore `function words` do contribute hate speech detection.

## Deep Learning Approaches

Generic pre-trained language models: `BERT`, RoBERTa, XLNet, ALBERT etc. Abuse-inclined HateBERT (Caselli et al., 2020): a re-trained BERT model for abusive language detection; re-trained on Reddit
comments from communities banned for being offensive, abusive, or hateful<br>
‘+’ State-of-the-art performance (Facebook uses RoBERTa)<br>
‘+’ No need for feature engineering<br>
‘-’ Data hungry<br>
‘-’ Explainability<br>

_Facebook uses RoBERTa_.<br>

## Ensemble Learning

• Ensemble methods: select the optimal classifier for a given instance<br>
• Advantages of different models: ensembling predictions by deep learning models or deep learning and machine learning<br>
• Most of the top-ranked teams in the shared task used pre-trained language models as part of ensembles<br>

`The main idea is that we want to select the best classifier for a particular message.`

## Ensemble Methods: Majority-Voting

![Voting](https://firebasestorage.googleapis.com/v0/b/birthday-react-6eca4.appspot.com/o/Natural-Language-Processing%2Fvoting.jpg?alt=media&token=9c01e015-e1ff-4229-be58-1a75155762bd)

## Encoding Conversational Context

Important for hate speech detection:<br> `“Go back home”` <br>
Previous studies considered the text of the post / previous _(‘parent’)_ comment in the discussion thread as context.<br>
**Çıkarım :** Context significantly affects annotation process; no evidence that
leads to a large or consistent improvement in model performance.

## Encoding Conversational Context

Focus on detecting the target of hates speech (migrants or not). Manually annotate relevant conversational context in Facebook discussion threads:<br>
• If context influences the annotators’ decision to assign a label to a comment or the target of hate speech is not sufficiently clear without contextual information →indicate ID of comment/post
that serves as context<br>
• Concatenate using `[SEP]` token before fine-tuning BERTje<br>

#### Implicit contextual Information

• _Previous_: explicit context (contextual information explicitly appears in the discussion thread)<br>
• _Implicit_: relies on world knowledge to understand the meaning<br>
_Example_: _“Welcoming migrants is like **going to the dentist**.”_ _(going to dentist is a hidden hate speech, not a direct one. therefore you need to know the context and understand the meaning)_<br>
• Different annotation strategies; retrieving external knowledge<br>

## Toxic Spans Detection Approaches

Toxic Spans Detection shared task (2021)<br>
Detection of the spans that make a post toxic<br>
`Spans`: sequence of words that attribute to the post’s toxicity<br>
• Word-level _BIO tags_<br>
• `Model-specific` or model-agnostic rationale extraction mechanisms to produce toxic spans as explanations of the decisions of the classifier<br>
• `Lexicon-based`: list of toxic words for lookup operations<br>

## Common Errors in Automatic Hate Speech Detections

Error classes of false negatives (hateful messages that were missed by the algorithm):<br>
• Doubtful labels<br>
• Toxicity without swear words _(e.g., “she looks like a horse”)_<br>
• Rhetorical questions _(e.g., “have you no brain?!?!”)_<br>
• Metaphors and comparisons _(e.g., “Who are you a sockpuppet for?”)_<br>
• Sarcasm and irony _(e.g., “hope you’re proud of yourself. Another milestone in idiocy”)_<br>
• Idiosyncratic and rare words: _misspellings_, _neologisms_, obfuscations, _abbreviations_ and _slang_ words<br>

## Counter Narratives

• Currently: moderation activities, e.g., content removal, account suspension, or shadow-banning
Risk: hinder the freedom of expression<br>
• Alternative: counter narratives (CNs), i.e., non-aggressive response using credible evidence, factual arguments, alternative viewpoints. Effective strategy to combat online hate speech.Example of `HS/CN` pair:<br>
• `HS` _(hate speech)_: Women are basically childlike, they remain this way most of their lives.
Soft and emotional. It has devastated our once great patriarchal
civilizations.<br>
• `CN`_(counter narratives)_: Without softness and emotions there would be just brutality and
cruelty. Not all women are soft and emotional and many men have these
characteristics. To perpetuate these socially constructed gender profiles
maintains norms which oppress anybody.<br>

Counter Narratives is basically hate speech in covered in order to not removed. **Counter-Narrative Data Collection** is done with <br>
• `Crawling`: automatically scraping web-sites, starting from an HS content and searching for possible CNs among the responses.<br>
• `Crowdsourcing`: CNs are written by _non-expert_ paid workers as responses to provided hate content<br>
• `Nichesourcing`: relies on a niche _group of experts_ for data collection, i.e., NGO’s operators<br>
• `Hybrid approaches`: combination of language models for generation and humans for post-editing<br>
