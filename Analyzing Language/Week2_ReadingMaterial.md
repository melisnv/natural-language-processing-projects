### Precision

![Precision](https://firebasestorage.googleapis.com/v0/b/birthday-react-6eca4.appspot.com/o/Natural-Language-Processing%2FWeek%202%20%3A%20Classifying%20Language%2Fprecision.jpg?alt=media&token=275c11d5-c81f-410f-944d-cf1a6ff60fa3)

True Positive divided by all the tweets that labeled as ironic by the model (green). _If we say a tweet is ironic, how often we are right._

### Recall

![Recall](https://firebasestorage.googleapis.com/v0/b/birthday-react-6eca4.appspot.com/o/Natural-Language-Processing%2FWeek%202%20%3A%20Classifying%20Language%2Frecall.jpg?alt=media&token=e765a3e7-af72-4f64-baea-2f942580fc8b)

Not miss a tweet that actually ironic.<br>

If we tune for high precision then _system should not make a mistake_. If we tune for high recall then _system should not miss a case_.

### F1 Score

The harmonic mean - a good trade-off between precision and recall-.

![F1 Score](https://firebasestorage.googleapis.com/v0/b/birthday-react-6eca4.appspot.com/o/Natural-Language-Processing%2FWeek%202%20%3A%20Classifying%20Language%2Ff1.jpg?alt=media&token=37d4e8d6-0c2d-4bb2-8a20-e2c6f2b562a5)

### Evaluation for Multiple Classes

![Multiple Classes](https://firebasestorage.googleapis.com/v0/b/birthday-react-6eca4.appspot.com/o/Natural-Language-Processing%2FWeek%202%20%3A%20Classifying%20Language%2Fmultipleclass.jpg?alt=media&token=914df12f-7b3d-4127-a26f-63182d1021cd)

![Mocaro Average and Weighted Average](https://firebasestorage.googleapis.com/v0/b/birthday-react-6eca4.appspot.com/o/Natural-Language-Processing%2FWeek%202%20%3A%20Classifying%20Language%2Fmacro.jpg?alt=media&token=523d63a8-cffa-4eb9-a661-e4118a1e4706)

## Macro Average vs Weighted Average

`Macro Average` says the function to compute **f1** for each label, and returns the average _without considering the proportion for each label in the dataset_. `Weighted Average` says the function to compute **f1** for each label, and returns the average _considering the proportion for each label in the dataset_.

## Feed-Forward Networks

We call softmax function because Neural Networks do not output a probability distribution but a activation for each label.

![Neural Networks Output](https://firebasestorage.googleapis.com/v0/b/birthday-react-6eca4.appspot.com/o/Natural-Language-Processing%2FWeek%202%20%3A%20Classifying%20Language%2Fnnoutput.jpg?alt=media&token=8ce001d7-4ba7-4e85-94fc-298fb6aa499e)

In the **hidden states**, the network learns abstractions over the input to extract the relevant information for predicting the output. The arrows symbolize weights to determine the importance of each input element. _Non-linear activation functions_ are essential (sigmoid,tanh etc.). _Fully connected Neural Network_ is when all inputscontribute to all hidden units. Dimension of output are determined by dataset, dimension of input and hidden state are experimental choices. _More hidden layers means more weigth matrices_. For good weights, start with random initialization then calculate the predicted output with the current weights, later compare the output with the expected output and calculate the error. Then for adjusting the weights, adjust them according to the size of the error,
their contribution to the error and the learning rate. (_Back-propogation_) Each try is one _epoch_. To avoid **overfitting**,apply regularization and early stopping.

`Recurrence` is a case where prediction of the next token depends on the prediction of the previous token.

![Recurrence Neural Networks(RNNs)](https://firebasestorage.googleapis.com/v0/b/birthday-react-6eca4.appspot.com/o/Natural-Language-Processing%2FWeek%202%20%3A%20Classifying%20Language%2Frecurrence.jpg?alt=media&token=812b954a-65ba-49e5-a35b-849dbb358785)

## Calculating the Baselines with the Given Table

![Random Baseline Calculation](https://firebasestorage.googleapis.com/v0/b/birthday-react-6eca4.appspot.com/o/Natural-Language-Processing%2FWeek%202%20%3A%20Classifying%20Language%2FEkran%20g%C3%B6r%C3%BCnt%C3%BCs%C3%BC%202022-05-26%20111506.jpg?alt=media&token=52252190-64de-40b3-aacb-01808c4891e3)<br>

`Precision` for _Ironic_ : 5 / (45 + 5) = 0.1 <br>
When calculating the **precision** you need to take the column since it's for models prediction.<br>

`Recall` for _Ironic_ : 5 / (5 + 5) = 0.5 <br>
When calculating the **recall** you need to take the row since it's for the actual label control.<br>

`F1 = 2 x ( Precision x Recall ) / ( Precision + Recall)`
`F1 Score` for _Ironic_ : F1 = 2*(0.1*0.5)/(0.1+0.5) = 0.17

`Precision` for _Not Ironic_ : 45 / (5 + 45) = 0.9 <br>
`Recall` for _Not Ironic_ : 45 / (45 + 45) = 0.5 <br>
`F1 Score` for _Not Ironic_ : F1 = 2*(0.9*0.5)/(0.9+0.5) = 0.64<br>

![Majority Baseline Calculation](https://firebasestorage.googleapis.com/v0/b/birthday-react-6eca4.appspot.com/o/Natural-Language-Processing%2FWeek%202%20%3A%20Classifying%20Language%2Fmajority.jpg?alt=media&token=fad4ed14-d25a-45fa-94c9-6909b9b5bc03)<br>

`Precision` for _Ironic_ : 0 / (0 + 0) = NaN <br>
`Recall` for _Ironic_ : 0 / (0 + 10) = 0 <br>
`F1 Score` for _Ironic_ : F1 = 2*(0*0)/(0+0) = 0<br>

`Precision` for _Not Ironic_ : 90 / (10 + 90) = 0.9 <br>
`Recall` for _Not Ironic_ : 90 / (90 + 0) = 1 <br>
`F1 Score` for _Not Ironic_ : F1 = 2*(1*0.9)/(1+0.9) = 0.95<br>

# Naive Bayes and Sentiment Classification (chapter 4)

In this chapter we introduce the naive Bayes algorithm and apply it to `text categorization`, the task of assigning a label or category to an entire text or document. We focus on one common text categorization task, `sentiment analysis`, the extraction of sentiment, the positive or negative orientation that a writer expresses toward some object.<br>

_Sentiment Analysis_ : Duygu analizi, duygusal durumları ve öznel bilgileri sistematik olarak tanımlamak, çıkarmak, ölçmek ve incelemek için doğal dil işleme, metin analizi, hesaplamalı dilbilim ve biyometrinin kullanılmasıdır. <br>

The simplest version of sentiment analysis is a binary classification task, and
the words of the review provide excellent cues. Consider, for example, the following
phrases extracted from positive and negative reviews of movies and restaurants.
Words like great, richly, awesome, and pathetic, and awful and ridiculously are very
informative cues. `Spam detection` is another important commercial application, the binary classification
task of assigning an email to one of the two classes _spam_ or _not-spam_. The task of `language id` is thus the first step in most language processing pipelines. Related text classification tasks like authorship attribution—determining a text’s author—are also relevant to the digital humanities, social sciences, and forensic linguistics. <br>

Even _language modeling_ can be viewed as `classification`: each word can be thought of as a class, and so predicting the next word is classifying the context-so-far into a class for each next word. A **part-of-speech tagger** classifies each occurrence of a word in a sentence as, e.g., a noun or a verb.
The goal of `classification` is to take a single observation, extract some useful features, and thereby classify the observation into one of a set of discrete classes. Most cases of classification in language processing are instead done via **supervised machine learning**. In supervised learning, we have a data set of input observations, each associated with some correct output (a ‘_supervision signal_’). The goal of the algorithm is to learn how to map from a new observation to a correct output.

## 4.7 Evaluation: Precision, Recall, F-measure

To introduce the methods for evaluating text classification, let’s first consider some simple binary detection tasks. For example, in spam detection, our goal is to label every text as being in the spam category (“positive”) or not in the spam category (“negative”). For each item (email document) we therefore need to know whether our system called it spam or not. We also need to know whether the email is actually spam or not, i.e. the human-defined labels for each document that we are trying to match. We will refer to these human labels as the **gold labels**. We need a metric for knowing how well our spam detector is doing. To evaluate any system for detecting things, we start by building a **confusion matrix**.<br>

![confusion matrix](https://firebasestorage.googleapis.com/v0/b/birthday-react-6eca4.appspot.com/o/Natural-Language-Processing%2FWeek%202%20%3A%20Classifying%20Language%2Fcm.jpg?alt=media&token=5402caa2-753f-495d-960d-ca69a4240ce6)

A confusion matrix is a table for visualizing how an algorithm performs with respect to the human gold labels, using two dimensions (system output and gold labels), and each cell labeling a set of possible outcomes. In the spam detection case, for example, `true positives` are documents that are indeed spam (indicated by human-created gold labels) that our system correctly said were spam. `False negatives` are documents that are indeed spam but our system incorrectly labeled as non-spam. To the bottom right of the table is the equation for `accuracy`, which asks what percentage of all the observations (for the spam or pie examples that means all emails or tweets) our system labeled correctly. Although accuracy might seem a natural metric, we generally don’t use it for text classification tasks. That’s because accuracy doesn’t work well when the classes are _unbalanced_ (as indeed they are with spam, which is a large majority of email, or with tweets). In other words, `accuracy` is not a good metric when the goal is
_to discover something that is rare_, or at least not completely balanced in frequency, which is a very common situation in the world. That’s why instead of accuracy we generally turn to two other metrics shown in Fig.4.4: _precision_ and _recall_. `Precision` measures the percentage of the items that
the system detected (i.e., the system labeled as positive) that are in fact positive (i.e., are positive according to the human gold labels). Precision is defined as **Percision** = (true positives) / (true positives + false positives) <br>

`Recall` measures the percentage of items actually present in the input that were correctly identified by the system. Recall is defined as **Recall**= (true positives) / (true positives + false negatives)<br>

`F-measure` comes from a weighted harmonic mean of precision and recall. The _harmonic mean_ of a set of numbers is the reciprocal of the arithmetic mean of reciprocals.<br>

Up to now we have been describing text classification tasks with only two classes. But lots of classification tasks in language processing have more than two classes. For sentiment analysis we generally have 3 classes (_positive_, _negative_, _neutral_) and even more classes are common for tasks like part-of-speech tagging, word sense disambiguation, semantic role labeling, emotion detection, and so on. Luckily the _naive Bayes algorithm_ is already a **multi-class classification algorithm**.

![Confusion Matrix for a 3-class categorization task](https://firebasestorage.googleapis.com/v0/b/birthday-react-6eca4.appspot.com/o/Natural-Language-Processing%2FWeek%202%20%3A%20Classifying%20Language%2Fmulti.jpg?alt=media&token=0f3c4737-adfa-4220-b5f6-a97751450e0e)

**Note :** Bu figürdeki **Gold Label** ve **System Output**(_Predicted label_) yeri farklı. Buna dikkat et!

But we’ll need to slightly modify our definitions of precision and recall. Consider the sample confusion matrix for a hypothetical 3-way one-of email categorization decision _(urgent, normal, spam)_ shown in **Fig. 4.5**. The matrix shows, for example, that the system mistakenly labeled one spam document as urgent, and we have shown how to compute a distinct precision and recall value for each class. In order to derive a single metric that tells us how well the system is doing, we can combine these values in two ways. In `macroaveraging`, we compute the performance for each class, and then average over classes. In `microaveraging`, we collect the decisions for all classes into a single confusion matrix, and then compute precision and recall from that table. **Fig. 4.6** shows the confusion matrix for each class separately, and shows the computation of microaveraged and macroaveraged precision. As the figure shows, a `microaverage` is dominated by the more frequent class (in this case spam), since the counts are pooled. The `macroaverage` better reflects the statistics of the smaller classes, and so is more appropriate when performance on all the classes is equally important.

![Separate confusion matrices for the 3 classes](https://firebasestorage.googleapis.com/v0/b/birthday-react-6eca4.appspot.com/o/Natural-Language-Processing%2FWeek%202%20%3A%20Classifying%20Language%2Fcm2.jpg?alt=media&token=4703f05c-50f6-4b62-b089-6e83a29f6484)

## 4.8 Test sets and Cross-validation

The training and testing procedure for text classification follows what we saw with language modeling : we use the training set to train the model, then use the **development test set** (also called a devset) to perhaps tune some parameters, and in general decide what the best model is. Once we come up with what we think is the best model, we run it on the (hitherto unseen) test set to report its performance. While the use of a devset avoids overfitting the test set, having a fixed training set, devset, and test set creates another problem: in order to save lots of data for training, the test set (or devset) might not be large enough to be representative. Wouldn’t it be better if we could somehow use all our data for training and still use all our data for test? We can do this by **cross-validation**. In `cross-validation`, we choose a number k, and partition our data into k disjoint subsets called folds. Now we choose one of those k folds as a test set, train our classifier on the remaining k􀀀1 folds, and then compute the error rate on the test set. Then we repeat with another fold as the test set, again training on the other _(k-1)_ folds. We do this sampling process k times and average the test set error rate from these k runs to get an average error rate. If we choose k = 10, we would train 10
different models (each on 90% of our data), test the model 10 times, and average these 10 values. This is called `10-fold cross-validation`. The only _problem_ with cross-validation is that because all the data is used for testing, we need the whole corpus to be blind; we can’t examine any of the data to suggest possible features and in general see what’s going on, because we’d be peeking at the test set, and such cheating would cause us to overestimate the performance of our system. However, looking at the corpus to understand what’s going on is important in designing NLP systems! What to do? For this reason, it is common to create a fixed training set and test set, then do 10-fold cross-validation inside the training set, but compute error rate the normal way in the test set.

## 4.9 Statistical Significance Testing

_Idea :_ Statistical significance tests should be used to determine whether we can be
confident that one version of a classifier is better than another.

#### 4.9.1 The Paired Bootstrap Test

The word `bootstrapping` refers to repeatedly drawing large numbers of smaller samples with replacement (called bootstrap samples) from an original larger sample. The intuition of the bootstrap test is that we can create many virtual test sets from an observed test set by repeatedly sampling from it.<br>

#### 4.10 Avoiding Harms in Classification

It’s important, when introducing any NLP model, to study these these kinds of factors and make them clear. One way to do this is by releasing a `model card` for each version of a model. A _model card_ documents a machine learning model with information like:<br>
• training algorithms and parameters<br>
• training data sources, motivation, and preprocessing<br>
• evaluation data sources, motivation, and preprocessing<br>
• intended use and users<br>
• model performance across different demographic or other groups and environmental
situations<br>

# 4.11 Summary

This chapter introduced the naive Bayes model for classification and applied it to
the text categorization task of sentiment analysis.<br>
• Many language processing tasks can be viewed as tasks of classification.<br>
• Text categorization, in which an entire text is assigned a class from a finite set,
includes such tasks as _sentiment analysis_, _spam detection_, _language identification_,
and _authorship attribution_.<br>
• Sentiment analysis classifies a text as reflecting the positive or negative orientation
(sentiment) that a writer expresses toward some object.<br>
• Classifiers are evaluated based on **precision** and **recall**.<br>
• Classifiers are trained using distinct training, dev, and test sets, including the
use of **cross-validation** in the training set.<br>
• Statistical significance tests should be used to determine whether we can be
confident that one version of a classifier is better than another.<br><br>
• Designers of classifiers should carefully consider harms that may be caused
by the model, including its training data and other components, and report
model characteristics in a model card.<br>

# Neural Networks and Neural Language Models (chapter 7)

_Neural networks_ are a fundamental computational tool for language processing,and a very old one.A modern neural network is a network of small computing units, each of which takes a vector of input values and produces a single output value. In this chapter the neural net applied to classification is introduced. The architecture introduced is called a `feedforward network` because the computation proceeds iteratively from one layer of units to the next. The use of modern neural nets is often called deep learning, because modern networks are often deep (have many layers). Neural net classifiers are different from logistic regression in another way. With _logistic regression_, we applied the regression classifier to many different tasks by developing many rich kinds of feature templates based on domain knowledge. When working with neural networks, it is more common to avoid most uses of rich hand-derived
features, instead building neural networks that take raw words as inputs and learn to induce features as part of the process of learning to classify. Nets that are very deep are particularly good at representation learning. For that reason deep neural nets are the right tool for large scale problems that offer sufficient data to learn features automatically.<br>

The three popular _non-linear functions_ f() are the _sigmoid_, the _tanh_, and _the rectified linear unit_ or _ReLU_. The `sigmoid` has a number of advantages; it maps the output into the range [0;1], which is useful in squashing outliers toward 0 or 1 and it’s differentiable.<br>

**Fig. 7.2** shows a final schematic of a basic _neural unit_. In this example the unit takes 3 input values x1,x2,and x3, and computes a weighted sum, multiplying each value by a weight (w1, w2, and w3, respectively), adds them to a _bias_ term _b_, and then passes the resulting sum through a `sigmoid function` to result in a number between 0 and 1.

![neural unit](https://firebasestorage.googleapis.com/v0/b/birthday-react-6eca4.appspot.com/o/Natural-Language-Processing%2FWeek%202%20%3A%20Classifying%20Language%2Fnu.jpg?alt=media&token=8ac475b6-8135-4adf-98bd-b5b648cb89f6)

Let’s walk through an example just to get an intuition. Let’s suppose we have a unit with the following weight vector and bias:<br>
w = [0.2 0.3 0.9]<br>
b = 0.5<br>
With the following input vector:<br>
x = [0.5 0.6 0.1]<br>
The resulting output **y** would be:<br>
![Output of Neural Unit](https://firebasestorage.googleapis.com/v0/b/birthday-react-6eca4.appspot.com/o/Natural-Language-Processing%2FWeek%202%20%3A%20Classifying%20Language%2Fsig.jpg?alt=media&token=bb56af84-0278-4e9d-bcff-21a9591f03c2)

In practice, the sigmoid is not commonly used as an activation function. A function
that is `tanh` very similar but almost always better is the tanh function shown in **Fig. 7.3a**
tanh is a variant of the sigmoid that ranges from -1 to +1:
![tanh](https://firebasestorage.googleapis.com/v0/b/birthday-react-6eca4.appspot.com/o/Natural-Language-Processing%2FWeek%202%20%3A%20Classifying%20Language%2Ftanh.jpg?alt=media&token=71990db5-613d-489c-bc41-c92149e435e8)<br>

The simplest activation function, and perhaps the most commonly used, is the rectified linear unit, also called the `ReLU`. It’s just the same as z when z is positive, and 0 otherwise: y = ReLU(z) = max(z;0)<br>

These activation functions have different properties that make them useful for different language applications or network architectures. For example, the `tanh` function has the nice properties of being smoothly differentiable and mapping outlier values toward the mean. The `rectifier function`, on the other hand has nice properties that result from it being very close to linear. In the sigmoid or tanh functions, very high values of z result in values of y that are saturated, i.e., extremely close to 1, and have derivatives very close to 0.

### 7.3 Feedforward Neural Networks

A `feedforward network` is a _multilayer_ network in which the units are connected with _no cycles_; the outputs from units in each layer are passed to units in the next higher layer, and no outputs are passed
back to lower layers. (In Chapter 9 we’ll introduce networks with cycles, called `recurrent neural networks`.) Simple feedforward networks have three kinds of nodes: input units, hidden units, and output units. The _input layer_ **x** is a vector of simple scalar values just as we saw in **Fig. 7.2**. The core of the neural network is the _hidden layer_ **h** formed of hidden units **hi**,each of which is a neural unit taking a weighted sum of its inputs and then applying a non-linearity. In the standard architecture, each layer is `fully-connected`, meaning that each unit in each layer takes as input the outputs from all the units in the previous layer, and there is a link between every pair of units from two adjacent layers. Thus each hidden unit sums over all the input units.

![Simple Feed Forward Network](https://firebasestorage.googleapis.com/v0/b/birthday-react-6eca4.appspot.com/o/Natural-Language-Processing%2FWeek%202%20%3A%20Classifying%20Language%2Fff.jpg?alt=media&token=d0d80b6e-de9d-4a49-a3c4-13581850cd3d)

Recall that a single hidden unit has as parameters a weight vector and a bias. We represent the parameters for the entire hidden layer by combining the weight vector and bias for each unit i into a single _weight matrix W_ and a single _bias_ vector _b_ for the whole layer. Each elementWji of the weight matrix _W_ represents the weight of the connection from the _ith_ input unit _xi_ to the _jth_ hidden unit _hj_. The advantage of using a single matrix W for the weights of the entire layer is that now the hidden layer computation for a feedforward network can be done very efficiently with simple matrix operations. In fact, the computation only has three steps: multiplying the weight matrix by the input vector x, adding the bias vector b, and applying the activation function g (such as the sigmoid, tanh, or ReLU activation function defined above). The output of the hidden layer, the vector h, is thus the following : `h = σ(Wx+b)` and this resulting value **h** (for hidden but also for hypothesis) forms a _representation_ of the input. The role of the output layer is to take this new _representation h_ and compute a final output. This output could be a realvalued number, but in many cases the goal of the network is to make some sort of classification decision, and so we will focus on the case of classification.<br>

That means we can think of a neural network classifier with one hidden layer as building a vector h which is a hidden layer representation of the input, and then running standard multinomial logistic regression on the features that the network develops in h. Here are the final equations for a feedforward network with a single hidden layer, which takes an _input vector_ **x**, outputs a _probability distribution_ **y**, and is parameterized by _weight matrices_ **W** and **U** and a _bias vector_ **b**:<br>
`h = σ(Wx+b)`<br>
`z = Uh`<br>
`y = softmax(z)`<br>

## 7.4 Feedforward networks for NLP: Classification

Let’s begin with a simple 2-layer sentiment classifier. The input element _x_i_ could be scalar features x_1 = count(words 2 doc), x_2 = count(positive lexicon words 2 doc), x_3 = 1 if “no” 2 doc, and so on. And the output layer ˆy could have two nodes (one each for positive and negative), or 3 nodes (positive, negative, neutral), in which case y_1 would be the estimated probability of positive sentiment, y_2 the probability of negative and y_3 the probability of neutral. The resulting equations would be just what we saw above for a 2-layer network (as always, we’ll continue to use the s to stand for any non-linearity, whether sigmoid, ReLU or other).<br>
`x = [x_1 x_2 ... x_N]` (each x_i is a hand-designed feature)<br>
`h = σ(Wx+b)`<br>
`z = Uh`<br>
`y = softmax(z)`<br>

![Feedforward network sentiment analysis](https://firebasestorage.googleapis.com/v0/b/birthday-react-6eca4.appspot.com/o/Natural-Language-Processing%2FWeek%202%20%3A%20Classifying%20Language%2Fhh.jpg?alt=media&token=896a9150-d5de-4b87-b4f0-de2e401ef6ed)

As we mentioned earlier, adding this hidden layer to our logistic regression classifier allows the network to represent the non-linear interactions between features. This alone might give us a better sentiment classifier. Most neural NLP applications do something different, however. Instead of using
hand-built human-engineered features as the input to our classifier, we draw on deep learning’s ability to learn features from the data by representing words as embeddings, like the word2vec or GloVe embeddings we saw in Chapter 6. There are various ways to represent an input for classification. One simple baseline is to apply some sort of `pooling function` to the embeddings of all the words in the input. Here are the equations for this classifier assuming mean pooling; the architecture is sketched in **Fig. 7.11**:<br>

`x = mean(e(w_1)e(w_2) ... e(w_n))`<br>
`h = s(Wx+b)`<br>
`z = Uh`<br>
`y = softmax(z)`<br>

![Feedforward network sentiment analysis](https://firebasestorage.googleapis.com/v0/b/birthday-react-6eca4.appspot.com/o/Natural-Language-Processing%2FWeek%202%20%3A%20Classifying%20Language%2Fmean.jpg?alt=media&token=a943ca95-6a81-4491-af87-dc6ce11c595e)

The idea of using word2vec or GloVe embeddings as our input representation—and more generally the idea of relying on another algorithm to have already learned an embedding representation for our input words—is called `pretraining`. Using _pretrained_ embedding representations, whether simple static word embeddings like _word2vec_ or the much more powerful contextual embeddings is one of the central ideas of deep learning.

# 7.8 Summary

• Neural networks are built out of neural units, originally inspired by human
neurons but now simply an abstract computational device.<br>
• Each neural unit multiplies input values by a weight vector, adds a bias, and
then applies a non-linear activation function like sigmoid, tanh, or rectified
linear unit.<br>
• In a fully-connected, feedforward network, each unit in layer i is connected
to each unit in layer i+1, and there are no cycles.<br>
• The power of neural networks comes from the ability of early layers to learn
representations that can be utilized by later layers in the network.<br>

# Deep Learning Architectures for Sequence Processing (chapter 9)

This chapter introduces two important deep learning architectures designed to address these challenges: recurrent neural networks and transformer networks. Both approaches have mechanisms to deal directly with the sequential nature of language that allow them to capture and exploit the temporal nature of language. The recurrent network offers a new way to represent the prior context, allowing the model’s decision to depend on information from hundreds of words in the past. The transformer offers new mechanisms (self-attention and positional encodings) that help represent time and help focus on how words relate to each other over long distances.

## 9.2 Recurrent Neural Networks

A _recurrent neural network_ (_RNN_) is any network that contains a **cycle** within its network connections, meaning that the value of some unit is directly, or indirectly, dependent on its own earlier outputs as an input. While powerful, such networks are difficult to reason about and to train. However, within the general class of recurrent networks there are constrained architectures that have proven to be extremely effective when applied to language. These networks are useful in their own right and serve as the basis for more complex approaches like the Long Short-Term Memory (LSTM) networks discussed later in this chapter. In this chapter when we use the term RNN we’ll be referring to these simpler more constrained networks.

![Simple RNN](https://firebasestorage.googleapis.com/v0/b/birthday-react-6eca4.appspot.com/o/Natural-Language-Processing%2FWeek%202%20%3A%20Classifying%20Language%2Fsimprnn.jpg?alt=media&token=c1c3d4aa-02e0-4697-bd24-1e8a1bddd64c)

**Fig. 9.2** illustrates the structure of an `RNN`. As with ordinary feedforward networks, an input vector representing the current input, x*t , is multiplied by a weight matrix and then passed through a non-linear activation function to compute the values for a layer of hidden units. This \_hidden layer* is then used to calculate a corresponding output, y*t . In a departure from our earlier window-based approach, sequences are processed by presenting one item at a time to the network. x_t will mean the input vector x at time t. The key difference from a \_feedforward network* lies in the recurrent link shown in the figure with the dashed line. This link augments the input to the computation at the hidden
layer with the value of the hidden layer from the preceding point in time. The hidden layer from the previous time step provides a form of memory, or context, that encodes earlier processing and informs the decisions to be made at later points in time. Critically, this approach does not impose a fixed-length limit on this prior context; the context embodied in the previous hidden layer can include information extending back to the beginning of the sequence. Adding this temporal dimension makes `RNNs` appear to be more complex than non-recurrent architectures. But in reality, they’re not all that different. Given an
input vector and the values for the hidden layer from the previous time step, we’re still performing the standard feedforward calculation.

![RNNs](https://firebasestorage.googleapis.com/v0/b/birthday-react-6eca4.appspot.com/o/Natural-Language-Processing%2FWeek%202%20%3A%20Classifying%20Language%2Frnns.jpg?alt=media&token=cb753cfd-0518-4306-ab1b-f59550c62957)

The most significant change lies in the new set of weights, U, that connect the hidden layer from the previous time step to the current hidden layer. These weights determine how the network makes use of past context in calculating the output for the current input. As with the other weights in the network, these connections are trained via `backpropagation`.

### 9.2.1 Inference in RNNs

Forward inference (mapping a sequence of inputs to a sequence of outputs) in an RNN is nearly identical to what we’ve already seen with feedforward networks. To compute an output _y_t_ for an input _x_t_ , we need the activation value for the hidden layer _h_t_ . To calculate this, we multiply the input xt with the weight matrix _W_, and the hidden layer from the previous time step _ht-1_ with the weight matrix _U_. We add these values together and pass them through a suitable activation function _g_, to arrive at the activation value for the current hidden layer, _h_t_ . Once we have the values for the hidden layer, we proceed with the usual computation to generate the output vector.<br>

`ht = g(Uh_t-1+Wxt )`<br>
`yt = f (Vht )`<br>

Tailoring the backpropagation algorithm to this situation leads to a two-pass algorithm for training the weights in RNNs. In the first pass, we perform forward inference, computing ht , yt , accumulating the loss at each step in time, saving the value of the hidden layer at each step for use at the next time step. In the second phase, we process the sequence in reverse, computing the required gradients as we go, computing and saving the error term for use in the hidden layer for each step backward in time.

![Simple RNN Unrolled](https://firebasestorage.googleapis.com/v0/b/birthday-react-6eca4.appspot.com/o/Natural-Language-Processing%2FWeek%202%20%3A%20Classifying%20Language%2Fsimprnn.jpg?alt=media&token=f2e48041-b917-452a-90b5-80747b6ee428)

Fortunately, with modern computational frameworks and adequate computing resources, there is no need for a specialized approach to training RNNs. As illustrated in **Fig. 9.5**, explicitly unrolling a recurrent network into a feedforward computational graph eliminates any explicit recurrences, allowing the network weights to be trained directly. In such an approach, we provide a template that specifies the basic structure of the network, including all the necessary parameters for the input, output, and hidden layers, the weight matrices, as well as the activation and output functions to be used. Then, when presented with a specific input sequence, we can generate an unrolled feedforward network specific to that input, and use that graph to perform forward inference or training via ordinary backpropagation. For applications that involve much longer input sequences, such as speech recognition, character-level processing, or streaming of continuous inputs, unrolling an entire input sequence may not be feasible. In these cases, we can unroll the input into manageable fixed-length segments and treat each segment as a distinct training item.

## 9.3 RNNs as Language Models

RNN language models process the input sequence one word at a time, attempting to predict the next word from the current word and the previous hidden state. RNNs don’t have the limited context problem that n-gram models have, since the hidden state can in principle represent information about all of the preceding words all the way back to the beginning of the sequence.<br>

To train an RNN as a language model, we use a corpus of text as training material, having the model predict the next word at each time step t. We train the model to minimize the error in predicting the true next word in the training sequence, using cross-entropy as the loss function. Recall that the cross-entropy loss measures the difference between a predicted probability distribution and the correct distribution. In the case of language modeling, the correct distribution yt comes from knowing the next word. This is represented as a one-hot vector corresponding to the vocabulary where the entry for the actual next word is 1, and all the other entries are 0. Thus, the cross-entropy loss for language modeling is determined by the probability the model assigns to the correct next word. So at time t the CE loss is the negative log probability the model assigns to the next word in the training sequence. Thus at each word position t of the input, the model takes as input the correct sequence of tokens w1:t , and uses them to compute a probability distribution over possible next words so as to compute the model’s loss for the next token wt+1. Then we move to the next word, we ignore what the model predicted for the next word and instead use the correct sequence of tokens w1:t+1 to estimate the probability of token wt+2. This idea that we always give the model the correct history sequence to predict the next word (rather than feeding the model its best case from the previous time step) is called `teacher forcing`.

## 9.4 RNNs for other NLP tasks

### 9.4.1 Sequence Labeling

In sequence labeling, the network’s task is to assign a label chosen from a small fixed set of labels to each element of a sequence, like the part-of-speech tagging and named entity recognition tasks. In an RNN approach to sequence labeling, inputs are word embeddings and the outputs are tag probabilities generated
by a softmax layer over the given tagset.

![Part-of-speech tagging as sequence labeling with a simple RNN](https://firebasestorage.googleapis.com/v0/b/birthday-react-6eca4.appspot.com/o/Natural-Language-Processing%2FWeek%202%20%3A%20Classifying%20Language%2Fpos.jpg?alt=media&token=0e1db0a1-700d-4082-a916-942dbd0c0bc8)

The inputs at each time step are pre-trained word embeddings corresponding to the input tokens. The RNN block is an abstraction that represents an unrolled simple recurrent network consisting of an input layer, hidden layer, and output layer at each time step, as well as the shared U, V and W weight matrices
that comprise the network. The outputs of the network at each time step represent the distribution over the POS tagset generated by a softmax layer. To generate a sequence of tags for a given input, we run forward inference over the input sequence and select the most likely tag from the softmax at each step. Since we’re using a softmax layer to generate the probability distribution over the output tagset at each time step, we will again employ the cross-entropy loss during training.

### 9.4.2 RNNs for Sequence Classification

Another use of RNNs is to classify entire sequences rather than the tokens within them. We’ve already encountered sentiment analysis in Chapter 4, in which we classify a text as positive or negative. Other sequence classification tasks for mapping sequences of text to one from a small set of categories include document-level topic classification, spam detection, or message routing for customer service applications.
To apply RNNs in this setting, we pass the text to be classified through the RNN a word at a time generating a new hidden layer at each time step. We can then take the hidden layer for the last token of the text, hn, to constitute a compressed representation of the entire sequence. We can pass this representation _h_n_ to a feedforward network that chooses a class via a softmax over the possible classes.

Note that in this approach there don’t need intermediate outputs for the words in the sequence preceding the last element. Therefore, there are no loss terms associated with those elements. Instead, the loss function used to train the weights in the network is based entirely on the final text classification task. The output from the softmax output from the feedforward classifier together with a cross-entropy loss drives the training. The error signal from the classification is backpropagated all the way through the weights in the feedforward classifier through, to its input, and then through to the three sets of weights in the RNN as described earlier. The training regimen that uses the loss from a downstream application to adjust the weights all the way through the network is referred to as **end-to-end training**.

### 9.4.3 Generation with RNN-Based Language Models

RNN-based language models can also be used to generate text. Text generation is of enormous practical importance, part of tasks like question answering, machine translation, text summarization, and conversational dialogue; any ask where a system needs to produce text, conditioned on some other text.

We first randomly sample a word to begin a sequence based on its suitability as the start of a sequence. We then continue to sample words conditioned on our previous choices until we reach a pre-determined length, or an end of sequence token is generated. Today, this approach of using a language model to incrementally generate words by repeatedly sampling the next word conditioned on our previous choices is called **autoregressive generation**. Technically an autoregressive model is a model that predicts a value at time t based on a linear function of the previous values at times t 􀀀1, t 􀀀2, and so on. Although language models are not linear (since they have many layers of non-linearities), we
loosely refer to this generation technique as autoregressive generation since the word generated at each time step is conditioned on the word selected by the network from the previous step.

## 9.5 Stacked and Bidirectional RNN architectures

### 9.5.1 Stacked RNNs

In our examples thus far, the inputs to our RNNs have consisted of sequences of word or character embeddings (vectors) and the outputs have been vectors useful for predicting words, tags or sequence labels. However, nothing prevents us from using the entire sequence of outputs from one RNN as an input sequence to another one. Stacked RNNs consist of multiple networks where the output of one layer serves as
the input to a subsequent layer.

![Stacked RNNs](https://firebasestorage.googleapis.com/v0/b/birthday-react-6eca4.appspot.com/o/Natural-Language-Processing%2FWeek%202%20%3A%20Classifying%20Language%2Fstackedrnn.jpg?alt=media&token=1bf6cb67-60a8-4e67-9188-f4b799ba01dc)

`Stacked RNNs` generally outperform single-layer networks. One reason for this success seems to be that the network induces representations at differing levels of abstraction across layers. Just as the early stages of the human visual system detect edges that are then used for finding larger regions and shapes, the initial layers of stacked networks can induce representations that serve as useful abstractions for further layers—representations that might prove difficult to induce in a single RNN. The optimal number of _stacked RNNs_ is specific to each application and to each training set. However, as the number of stacks is increased the training costs rise quickly.

### 9.5.2 Bidirectional RNNs

The RNN uses information from the left (prior) context to make its predictions at time t. But in many applications we have access to the entire input sequence; in those cases we would like to use words from the context to the right of t. One way to do this is to run _two separate RNNs_, one left-to-right, and one right-to-left, and **concatenate** their representations. In the left-to-right RNNs we’ve discussed so far, the hidden state at a given time t represents everything the network knows about the sequence up to that point. The state is a function of the inputs _x_1 ... x_t_ and represents the context of the network to the left of the current time. To take advantage of context to the right of the current input, we can train an RNN on a `reversed input sequence`. With this approach, the hidden state at time t represents information about the sequence to the **right** of the current input.<br>

A `bidirectional RNN` combines two independent RNNs, one where the input is processed from the start to the end, and the other from the end to the start. We then concatenate the two representations computed by the networks into a single vector that captures both the left and right contexts of an input at each point in time.

**Fig. 9.11** illustrates such a bidirectional network that concatenates the outputs of the forward and backward pass. Other simple ways to combine the forward and backward contexts include element-wise addition or multiplication. The output at each step in time thus captures information to the left and to the right of the current input. In sequence labeling applications, these concatenated outputs can serve as the basis for a local labeling decision. Bidirectional RNNs have also proven to be quite effective for sequence classification. Recall from **Fig. 9.8** that for sequence classification we used the final hidden state of the RNN as the input to a subsequent feedforward classifier. A difficulty with this approach is that the final state naturally reflects more information about the end of the sentence than its beginning. Bidirectional RNNs provide a simple solution to this problem; as shown in **Fig. 9.12**, we simply combine the final hidden states from the forward and backward passes (for example by concatenation) and use that as input for follow-on processing.

![Bidirectional RNNs](https://firebasestorage.googleapis.com/v0/b/birthday-react-6eca4.appspot.com/o/Natural-Language-Processing%2FWeek%202%20%3A%20Classifying%20Language%2Fbirnn.jpg?alt=media&token=6c973878-97e1-4104-972a-742efc8f1851)

## 9.6 The LSTM

`The flights the airline was cancelling were full.`<br>

Assigning a high probability to _was_ following _airline_ is straightforward since _airline_ provides a strong local context for the singular agreement. However, assigning an appropriate probability to _were_ is quite difficult, not only because the plural _flights_ is quite distant, but also because the intervening context involves singular constituents. Ideally, a network should be able to retain the distant information about plural flights until it is needed, while still processing the intermediate parts of the sequence correctly. One reason for the inability of RNNs to carry forward critical information is that the hidden layers, and, by extension, the weights that determine the values in the hidden layer, are being asked to perform two tasks simultaneously: provide information useful for the current decision, and updating and carrying forward information required for future decisions.

A second difficulty with training RNNs arises from the need to `backpropagate` the error signal back through time. Recall from Section 9.2.2 that the hidden layer at time t contributes to the loss at the next time step since it takes part in that calculation. As a result, during the backward pass of training, the hidden layers are subject to repeated multiplications, as determined by the length of the sequence. A frequent result of this process is that the gradients are eventually driven to zero, a situation called the `vanishing gradients problem`.<br>

The most commonly used such extension to RNNs is the `Long short-term memory` (_LSTM_) network. LSTMs divide the context management problem into two sub-problems: removing information no longer needed from the context, and adding information likely to be needed for later decision making. The key to solving both problems is to learn how to manage this context rather than hard-coding a strategy into the architecture. LSTMs accomplish this by first adding an explicit context layer to the architecture (in addition to the usual recurrent hidden layer), and through the use of specialized neural units that make use of gates to control the flow of information into and out of the units that comprise the network layers. These gates are implemented through the use of additional weights that operate sequentially on the input, and previous hidden layer, and previous context layers.<br>

The gates in an LSTM share a common design pattern; each consists of a feedforward layer, followed by a sigmoid activation function, followed by a pointwise multiplication with the layer being gated. The choice of the sigmoid as the activation function arises from its tendency to push its outputs to either 0 or 1. Combining this with a pointwise multiplication has an effect similar to that of a binary mask. Values in the layer being gated that align with values near 1 in the mask are passed through nearly unchanged; values corresponding to lower values are essentially erased.

The first gate we’ll consider is the `forget gate`. The purpose of this gate to delete information from the context that is no longer needed. The forget gate computes a weighted sum of the previous state’s hidden layer and the current input and passes that through a sigmoid. This mask is then multiplied element-wise by the context vector to remove the information from context that is no longer required. The next task is compute the actual information we need to extract from the previous hidden state and current inputs—the same basic computation we’ve been using for all our recurrent networks. Next, we generate the mask for the add gate to select the information to add to the current context. Given the appropriate weights for the various gates, an LSTM accepts as input the context layer, and hidden layer from the previous time step, along with the current input vector. It then generates updated context and hidden vectors as output. The hidden layer, ht , can be used as input to subsequent layers in a stacked RNN, or to generate an output for the final layer of a network.

### What is the challenge of sentence segmentation regarding the period character “.” and how can it be addressed?

The challenge is that it's ambitious (like _Mr._ and _Mrs._) and the solution is machine learning to decide if it's a part of a sentence or the end.

### What is morphological parsing?

It is parsing a word such as _cats_ into two morphemes _cat_ and _s_.

### How can neologisms(a newly coined word or expression) be created?

Neologisms are often formed by combining existing words or by giving words new and unique _suffixes_ or _prefixes_. (by combining morphemes). Like **“hangry”** is a _neologism_, it’s just combining things with meanings

### 3 examples of language varieties?

Language variety is a general term for any distinctive form of a language or linguistic expression. Linguists commonly use language variety (or simply variety) as a cover term for any of the overlapping subcategories of a language, including dialect, register, jargon,idiolect, and syntactic variation etc.
