# Week 2 : Classifying Language

Represent language in a way that a computer can process it.

### Single-label Classification

Input can be a word, word pair, sentence, sentence pair, paragraph, document and output can be a single label.

### Multi-label Classification

Output can be multiple labels. In practice, multi-label problems are often transformed into multiple
single-label classification problems.

## Classifier Development

### 1. Development and training:

Represent the input (training data)
Develop a model
Train the model

### 2. Evaluation:

Represent the input (development data)
Make predictions with trained model
Evaluate the predictions

#### Improve:

Modify model or input representation, add regularization to avoid overfitting.

# Supervised Machine Learning

## Evaluation

### Precision vs Recall and F1 Score

True Positive ve True Negative modelin doğru olarak tahminlediği, False Positive ve False Negative ise modelin yanlış olarak tahminlediği alanlardır. Kesinlik(Precision) Positive olarak tahminlediğimiz değerlerin gerçekten kaç adedinin Positive olduğunu göstermektedir. Kesinlik değeri özellikle False Positive tahminlemenin maliyeti yüksek olduğu durumlarda çok önemlidir. Duyarlılık(Recall) ise Positive olarak tahmin etmemiz gereken işlemlerin ne kadarını Positive olarak tahmin ettiğimizi gösteren bir metriktir.

Precision (Kesinlik) = True Positives / (True Positives + False Positives)
Recall(Duyarlılık) = True Positives / (True Positives + False Negatives)
Tuning for high precision → The system should not make a mistake.
Tuning for high recall → The system should not miss a case.

F1 Score değeri bize Kesinlik (Precision) ve Duyarlılık (Recall) değerlerinin harmonik ortalamasını göstermektedir. Basit bir ortalama yerine harmonik ortalama olmasının sebebi ise uç durumları da gözardı etmememiz gerektiğidir. Eğer basit bir ortalama hesaplaması olsaydı Precision değeri 1 ve Recall değeri 0 olan bir modelin F1 Score’u 0.5 olarak gelecektir ve bu bizi yanıltacaktır. Doğruluk (Accuracy) yerine F1 Score değerinin kullanılmasının en temel sebebi eşit dağılmayan veri kümelerinde hatalı bir model seçimi yapmamaktır.

# Classification with Feed-Forward Networks

How can we represent a sentence so that the computer can process it?
As a vector x with n dimensions → Representing Language
Represent each input i as a vector xi. Number of dimensions for x is an experimental choice.
Prediction: a probability distribution over all languages in the training set.
Input: C’est bon la vie. → x1 = [0.2 0.4 0.1] Neural networks do not output a probability distribution. Output: FRA NED → o1 = [0.9 0.5] We need to take the softmax-function over the output vector: softmax(o1) = [0.57,0.43]

#### Hidden States

In the hidden states, the network learns abstractions over the input to extract the relevant information for predicting the output. The arrows symbolize weights to determine the importance of each input element.
h = f(Wx) o = g(Uh)
○ x, h, o are vectors
○ W and U are weight matrices
○ f and g are non-linear activation functions (tanh, sigmoid, ...)
Dimensions of o are determined by dataset, dimensions of x and h are experimental choices. We can add more hidden layers to our network. The output of the previous layer serves as input for the current layer. In the previous slides, we looked at one specific toy example to understand what is going on.
In the literature, the visualization is usually simplified to abstract from the exact number of dimensions. In order for to learn good weights start with a random initialization.
Train the network:

1. Calculate the predicted output with the current weights.
2. Compare the predictions to the expected output and calculate the error.
   Adjust the weights relative to the size of the error, their contribution to the error,the learning rate.

During learning, the error on the training set hopefully decreases. If we are too perfectionist at this stage, the network is too fitted to the training data and is not useful for new data anymore. To avoid this, we need to apply regularization and early stopping.

# Sequence Labeling

Feed-forward networks work well if the order of the input is not relevant and if the output decisions are independent.
Recurrent neural networks (RNNs) are sensitive to word order and can process arbitrarily long sequences. RNNs have improved sequential tasks in NLP tremendously. Due to the shared weights, we can fold the RNN. The hidden state is often called the activation or the memory. The prediction of the next word depends on the previous predictions. Always provide the correct sequence history for the next prediction during training instead of relying on faulty model predictions (avoid error propagation).

# LSTMs

Long short-term memory networks (LSTMs) are special recurrent neural networks. They have an additional context layer and use three types of gates to control the flow of information such as forget gate, add gate and output gate. The gates are implemented through the use of additional weights that operate
sequentially on the input, and the previous hidden layer and previous context layers.
