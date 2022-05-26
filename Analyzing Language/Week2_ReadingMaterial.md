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
