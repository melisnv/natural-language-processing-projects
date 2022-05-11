# Simple Transformers

Simple Transformer models are built with a particular **Natural Language Processing (NLP)** task in mind. Each such model comes equipped with features and functionality designed to best fit the task that they are intended to perform. The high-level process of using Simple Transformers models follows the same pattern.

1. Initialize a task-specific model
2. Train the model with **train_model()**
3. Evaluate the model with **eval_model()**
4. Make predictions on _(unlabelled)_ data with **predict()**

The currently implemented task-specific Simple Transformer models, along with their task, are given below.

![transformers models](https://firebasestorage.googleapis.com/v0/b/birthday-react-6eca4.appspot.com/o/Natural-Language-Processing%2FWeek%201%20%3A%20Analyzing%20Language%2Fbertmodel.jpg?alt=media&token=21766942-ec1a-455c-9ba8-cbb39abae5e6)

## Binary Classification

The goal of binary text classification is to classify a text sequence into one of two classes. A transformer-based binary text classification model typically consists of a transformer model with a classification layer on top of it. The classification layer will have two output neurons, corresponding to each class.

## Classification Models

There are two task-specific Simple Transformers classification models, **ClassificationModel** and **MultiLabelClassificationModel**. The two are mostly identical except for the specific use-case and a few other minor differences detailed below.

### ClassificationModel

The `ClassificationModel` class is used for all text classification tasks except for multi label classification.

To create a `ClassificationModel`, you must specify a _model_type_ and a _model_name_.

- `model_type` should be one of the model types from the supported models (e.g. _bert_, electra, xlnet)
- `model_name` specifies the exact architecture and trained weights to use. This may be a Hugging Face Transformers compatible pre-trained model, a community model, or the path to a directory containing model files.

By default, a _ClassificationModel_ will behave as a **binary classifier**. You can specify the number of classes/labels to use it as a multi-class classifier or as a regression model.

## Training a Classification Model

The `train_model()` method is used to train the model. The `train_model()` method is identical for _ClassificationModel_ and _MultiLabelClassificationModel_, except for the _multi_label_ argument being True by default for the latter.

```
model.train_model(train_df)
```

## Evaluating a Classification Model

The `eval_model()` method is used to evaluate the model. The `eval_model()` method is identical for _ClassificationModel_ and _MultiLabelClassificationModel_, except for the multi_label argument being True by default for the latter.

```
result, model_outputs, wrong_predictions = model.eval_model(eval_df)
```

- `result` _(dict)_ - Dictionary containing evaluation results.
- `model_outputs` _(list)_ - List of model outputs for each row in _eval_df_
- `wrong_preds` _(list)_ - List of InputExample objects corresponding to each incorrect prediction by the model.

## References

- https://simpletransformers.ai (https://simpletransformers.ai/) <br/>
- Vrije Universiteit Amsterdam NLP Lecture Notes
