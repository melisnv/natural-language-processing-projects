# Majority baseline: always assigns the majority class of the training data
# Random baseline: randomly assigns one of the classes. Make sure to set a random seed and average the accuracy over 100 runs.

import pandas as pd
from model.data_loader import DataLoader
from collections import Counter
from sklearn.metrics import accuracy_score
import random
import spacy
from wordfreq import zipf_frequency, word_frequency

nlp = spacy.load("en_core_web_sm")


# Each baseline returns predictions for the test data. The length and frequency baselines determine a threshold using the development data.

def majority_baseline(train_text_input, train_labels_input, test_text_input, test_labels_input):
    train_labels_list = []

    for element in train_text_input:
        train_labels_list.append(element.strip())

    train_labels_list = " ".join(train_labels_list).split(" ")
    train_labels_list

    one_label_occurences = train_labels_list.count(1)
    zero_label_occurences = train_labels_list.count(0)

    if one_label_occurences > zero_label_occurences:
        majority_class = 1
    else:
        majority_class = 0

    predictions = []
    tokens_list = []
    for instance in test_text_input:
        tokens = instance.split(" ")
        instance_predictions = [majority_class for t in tokens]
        predictions.append(instance_predictions)
        tokens_list.append(tokens)

    final_predictions = [item for sublist in predictions for item in sublist]
    final_tokens = [item for sublist in tokens_list for item in sublist]

    test_labels_list = []
    for element in test_labels_input:
        test_labels_list.append(element)

    final_df = pd.DataFrame({"tokens": final_tokens, "predictions": final_predictions})
    length = len(test_labels_list)
    accuracy = accuracy_score(test_labels_list, final_predictions[:length])

    return accuracy, final_df


def random_baseline(train_text_input, train_labels_input, test_text_input, test_labels_input):
    subjects = [1, 1]

    predictions = []
    tokens_list = []
    for instance in train_text_input:
        tokens = instance.split(" ")
        instance_predictions = [random.choice(subjects) for t in tokens]
        predictions.append(instance_predictions)
        tokens_list.append(tokens)

    predictions = [item for sublist in predictions for item in sublist]
    final_tokens = [item for sublist in tokens_list for item in sublist]

    test_labels_list = []
    for element in test_labels_input:
        test_labels_list.append(element)

    final_df = pd.DataFrame({"tokens": final_tokens, "predictions": predictions})
    length = len(test_labels_list)
    accuracy = accuracy_score(test_labels_list, predictions[:length])
    return accuracy, final_df



if __name__ == '__main__':

    # Note: this loads all instances into memory. If you work with bigger files in the future, use an iterator instead.
    train_text_input = pd.read_csv("data/olid-train.csv", encoding='utf-8')['text']
    train_labels_input = pd.read_csv("data/olid-train.csv", encoding='utf-8')['labels']

    dev_text_input = pd.read_csv("data/olid-subset-diagnostic-tests.csv", encoding='utf-8')['text']
    dev_labels = pd.read_csv("data/olid-subset-diagnostic-tests.csv", encoding='utf-8')['labels']

    test_text_input = pd.read_csv("data/olid-test.csv", encoding='utf-8')['text']
    test_labels_input = pd.read_csv("data/olid-test.csv", encoding='utf-8')['labels']


    majority_accuracy, majority_predictions = majority_baseline(train_text_input, train_labels_input, test_text_input, test_labels_input)
    majority_accuracy_dev, majority_predictions_dev = majority_baseline(train_text_input, train_labels_input, dev_text_input,
                                                                        dev_labels)

    print("Test acuracy for majority test:", majority_accuracy)
    print("Test acuracy for majority dev:", majority_accuracy_dev)

    random_accuracy, random_predictions = random_baseline(train_text_input, train_labels_input, test_text_input, test_labels_input)
    random_accuracy_dev, random_predictions_dev = random_baseline(train_text_input, train_labels_input, dev_text_input,
                                                                        dev_labels)

    print("Test acuracy for random test:", random_accuracy)
    print("Test acuracy for random dev:", random_accuracy_dev)


    majority_predictions.to_csv("experiments/baselines/majority_test.csv", index=False)
    random_predictions.to_csv("experiments/baselines/random_test.csv", index=False)