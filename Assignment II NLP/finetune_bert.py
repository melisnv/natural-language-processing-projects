# TODO : Load the training set (olid-train.csv) and analyze the number of instances for each of the two classification labels.

import pandas as pd
import matplotlib.pyplot as plt
from random import randint

data = pd.read_csv("../datas/assignment2/olid-train.csv")
print(data.head())

# Number of Instances
label_zero = data[(data["labels"] == 0)]
number_of_instances_0 = len(label_zero)

label_one = data[(data["labels"] == 1)]
number_of_instances_1 = len(label_one)

print("Class 0:", number_of_instances_0)
print("Class 1:", number_of_instances_1)

# Relative Label Frequency (%)
total = len(data)
freq_zero = (number_of_instances_0 / total)
freq_one = (number_of_instances_1 / total)

print("0 Label Frequency:", freq_zero)
print("1 Label Frequency:", freq_one)

# Example Tweet with This Label
example_tweet_0 = data["text"][(data["labels"] == 0)][4]
example_tweet_1 = data["text"][(data["labels"] == 1)][0]
print("Example tweet for label 0 : ", example_tweet_0)
print("Example tweet for label 1 : ", example_tweet_1)