# Assignment II : Offensive Language Detection

`Content warning:` this assignment contains an analysis of offensive language examples.

In this assignment, we will work with the [OLIDv1 dataset](https://github.com/idontflow/OLID), which contains 13,240 annotated tweets for offensive language detection. The detailed description of the dataset collection and annotation procedures can be found here. This dataset was used in the SemEval 2019 shared task on offensive language detection (OffensEval 2019).

We will focus on `Subtask A` (identify whether a tweet is offensive or not). We preprocessed the
dataset so that label `‘1’` corresponds to offensive messages (`‘OFF’` in the dataset description
paper) and `‘0’` to non-offensive messages (`‘NOT’` in the dataset description paper).

The training and test partitions of the OLIDv1 dataset (olid-train.csv and olid-test.csv,
respectively) can be found [here](https://canvas.vu.nl/courses/59974/files/4963294?wrap=1).
