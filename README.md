# deep_titanic
[Titanic: Learning from Disaster Kaggle Competition](https://www.kaggle.com/c/titanic/overview) submission using Deep Learning.

# Introduction

Despite being called a 'competition', the Titanic Competition on Kaggle is more widely known as an entry point to Machine Learning (ML) because it allows new learners to apply their skills to a popular dataset and test their model against many other previous submissions.

I was inspired by the Deep Learning course I completed in my university and chose this competition to apply my skills. I began by searching the previous top submissions and surprisingly, none of them were using neural networks so I thought it would be a great time to test if neural networks could beat other ML techniques.

# Problem Statement

The problem is relatively simple: 

>Given the passenger data on board of the Titanic, create a model to predict the survivability of any given passenger. 

The dataset is split into two CSV files; train.csv and test.csv. The train and test files contain the same data **except** the test.csv lacks a "Survived" column - this is the data your model will predict and submit.

# Working with the Dataset

## Analyzing the Dataset



## Feature Engineering

# Model

## Starting Point

The model uses Keras with TensorFlow as the backend. I began with a Sequential model with the following parameters

- Input Layer with 7 inputs
- 3 Dense layers with
  - 10 nodes per layer
  - relu activation 
  - he_normal initialization
- Output Layer with 1 output (0 or 1) with sigmoid activation

I wanted to start small to see if a shallow neural network would be able to find the pattern. During testing, it ran up to 80% training and validation accuracy. Comparing it with the previous top submissions, it seemed to be working relatively well but I am confident it can improve. One issue I ran into was a disparity between training and validation results suggesting overfitting, which I will solve at a later point - I will focus on increasing the accuracy first.

## Architectural Changes

The easiest way to increase the learning capacity of a neural network is to add more nodes, so that's what I did. I began experimenting with more nodes per layer and more layers overall and found the following results:

- Deeper networks isn't necessarily better - Having the same amount of nodes spread out over more layers didn't improve accuracy and in some cases, resulted in lower accuracy.
- Wider networks, as in more nodes per layer, helps a lot more than going deeper although it eventually hits a cap in accuracy.
- Changing the shape of the network, i.e. varying the sizes of the layers such as narrow-wide-narrow, had a minimal, if any impact, on the final accuuracy.


## Reducing Overfitting


## Final Model

# Submission

Credit to [Manav Seghal's Notebook](https://www.kaggle.com/startupsci/titanic-data-science-solutions) for help in feature engineering.
