# deep_titanic
[Titanic: Learning from Disaster Kaggle Competition](https://www.kaggle.com/c/titanic/overview) submission using Deep Learning.

# Introduction

The Titanic Competition on Kaggle is widely known as an entry point to Machine Learning (ML) because it allows new learners to apply their skills to a popular dataset and test their model against many other submissions.

I was inspired by the Deep Learning course I completed in my university and chose this competition to apply my skills. I began by searching the previous top submissions and surprisingly, none of them were using neural networks so I thought it would be a great time to test if neural networks could beat other ML techniques.

# Problem Statement

The problem is relatively simple: 

>Given the passenger data on board of the Titanic, create a model to predict the survivability of any given passenger. 

The dataset is split into two CSV files; train.csv and test.csv. The train and test files contain the same data **except** the test.csv lacks a "Survived" column - this is the data your model will predict and submit.

# Working with the Dataset

The first step in Machine Learning is to analyze the dataset. This gives you an idea on what kind of data you will be working with, including cleaning and feature engineering.

Using the pandas Python package, I loaded the .csv file into a pandas DataFrame object and ran relevant methods to view the data. 

## Feature Engineering

Feature Engineering is the process of transforming the available data into features that are better for analysis. 

One example of this being applied in this project is converting the "titles" of the passengers into numeric values. The reason why this is important is that according to the data, there are many available titles such as Mr., Miss, Mrs., Dr., and even Sir and "Jonkheer". The names do not make a difference to the ML model. To make this useful, I grouped similar titles together to better represent the passenger - male or female, nobility, or professional title. Then, I assigned numeric values from 1-5 to represent the groupings. This way, during training, it can look up the numeric values to see what kind of passenger they were based on title.

This happens often in ML to create useful features from raw data. It results in better analysis and training of the model.

## Cleaning the Data

Once I had a better feel for the data and created new features, I realize there were some missing data in the "Embarkment" and "Fare" columns. To fill these, I used a simple technique of using the median to fill in missing values. There are definitely better methods out there to fill the missing data but as a beginner, I used what I was comfortable with.


# The Model

## Starting Point

The model uses Keras with TensorFlow as the backend. I began with a Sequential model with the following parameters

- Input Layer with 7 inputs
- 3 Dense layers with
  - 10 nodes per layer
  - relu activation 
  - he_normal initialization
- Output Layer with 1 output using sigmoid activation

I wanted to start small to see if a shallow neural network would be able to find the pattern. During testing, it ran up to 75% training and validation accuracy. Comparing it with the previous top submissions, it seemed to be working relatively well but I am confident it can improve. One issue I ran into was a disparity between training and validation results suggesting overfitting, which I will solve at a later point - I focused on increasing the accuracy first.

## More Layers and Nodes!

The easiest way to increase the learning capacity of a neural network is to add more nodes, so that's what I did. I began experimenting with more nodes per layer and more layers overall and found the following results:

- Deeper networks isn't necessarily better - Having the same amount of nodes spread out over more layers didn't improve accuracy and in some cases, resulted in lower accuracy. This could be due to vanishing gradients.
- Wider networks, as in more nodes per layer, helps a lot more than going deeper although it eventually hits a cap in accuracy.
- Changing the shape of the network, i.e. varying the sizes of the layers such as narrow-wide-narrow, had a minimal, if any impact, on the final accuracy. I am sure this does not always hold true but for this case, it did.
  
## Reducing Overfitting

A couple of techniques that worked well in reducing overfitting was the use of Dropout layers and regularizers. I won't dive into too much detail about these but adding Dropout layers between my hidden layers and L2 regularizers to each hidden layer helped a lot in bridging the gap between the training and validation curves.

# Conclusion

After many iterations of testing, the final model hit a training and validation accuracy of roughly 82% and during submission, received a test accuracy of 79.99%.

As a beginner, I had hoped to hit a higher test accuracy, however, I learned a lot from this project such as data analysis, cleaning, feature engineering, and working with deep learning models. 

Credit to [Manav Seghal's Jupyter Notebook](https://www.kaggle.com/startupsci/titanic-data-science-solutions) for help in feature engineering.
