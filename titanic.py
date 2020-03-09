import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import tensorflow as tf
import keras

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
combine = [ train, test ]

# Dropping Ticket and Cabin columns - not useful for analysis.
train = train.drop( ['Ticket', 'Cabin'], axis=1 )
test = test.drop( ['Ticket', 'Cabin' ], axis=1 )
combine = [ train, test ]

# Create new Title feature for model training from Name
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Don', 'Sir', 'Jonkheer', 'Dona', 'Master'], 'Noble')
    dataset['Title'] = dataset['Title'].replace(['Dr', 'Col', 'Major', 'Rev', 'Capt'], 'Professional')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

#print(train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())

# Map Title groups into numerical groups
title_mapping = { 'Mr': 1,
                  'Miss': 2,
                  'Mrs': 3,
                  'Noble': 4,
                  'Professional': 5}

for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

#print(train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())

# Dropping Name and PassengerId features - not useful for analysis.
train = train.drop( ['Name', 'PassengerId'], axis=1 )
test = test.drop( ['Name', 'PassengerId'], axis=1 )
combine = [ train, test ]

# Mapping passenger genders to 0 (Male) or 1 (Female)
gender_mapping = { 'male': 0,
                   'female': 1 }

for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map(gender_mapping).astype(int)

# Check the average survival rate for both genders.
# print(train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean())

# Find average age based on Sex and Pclass features for filling NaN values in the Age column.

male_pclass1 = train.loc[ (train['Sex'] == 0) & (train['Pclass'] == 1) , 'Age'].mean().astype(int)
male_pclass2 = train.loc[ (train['Sex'] == 0) & (train['Pclass'] == 2) , 'Age'].mean().astype(int)
male_pclass3 = train.loc[ (train['Sex'] == 0) & (train['Pclass'] == 3) , 'Age'].mean().astype(int)

female_pclass1 = train.loc[ (train['Sex'] == 1) & (train['Pclass'] == 1) , 'Age'].mean().astype(int)
female_pclass2 = train.loc[ (train['Sex'] == 1) & (train['Pclass'] == 2) , 'Age'].mean().astype(int)
female_pclass3 = train.loc[ (train['Sex'] == 1) & (train['Pclass'] == 3) , 'Age'].mean().astype(int)

# print( "Average Age for Male (Sex=0) in Pclass=1: {0}".format(male_pclass1))
# print( "Average Age for Male (Sex=0) in Pclass=2: {0}".format(male_pclass2))
# print( "Average Age for Male (Sex=0) in Pclass=3: {0}".format(male_pclass3))
# print()
# print( "Average Age for Female (Sex=1) in Pclass=1: {0}".format(female_pclass1))
# print( "Average Age for Female (Sex=1) in Pclass=2: {0}".format(female_pclass2))
# print( "Average Age for Female (Sex=1) in Pclass=3: {0}".format(female_pclass3))

for dataset in combine:
    dataset.loc[ (dataset['Sex'] == 0) & (dataset['Pclass'] == 1) & (dataset['Age'].isna()), 'Age' ] = male_pclass1
    dataset.loc[ (dataset['Sex'] == 0) & (dataset['Pclass'] == 2) & (dataset['Age'].isna()), 'Age' ] = male_pclass2
    dataset.loc[ (dataset['Sex'] == 0) & (dataset['Pclass'] == 3) & (dataset['Age'].isna()), 'Age' ] = male_pclass3

    dataset.loc[ (dataset['Sex'] == 1) & (dataset['Pclass'] == 1) & (dataset['Age'].isna()), 'Age' ] = female_pclass1
    dataset.loc[ (dataset['Sex'] == 1) & (dataset['Pclass'] == 2) & (dataset['Age'].isna()), 'Age' ] = female_pclass2
    dataset.loc[ (dataset['Sex'] == 1) & (dataset['Pclass'] == 3) & (dataset['Age'].isna()), 'Age' ] = female_pclass3

for dataset in combine:
    
    # Children
    dataset.loc[ dataset['Age'] <= 12, 'Age' ] = 0 
    
    # Teenagers
    dataset.loc[ (dataset['Age'] > 12) & (dataset['Age'] <= 18), 'Age' ] = 1 
    
    # Young Adults
    dataset.loc[ (dataset['Age'] > 18) & (dataset['Age'] <= 30), 'Age' ] = 2

    # Adults
    dataset.loc[ (dataset['Age'] > 30) & (dataset['Age'] <= 45), 'Age' ] = 3

    # Middle Age
    dataset.loc[ (dataset['Age'] > 45) & (dataset['Age'] <= 65), 'Age' ] = 4

    # Seniors
    dataset.loc[ dataset['Age'] > 65, 'Age' ] = 5

print(train['Age'].unique())

# # Check the average survival rate based on age ranges.
# #print(train[['Age', 'Survived']].groupby(['Age'], as_index=False).mean())


# # Create new feature based on family size:
# # SibSp -> Siblings + Spouse(s)
# # Parch -> Parents + Children
# # +1 to include the person in question.
# for dataset in combine:
#     dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

# # Check the average survival rate based on family size.
# # print(train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean())

# # Remove SibSp and Parch features because it is accounted for in FamilySize.
# train = train.drop( ['Parch', 'SibSp'], axis=1)
# test = test.drop( ['Parch', 'SibSp'], axis=1)
# combine = [train, test]

# most_frequent_port = train.Embarked.dropna().mode()[0]

# port_mapping = { 'S': 1,
#                  'C': 2,
#                  'Q': 3 }

# for dataset in combine:
#     dataset['Embarked'] = dataset['Embarked'].fillna(most_frequent_port)
#     dataset['Embarked'] = dataset['Embarked'].map(port_mapping).astype(int)

# # print(train[['Embarked', 'Survived']].groupby(['Embarked']).mean())

# # print( train.loc[ train['Age'].isnull() ] )




