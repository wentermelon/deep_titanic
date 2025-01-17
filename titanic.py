import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Input, Dropout, LeakyReLU
from tensorflow.keras.initializers import RandomNormal, he_normal
from tensorflow.keras import optimizers
from tensorflow.keras import regularizers


train = pd.read_csv(r'train.csv')
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


test_passengers = test['PassengerId'].to_numpy()

test_passengers =  test_passengers.reshape(test_passengers.size, 1)

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

    dataset['Age'] = dataset['Age'].astype(int)

# # Create new feature based on family size:
# # SibSp -> Siblings + Spouse(s)
# # Parch -> Parents + Children
# # +1 to include the person in question.
for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
 
# Remove SibSp and Parch features because it is accounted for in FamilySize.
train = train.drop( ['Parch', 'SibSp'], axis=1)
test = test.drop( ['Parch', 'SibSp'], axis=1)
combine = [train, test]

most_frequent_port = train.Embarked.dropna().mode()[0]

port_mapping = { 'S': 1,
                 'C': 2,
                 'Q': 3 }

for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(most_frequent_port)
    dataset['Embarked'] = dataset['Embarked'].map(port_mapping).astype(int)

test['Fare'].fillna(test['Fare'].dropna().median(), inplace=True)

train['FareBand'] = pd.qcut(train['Fare'], 4)

for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare' ] = 0
    dataset.loc[ (dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare' ] = 1 
    dataset.loc[ (dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31.0), 'Fare' ] = 2
    dataset.loc[ (dataset['Fare'] > 31.0), 'Fare' ] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

train = train.drop(['FareBand'], axis=1)
combine = [train, test]

train = train.to_numpy()

train = train.T

x_train = train[1:, :]
y_train = train[0, :].reshape(1, x_train.shape[1])

x_train, x_validation, y_train, y_validation = train_test_split(    x_train.T, 
                                                                    y_train.T, 
                                                                    test_size = 0.25, 
                                                                    random_state=0, 
                                                                    shuffle=True 
                                                                    )


model = Sequential([
    Input(7),

    Dense(  128, 
            kernel_initializer=he_normal(),
            kernel_regularizer=regularizers.l2(0.001)
            ),

    LeakyReLU(),

    Dropout(0.5),

    Dense(  128, 
            kernel_initializer=he_normal(),
            kernel_regularizer=regularizers.l2(0.001)
            ),
    
    LeakyReLU(),


    Dropout(0.5),
    
    Dense(  128, 
            kernel_initializer=he_normal(),
            kernel_regularizer=regularizers.l2(0.001)
            ),

    LeakyReLU(),

    Dropout(0.5),

    Dense(  1, 
            ),

    Activation('sigmoid'),
    
    ])

# Compile the model to use categorial cross entropy loss, 
model.compile(  
    loss='binary_crossentropy', 
    optimizer='nadam',
    metrics=['accuracy']
    )

# Train the model
history = model.fit(
    x_train, 
    y_train, 
    validation_data=(x_validation, y_validation), 
    epochs=200,
    batch_size=1
    )

# Create a plot using the model's history data for both ACCURACY and LOSS
# and save it as a .png file in the folder.
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('accuracy.png')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('loss.png')

# Prepare and save the results as a .csv file for submission to Kaggle
x_test = test.to_numpy()

y_test = np.rint(model.predict(x_test))

prediction = pd.DataFrame( np.column_stack([test_passengers, y_test]) ).astype(int)

prediction.to_csv('prediction.csv', header=['PassengerId', 'Survived'], index=False)
