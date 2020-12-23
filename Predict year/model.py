import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

databis=pd.read_csv('/Users/luciedrevet/Documents/S7/Python for data analysis/Projet/YearPredictionMSD.csv', names=features)
databis['year'] = databis.year.apply(lambda year : year-(year%10))
databis.iloc[:,1:] = (databis.iloc[:,1:]-databis.iloc[:,1:].min())/(databis.iloc[:,1:].max() - databis.iloc[:,1:].min())

databis.drop(databis.iloc[:, 11:], inplace = True, axis = 1) 
databis.drop(databis.iloc[:, 7:10], inplace = True, axis = 1) 
databis.drop(databis.iloc[:,4:5], inplace = True, axis = 1)

# separate train attributes and test into different dataframes
Xg = databis.iloc[:,1:]
Yg = databis.iloc[:,0]
Yg = Yg - Yg.min()    

Traing = databis.iloc[0:463715]
Testg = databis.iloc[463715:]

# Train set
X_traing = Xg.iloc[0:463715].values
y_traing = Yg.iloc[0:463715].values

# Validation set
X_testg = Xg.iloc[463715:].values
y_testg = Yg.iloc[463715:].values
print("Train grouped : X ", X_traing.shape, ", Y ", y_traing.shape)
print("Test grouped: X ", X_testg.shape, ", Y ", y_testg.shape)

Y_traing = np_utils.to_categorical(y_traing-1, 90)
Y_testg = np_utils.to_categorical(y_testg-1, 90)

model3 = Sequential()
model3.add(Dense(90, input_shape=(6,), activation='tanh'))
model3.add(Dense(110, activation='tanh'))
model3.add(Dropout(0.2))
model3.add(Dense(90, activation='sigmoid'))

model3.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])

h3=model3.fit(
    X_traing,
    Y_traing,
    batch_size=64,
    epochs=4,
    validation_data=(X_testg, Y_testg)
)

P3 = model3.predict_classes(X_testg)
P3 = P3 + 1
print(P3)

cnf_matrix10 = metrics.confusion_matrix(y_testg, P3)
a4 = accuracy_score(y_testg, P3)
print(a4)