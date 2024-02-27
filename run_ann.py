import keras
from keras.models import Sequential
from keras import layers

import pandas
import numpy


dataset = pandas.read_csv("dataset.csv")
dataset = dataset.sample(frac=1)


target = dataset.iloc[:,-1].values
data = dataset.iloc[:,:-1].values
data = data/255.0

machine = Sequential()
machine.add(layers.Dense(512, 
            activation="sigmoid", 
            input_shape=(data.shape[1],)  
            ))
machine.add(layers.Dense(128, 
            activation="sigmoid"))
machine.add(layers.Dense(64, 
            activation="sigmoid"))
machine.add(layers.Dense(10, activation="softmax"))
machine.compile(optimizer="sgd", 
                loss="sparse_categorical_crossentropy", 
                metrics=['accuracy'])
  

machine.fit(data, target, epochs=90, batch_size=64)

# new_target = numpy.argmax(machine.predict(new_data), axis=-1)


