import numpy as np
from numpy import loadtxt
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense




n= 2000

y_train = np.zeros((n,1),dtype=float)

x_train = np.random.randint(50, size = (n,2))

x_train = np.array(x_train,dtype=float)

# print(x_train)

for i in range (n):
    y_train[i,0] = np.array(x_train[i,0]-x_train[i,1])

x1 = x_train[ : , 0]
x2 = x_train[ : , 1]
y = y_train[ : ]
print(x1,x2,y)
# dataset=loadtxt("/home/smn06/Downloads/sub/sub.csv",delimiter=',')
# x1=dataset[:,0:1]
# x2=dataset[:,1:2]
# y=dataset[:,2]
print(y)
model = keras.Sequential([
     keras.layers.Dense(60,input_shape=(2,), activation=tf.keras.activations.linear),
	keras.layers.Dense(30, activation=tf.keras.activations.linear),
    keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

merged_array = np.stack([x1,x2], axis=1)

model.fit(merged_array,y, epochs=100, batch_size=6)

_,accuracy = model.evaluate(merged_array, y)
print(accuracy)
print('Accuracy: %.2f' % (accuracy*100))

a= np.array([[60,40],[7,5],[44,44],[2,2],[11111,1111]])
print(model.predict(a))