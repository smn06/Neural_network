import tensorflow as tf
print(tf.__version__)
import numpy as np
print(np.version.version)
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)





# import the necessary packages
from keras.models import Sequential
from keras.layers.core import Dense
from tensorflow.keras.optimizers import Adam

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import load_iris
import matplotlib.pyplot as plot
from sklearn.metrics import confusion_matrix



"""dataset import"""

print("[INFO] loading data...")
dataset = load_iris()
(trainX, testX, trainY, testY) = train_test_split(dataset.data,
	dataset.target, test_size=0.25)

"""labeling"""

lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

"""neural network build"""

model = Sequential()
model.add(Dense(3, input_shape=(4,), activation="sigmoid"))
model.add(Dense(64, activation="sigmoid"))
model.add(Dense(32, activation="sigmoid"))
model.add(Dense(16, activation="sigmoid"))
model.add(Dense(3, activation="softmax"))


"""model trainning"""

# train the model using SGD
print("[INFO] training network...")
opt = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
    name='Adam')
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])
H = model.fit(trainX, trainY, validation_data=(testX, testY),
	epochs=250, batch_size=16)

"""evaluation"""

print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=16)
sea=classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1), target_names=dataset.target_names)
print(sea)
con_polt=confusion_matrix(testY.argmax(axis=1),predictions.argmax(axis=1))
print(con_polt)

plot_confusion_matrix(H,testX,testY) 
plot.show()


plot.plot(H.history['accuracy'])
plot.plot(H.history['val_accuracy'])
plot.title('Model accuracy')
plot.ylabel('Accuracy')
plot.xlabel('Epoch')
plot.legend(['Train', 'Test'], loc='upper left')
plot.show()

plot.plot(H.history['loss'])
plot.plot(H.history['val_loss'])
plot.title('Model loss')
plot.ylabel('Loss')
plot.xlabel('Epoch')
plot.legend(['Train', 'Test'], loc='upper left')
plot.show()




