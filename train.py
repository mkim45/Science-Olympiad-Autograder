import tensorflow
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from sklearn.model_selection import train_test_split
from keras.optimizers import RMSprop
from sklearn.preprocessing import LabelBinarizer
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import pandas as pd
from keras.models import load_model
import numpy as np
from sklearn.metrics import accuracy_score

data = pd.read_csv('fermiQuestions/train_digits.csv')
y_init = data['label']
X_init = data.drop('label', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X_init, y_init, test_size = 0.2, random_state = 42)
X_test = X_test.values.reshape(-1, 28, 28, 1) / 255.0

def data():

	X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.25, random_state = 42)
	X_train = X_train.values.reshape(-1, 28, 28, 1) / 255.0
	X_val = X_val.values.reshape(-1, 28, 28, 1) / 255.0
	binarizer = LabelBinarizer()
	y_train = binarizer.fit_transform(y_train)
	y_val = binarizer.transform(y_val)
	return X_train, X_val, y_train, y_val

def modeling():

	# initialize model and constants and compile model
	model = Sequential()
	model.add(Conv2D(32, kernel_size = (5, 5), activation = 'relu', input_shape = (28, 28, 1)))
	model.add(Conv2D(32, kernel_size = (5, 5), activation = 'relu'))
	model.add(MaxPooling2D(pool_size = (2, 2)))
	model.add(Dropout(0.25))
	model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu'))
	model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu'))
	model.add(MaxPooling2D(pool_size = (2, 2)))
	model.add(Flatten())
	model.add(Dense(128, activation = 'relu'))
	model.add(Dense(10, activation = 'softmax'))
	model.compile(optimizer = RMSprop(lr = 0.001, rho = 0.9, epsilon = 1e-08), loss = "categorical_crossentropy",
				  metrics = ["accuracy"])
	early_stopping = EarlyStopping(patience=3, restore_best_weights=True)

	# define x_train and y_train, fit model, and save model
	(X_train, X_val, y_train, y_val) = data()
	epochs = 10
	batch = 32
	model.fit(X_train, y_train, batch_size = batch, epochs = epochs, validation_data = (X_val, y_val),
						callbacks = [early_stopping])
	model.save('updated_model')

# load model
model = load_model('updated_model')
y_test = y_test.values.tolist()

# predict test values
y_pred = model.predict(X_test)
pred = []
for i in range(len(y_pred)):
    test_preds.append(np.argmax(y_pred[i]))

# get accuracy
accuracy = accuracy_score(y_test, test_preds)
print("Test Accuracy:", accuracy)