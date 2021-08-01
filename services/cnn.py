import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.losses import categorical_crossentropy  
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.utils import to_categorical


# Parameters
num_features = 32
num_labels = 7  
batch_size = 64  
epochs = 100
width, height = 48, 48  


def read_data():
	data = pd.read_csv('../files/fer2013.csv')
	"""
	Create diffrent lists to store training and testing image pixels
	"""
	X_train = []
	y_train = []
	X_test = []
	y_test = []
	for index, row in data.iterrows():
	    pixels = row['pixels'].split(' ')
	    #pixels = list(map(int, pixels))
	    if row['Usage'] == 'Training':
	        X_train.append(np.array(pixels, 'float32'))
	        y_train.append(row['emotion'])
	    elif row['Usage'] == 'PublicTest':
	        X_test.append(np.array(pixels, 'float32'))
	        y_test.append(row['emotion'])
	return (X_train, y_train, X_test, y_test)


def prepare_training():
	X_train, y_train, X_test, y_test = read_data()
	"""
	Prepare training and testing data before send into CNN model
	"""
	X_train = np.array(X_train, 'float32')
	y_train = np.array(y_train, 'float32')
	X_test = np.array(X_test, 'float32')
	y_test = np.array(y_test, 'float32')

	# Normalize data
	X_train -= np.mean(X_train, axis=0)
	X_train /= np.std(X_train, axis=0)  
	X_test -= np.mean(X_test, axis=0)  
	X_test /= np.std(X_test, axis=0) 

	X_train = X_train.reshape(X_train.shape[0], 48, 48, 1)
	X_test = X_test.reshape(X_test.shape[0], 48, 48, 1)

	y_train = to_categorical(y_train, num_classes=num_labels)
	y_test = to_categorical(y_test, num_classes=num_labels)
	return (X_train, y_train, X_test, y_test)


def train_CNN():
    X_train, y_train, X_test, y_test = prepare_training()

	"""
	CNN model for emotion detection with different layers
	"""
	model = Sequential()

	model.add(Conv2D(num_features, kernel_size=(3, 3), activation='relu', input_shape=(width, height, 1)))
	model.add(BatchNormalization())
	model.add(Conv2D(num_features, kernel_size=(3, 3), activation='relu', padding='same'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
	model.add(Dropout(0.5))

	model.add(Conv2D(2*num_features, (3, 3), activation='relu', padding='same'))
	model.add(BatchNormalization())
	model.add(Conv2D(2*num_features, (3, 3), activation='relu', padding='same'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
	model.add(Dropout(0.5))

	model.add(Conv2D(2*2*num_features, (3, 3), activation='relu', padding='same'))
	model.add(BatchNormalization())
	model.add(Conv2D(2*2*num_features, (3, 3), activation='relu', padding='same'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
	model.add(Dropout(0.5))

	model.add(Conv2D(2*2*2*num_features, (3, 3), activation='relu', padding='same'))
	model.add(BatchNormalization())
	model.add(Conv2D(2*2*2*num_features, (3, 3), activation='relu', padding='same'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
	model.add(Dropout(0.5))

	model.add(Flatten())

	# FC neural networks 
	model.add(Dense(2*2*2*2*num_features, activation='relu'))
	model.add(Dropout(0.4))
	model.add(Dense(2*2*2*2*num_features, activation='relu'))
	model.add(Dropout(0.4))

	model.add(Dense(num_labels, activation='softmax'))

	# Compile the model
    model.compile(optimizer=Adam(), loss=categorical_crossentropy, metrics=['accuracy'])

    #training the model
    model.fit(X_train, y_train, batch_size=batch_size, epochs=100, verbose=1, validation_data=(X_test, y_test), shuffle=True)


if __name__ == '__main__':
	train_CNN()