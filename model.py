import csv
import cv2
import numpy as np
lines = []

#Loading csv file
with open('data/driving_log.csv') as csvdata:
	reader = csv.reader(csvdata)
	
	for line in reader:
		lines.append(line)
		
images = []
measurements = []
k=0

#Extracting image data from paths obtained
for line in lines:
	path1 = line[0]
	path2 = line[1]
	path3 = line[2]
	file_name=[]
	file_name.append(path1.split('/')[-1])
	file_name.append(path2.split('/')[-1])
	file_name.append(path3.split('/')[-1])
	if file_name[2]=='right' or file_name[1]=='left' or file_name[0]=='center':
		continue
	measurement = float(line[3])
	#Restricting intake of 0 angle values
	if k==8000 and measurement == 0.0:
		continue
	#Corrective measurements for offset images	
	measurements.append(measurement)
	measurements.append(measurement+0.2)
	measurements.append(measurement-0.2)	
	for i in file_name:
		curr_path = 'data/IMG/'+i
		img = cv2.imread(curr_path)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		images.append(img)
	k+=1
X_train = np.array(images)
y_train = np.array(measurements)

#importing Keras modules for immplementing model
from keras.models import Sequential
from keras.layers import Flatten, Dense, Activation, Dropout, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

#Model Architecture
model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping = ((70, 20),(0, 0))))
model.add(Convolution2D(6,3,3))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(16,3,3))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Activation('relu'))
model.add(Convolution2D(64,3,3))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(64))
model.add(Dense(128))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dense(1))
#Running model on data loaded
model.compile(optimizer = 'adam', loss = 'mse', metrics = ['accuracy'])
model.fit(X_train, y_train, validation_split = 0.2, shuffle = True, nb_epoch = 4)
#Saving model
model.save('model1.h5')