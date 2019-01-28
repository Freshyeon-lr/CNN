import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.optimizers import SGD

np.random.seed(2)
width = 100
height = 100
train_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory('/home/yu/img3/train',target_size=(100,100),batch_size=2,class_mode='categorical')

validation_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = validation_datagen.flow_from_directory('/home/yu/img3/validation', target_size = (100,100),batch_size=2,class_mode='categorical')

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory('/home/yu/img3/test',target_size=(100,100),batch_size=2,class_mode='categorical')


model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), strides=(1,1), activation='relu',input_shape=(100,100,3)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())

#Flatten : 완전연결계층과 연결하기 위해, 3차원 데이터를 1차원으로
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])

model.fit_generator(train_generator,steps_per_epoch=500,epochs=50,validation_data=validation_generator,validation_steps=250)

print("--ValEvaluate--")
scores = model.evaluate_generator(validation_generator,steps=5)
print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))

print("-- ValPredict --")
output = model.predict_generator(validation_generator, steps=5)
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
print(validation_generator.class_indices)
print(output)

print("--Evaluate--")
scores = model.evaluate_generator(test_generator,steps=5)
print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))

print("-- Predict --")
output2 = model.predict_generator(test_generator, steps=5)
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
print(test_generator.class_indices)
print(output2)
#print(test_generator.filenames)
