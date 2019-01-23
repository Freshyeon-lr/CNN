import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator

#np.random.seed(3)
width = 100
height = 100
train_datagen = ImageDataGenerator(rescale=1./255)

#(x_train, y_train), (x_test, y_test) = train_datagen.flow_from_directory('/home/yu/img3/train'

train_generator = train_datagen.flow_from_directory('/home/yu/img3/train',target_size=(100,100),batch_size=3,class_mode='categorical')

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory('/home/yu/img3/test',target_size=(100,100),batch_size=3,class_mode='categorical')


model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(100,100,3)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
#model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
#model.add(Dropout(0.25))
model.add(Flatten())

#Flatten : 완전연결계층과 연결하기 위해, 3차원 데이터를 1차원으로
model.add(Dense(256, activation='relu'))
#128개 노드로 구성
#model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit_generator(train_generator,steps_per_epoch=2,epochs=200,validation_data=test_generator,validation_steps=300)

print("--Evaluate--")
scores = model.evaluate_generator(test_generator,steps=5)
print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))

print("-- Predict --")
output = model.predict_generator(test_generator, steps=5)
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
print(test_generator.class_indices)
print(output)


