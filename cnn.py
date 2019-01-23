import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
# Todo list1 (H/W) - Validation set, Validation set, Test set 의 의미 찾아보기
# Todo list2 (H/W) - 딥러닝에서 많이 활용되는 Optimizer 다른종류도 적용 해보기(Hint Adam 같은것들)
# Todo list3 (H/W) - Learing rate의 의미 파악해보기, 그리고 Learning rate 변경해서 학습 해보기
# Todo list4-1  - VGG 논문 읽기(논문에서 어려운부분은 아무때나 상관없으니까 연구실 사람들한테 물어볼것, 영어번역이 안되는것도 다포함)
# "Very Deep Convolutional Networks for Large-Scale Image Recognition(VGG)" - https://arxiv.org/pdf/1409.1556.pdf
# Todo list4-2 (H/W) - VGG 논문 구조 따라 모델 구조 작성해보기(주연이 돌아오고 1주뒤까지)
# Todo list2 Hint : Conv2D(채널수(=필터수), kernel_size(필터 사이즈, 필터 사이즈), activation='활성함수종류'

#np.random.seed(3)
width = 100
height = 100
train_datagen = ImageDataGenerator(rescale=1./255)

#(x_train, y_train), (x_test, y_test) = train_datagen.flow_from_directory('/home/yu/img3/train'

train_generator = train_datagen.flow_from_directory('/home/yu/img3/train',target_size=(100,100),batch_size=3,class_mode='categorical')

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory('/home/yu/img3/test',target_size=(100,100),batch_size=3,class_mode='categorical')

#Todo - Convolution 채널낮추고 층수더 깊게 해보기

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
#Todo - fit_generator 에서 Learning rate 변경하기, validation_data에 validation set 추가하기

model.fit_generator(train_generator,steps_per_epoch=2,epochs=200,validation_data=test_generator,validation_steps=300)

print("--Evaluate--")
scores = model.evaluate_generator(test_generator,steps=5)
print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))

print("-- Predict --")
output = model.predict_generator(test_generator, steps=5)
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
print(test_generator.class_indices)
print(output)
