
from read_dataset import import_data

from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, BatchNormalization,Flatten
from keras.optimizers import Adam

#from torchsummary import summary  # install pip install torchsummary


epochs=10
batch_size=64


### load data
path='./own_dataset'
train_x, train_y, val_x, val_y =import_data(path=path, shuffle=True, one_hot_encoding=True, split=True)


### build&compile model

model = Sequential()
model.add(Conv2D(64,kernel_size=3,activation='relu', padding='same',input_shape=(40,40,3)))
model.add(Conv2D(64,kernel_size=3, padding='same',activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D())
model.add(Conv2D(128,kernel_size=3, padding='same',activation='relu'))
model.add(Conv2D(128,kernel_size=3, padding='same',activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D())
model.add(Conv2D(256,kernel_size=3, padding='same',activation='relu'))
model.add(Conv2D(256,kernel_size=3, padding='same',activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D())
model.add(Conv2D(512,kernel_size=5,activation='relu'))
model.add(Conv2D(1024,kernel_size=1,activation='relu'))
model.add(BatchNormalization())

model.add(Conv2D(1024,kernel_size=1,activation='relu'))
model.add(Conv2D(512,kernel_size=1,activation='relu'))
model.add(BatchNormalization())

model.add(Conv2D(5,kernel_size=1,activation='sigmoid'))
model.add(Flatten())

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
model.fit(x=train_x, y=train_y, batch_size=32, epochs=10, validation_data=(val_x,val_y))
