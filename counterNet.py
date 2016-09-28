from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution3D,Convolution2D, Flatten



c3d = Convolution3D(32, 3, 3, 10, border_mode='valid', input_shape=(3,10, 100, 100))

model = Sequential()
model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=(3, 100, 100)))

model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(output_dim=32))
model.add(Activation("relu"))
model.add(Dense(output_dim=10))
model.add(Activation("softmax"))
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
