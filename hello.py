from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras_preprocessing.image import ImageDataGenerator
from keras_preprocessing import image
import PIL
import keras
from keras.utils import np_utils
from contextlib import redirect_stdout

#dataset
(X_train,y_train) , (X_test, y_test)= mnist.load_data('mymnist.db')
img_rows = X_train[0].shape[0]
img_cols = X_train[1].shape[0]

# Getting our date in the right 'shape' needed for Keras
# We need to add a 4th dimenion to our date thereby changing our
# Our original image shape of (60000,28,28) to (60000,28,28,1)
X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)

# store the shape of a single image
input_shape = (img_rows, img_cols, 1)

# change our image type to float32 data type
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# Normalize our data by changing the range from (0 to 255) to (0 to 1)
X_train /= 255
X_test /= 255

# Now we one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

num_classes = y_test.shape[1]
num_pixels = X_train.shape[1] * X_train.shape[2]
#model compiling
model = Sequential()
i=1
filters = 2
for i in range(i):
    model.add(Convolution2D(filters = filters,
                        kernel_size = (3,3),
                        activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    filters *= 2
    
model.add(Flatten())


#addingDense
model.add(Dense(units = 3 , activation = 'relu')) 


    
model.add(Dense(units = 10 , activation = 'softmax'))

model.compile(loss = 'categorical_crossentropy',
              optimizer = keras.optimizers.Adadelta(),
              metrics = ['accuracy'])

# Training Parameters
batch_size = 128
epochs = 10

model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(X_test, y_test),
          )


# Evaluate the performance of our trained model
scores = model.evaluate(X_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])


with open('accuracy.txt', 'w') as f:
    with redirect_stdout(f):
        print(str(int(scores[1]*100)))