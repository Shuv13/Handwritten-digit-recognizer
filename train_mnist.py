import tensorflow as tf
from tensorflow.keras import layers, utils, Sequential
from tensorflow.keras.datasets import mnist

num_classes = 10
batch_size = 128
epochs = 10

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1,28,28,1).astype("float32") / 255
x_test  = x_test.reshape(-1,28,28,1).astype("float32") / 255
y_train = utils.to_categorical(y_train, num_classes)
y_test  = utils.to_categorical(y_test, num_classes)

model = Sequential([
    layers.Conv2D(32, (3,3), activation="relu", input_shape=(28,28,1)),
    layers.Conv2D(64, (3,3), activation="relu"),
    layers.MaxPooling2D((2,2)),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation="softmax"),
])

model.compile(optimizer="adadelta",
              loss="categorical_crossentropy",
              metrics=["accuracy"])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

model.save("mnist.h5")
