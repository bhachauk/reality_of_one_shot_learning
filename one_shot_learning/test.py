from keras.models import Sequential
from keras.layers import Dense
from matplotlib import pyplot as plt
import numpy as np
import time

model = Sequential()
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

a = np.array(range(1000)).reshape(-1, 1)
b = np.array([0] * 500 + [1] * 500).reshape(-1, 1)


def test(epoch):
    start = time.time()
    history = model.fit(a, b, epochs=epoch, verbose=0)
    print("Epoch size : {}, training time : {} s".format(epoch, time.time() - start))
    score = model.evaluate(a, b)
    print("For epoch : {}, Loss : {}, Score : {}".format(epoch, score[0], score[1]))
    plt.plot(history.history['accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train'], loc='upper left')
    plt.savefig("HighEpoch_Train.png")
    plt.close()
    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train'], loc='upper left')
    plt.savefig("HighEpoch_Test.png")
    plt.close()


test(500)

for i in range(450, 550):
    print(i," : ", model.predict([i])[0])
