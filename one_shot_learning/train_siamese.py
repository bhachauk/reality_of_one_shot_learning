from one_shot_learning import util
from keras.layers.core import *
from keras.layers import *
from keras import *
from keras_vggface.vggface import VGGFace
from keras.optimizers import Adam
import matplotlib.pyplot as plt


input_shape = (243, 243, 3)
initialize_weights = 'random_uniform'
initialize_bias = 'zeros'
temp_set = list()


def get_siamese_data_format(image_paths_dict):
    data = list()
    for x in image_paths_dict.keys():
        temp_class = None
        for i in image_paths_dict[x]:
            for y in image_paths_dict.keys():
                for j in image_paths_dict[y]:
                    tp = (i, j)
                    if i == j or set(tp) in temp_set:
                        continue
                    elif x == y:
                        data.append((util.input_label_map[x], i, j, 1))
                    elif not temp_class == y:
                    # else:
                        data.append((util.input_label_map[x], i, j, 0))
                    temp_class = y
                    temp_set.append(set(tp))
    print("Collected data set size : {}".format(len(data)))
    temp_set.clear()
    return data


def get_renamed_model(model, num):
    for layer in model.layers:
        model.get_layer(name=layer.name).name = "{}_{}".format(layer.name, num)
    return model


def get_renamed_models(model1, model2):
    model1 = get_renamed_model(model1, 1)
    model2 = get_renamed_model(model2, 2)
    return model1, model2


def get_siamese_model():
    model1 = VGGFace(model="resnet50", include_top=False, input_shape=(243, 243, 3), pooling='avg')
    model2 = VGGFace(model="resnet50", include_top=False, input_shape=(243, 243, 3), pooling='avg')
    model1, model2 = get_renamed_models(model1, model2)
    l1_layer = Lambda(lambda tensors: K.sqrt(K.maximum(K.sum(K.square(tensors[0] - tensors[1]), axis=1,
                                                             keepdims=True), K.epsilon())))
    l1_distance = l1_layer([model1.layers[-1].output, model2.layers[-1].output])
    prediction = Dense(1, activation='sigmoid', bias_initializer=initialize_bias)(l1_distance)
    siamese_net = Model(inputs=[model1.layers[0].input, model2.layers[0].input], outputs=prediction, name="siamese_resnet")
    return siamese_net


def get_siamese_data(train_dir):
    image_paths_dict = util.get_data_arrays(train_dir)
    siamese_data = get_siamese_data_format(image_paths_dict)
    return siamese_data


def parse_siamese_data(siamese_data):
    train_data_left = list()
    train_data_right = list()
    target_list = list()
    print("Siamese Test data :")
    for x in siamese_data:
        print("{} --> {} : {}".format(x[1], x[2], x[-1]))
        left_image = util.get_vgg_read_image(x[1])
        right_image = util.get_vgg_read_image(x[2])
        train_data_left.append(left_image)
        train_data_right.append(right_image)
        target_list.append(x[-1])
    print("####################")
    return train_data_left, train_data_right, target_list


def main():
    train_dir = util.train_dir
    siamese_data = get_siamese_data(train_dir)
    train_data_left, train_data_right, target_list = parse_siamese_data(siamese_data)
    siamese_model = get_siamese_model()
    siamese_model.compile(optimizer=Adam(lr=0.0006), loss="binary_crossentropy", metrics=['accuracy'])
    print("Started Fitting Model...")
    print("Input shape : {}".format(np.array(train_data_left).shape))
    history = siamese_model.fit([train_data_left, train_data_right], target_list, epochs=50, batch_size=10,
                                validation_split=0.1)
    scores = siamese_model.evaluate([train_data_left, train_data_right], target_list, verbose=0)
    print("Training Score %s: %.2f%%" % (siamese_model.metrics_names[1], scores[1] * 100))
    siamese_model.save("siamese_resnet_model.h5")
    # Plot training & validation accuracy values
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig("SiameseResNet_Accuracy.png")
    plt.close()
    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig("SiameseResNet_Loss.png")
    plt.close()


if __name__ == '__main__':
    main()
