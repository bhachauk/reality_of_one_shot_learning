import os, shutil
import time
import numpy as np
import face_recognition
import cv2
from tabulate import tabulate
from keras_vggface.vggface import VGGFace
from keras.models import load_model
import logging
logging.basicConfig(filename='logs.out', format='%(asctime)-15s : %(filename)s:%(lineno)s : %(funcName)s() '
                                                          ': %(message)s', filemode='a',
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)


def get_logging():
    return logging


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
vgg_model_resnet = VGGFace(model="resnet50", include_top=False, input_shape=(243, 243, 3), pooling='avg')
open_keras_model = None #load_model(os.path.realpath("models/nn4.small2.v1.h5"))
current_feature_size = 0


def get_all_dir(input_dir):
    return [x for x in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, x))]


def get_all_files(input_dir):
    return [x for x in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, x))]


def get_data_arrays(input_dir):
    master_paths_array = dict()
    for class_name in get_all_dir(input_dir):
        master_paths_array[class_name] = list()
        for image_file in get_all_files(os.path.join(input_dir, class_name)):
            master_paths_array[class_name].append(os.path.join(input_dir, class_name, image_file))
    return master_paths_array


def get_emb_arrays(path_dict, m):
    global current_feature_size
    embed_dict = dict()
    for c in path_dict.keys():
        embed_dict[c] = list()
        for path in path_dict[c]:
            embed = models_dict[m](path)
            if isinstance(embed, int):
                continue
            embed_dict[c].append(embed)
            current_feature_size = len(embed)
    return embed_dict


def get_vgg_read_image(img_path):
    loaded_image = face_recognition.load_image_file(img_path)
    loaded_image = cv2.resize(loaded_image, dsize=(243, 243), interpolation=cv2.INTER_CUBIC)
    return loaded_image


def get_vgg_deep_read_image(img_path):
    loaded_image = face_recognition.load_image_file(img_path)
    loaded_image = cv2.resize(loaded_image, dsize=(96, 96), interpolation=cv2.INTER_CUBIC)
    return loaded_image


def get_vgg_resnet_emb_array(img_path):
    loaded_image = get_vgg_read_image(img_path)
    return vgg_model_resnet.predict(np.expand_dims(loaded_image, 0))[0]


def get_dlib_emb_array(image_path):
    img = face_recognition.load_image_file(image_path)
    d = face_recognition.face_encodings(img)
    return d[0] if d else 0


def get_open_emb_array(img_path):
    global open_keras_model
    return open_keras_model.predict(np.expand_dims(np.array(get_vgg_deep_read_image(img_path)).T, axis=0))[0]


def face_distance(face_encodings, face_to_compare):
    if len(face_encodings) == 0:
        return np.empty((0))
    return np.linalg.norm(face_encodings - face_to_compare, axis=1)


def minimum_distance_classifier(train_embed_dict, face_encoding_to_check):
    result = {}
    for class_name in train_embed_dict.keys():
        x_embed_data = train_embed_dict[class_name]
        distances = face_distance(x_embed_data, face_encoding_to_check)
        avg = sum(distances)/len(distances)
        logging.info("[Test] : Average distance value : {} for the class : {}".format(str(avg), class_name))
        logging.info("[Test] : Average distance value : {} for the class : {}".format(str(avg), class_name))
        result[class_name] = avg
    result_class = min(result, key=result.get)
    logging.info("[Result] : Minimum distance observed : {} for the class : {}".format(str(result[result_class]),
                                                                                       result_class))
    return [result[result_class], result_class]


class ModelMetrics:

    total_time = 0   # creation time per image
    total_images_size = 0 # total images size
    result = dict()
    feature_size = 0

    def __init__(self, name):
        self.name = name
        self.start_time = time.time()

    def set_start_time(self):
        self.start_time = time.time()

    def elapsed_time(self):
        return time.time() - self.start_time

    def set_result(self, in_result):
        self.result = in_result

    def get_tpi(self):
        return self.total_time / self.total_images_size


def count_lists_indict(indict):
    val = 0
    for k in indict.keys():
        val = val + len(indict[k])
    return val


def get_label_map(in_dict):
    val = dict()
    for i, x in enumerate(in_dict.keys()):
        val[x] = i
    return val


def print_table(data, header):
    print(tabulate(data, header, tablefmt='psql'))


train_dir = os.path.realpath('train')
test_dir = os.path.realpath('test')
train_dir_dict = get_data_arrays(train_dir)
test_dir_dict = get_data_arrays(test_dir)
input_label_map = get_label_map(train_dir_dict)

models_info = {
    "dlib": 0.6,
    "keras_openface": 0.1,
    "vgg_resnet_50" : 120
}

models_dict = {
    "dlib": get_dlib_emb_array,
    "keras_openface": get_open_emb_array,
    "vgg_resnet_50" : get_vgg_resnet_emb_array
}




