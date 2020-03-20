from keras.models import load_model
from one_shot_learning import util, train_siamese
import os
import time
import numpy as np
import os


def get_path_dict(indir):
    return util.get_data_arrays(indir)


def siamese_predict(train_embed_dict, test_image_path):
    predict_result = dict()
    for train_class in train_embed_dict.keys():
        temp_list = list()
        for train_image_path in train_embed_dict[train_class]:
            train_image = np.expand_dims(util.get_vgg_read_image(train_image_path), axis=0)
            test_image = np.expand_dims(util.get_vgg_read_image(test_image_path), axis=0)
            predicted = siamese_model.predict([train_image, test_image])[0][0]
            test_class = test_image_path.split(os.path.sep)[-2]
            print("Train image from class : {}, test image : {}, predicted val: {}".format(train_class, test_class, predicted))
            temp_list.append(predicted)
        predict_result[train_class] = float(sum(temp_list) / len(temp_list))
    pc = min(predict_result, key=predict_result.get)
    return pc, predict_result[pc]


def evaluate(train_dir, test_dir):
    train_paths_dict, test_paths_dict = get_path_dict(train_dir), get_path_dict(test_dir)
    result = dict()
    for class_name in test_paths_dict.keys():
        result[class_name] = dict()
        actual_count = 0
        count = 0
        cpc = 0
        for test_image_path in test_paths_dict[class_name]:
            actual_count = actual_count + 1
            predicted_class, val = siamese_predict(train_paths_dict, test_image_path)
            if val < 100:
                count = count + 1
                if class_name == predicted_class:
                    cpc = cpc + 1
                print("For Class : {}, Count : {}, predicted as : {}".format(class_name, actual_count, predicted_class))
        accuracy = cpc / actual_count
        false_prediction = (count - cpc) / actual_count
        print("class: {}, accuracy: {:.3f}, total: {}".format(class_name, accuracy, actual_count))
        if false_prediction:
            print("Warning. False prediction found in class {} : {}".format(class_name, str(false_prediction)))
        result[class_name]['accuracy'] = accuracy
        result[class_name]['false_prediction_rate'] = false_prediction
    return result


def get_result_details(result_dict):
    header = ['class_name', 'accuracy', 'False Prediction Rate']
    data = list()
    for class_name in result_dict.keys():
        data.append([class_name, result_dict[class_name]['accuracy'], result_dict[class_name]['false_prediction_rate']])
    return data, header


train_dir = util.train_dir
test_dir = util.test_dir
print("Evaluating Siamese trained : {}, {}".format(train_dir, test_dir))
siamese_data = train_siamese.get_siamese_data(train_dir)
print("Trained observations : {}".format(len(siamese_data)))
start = time.time()
siamese_model = load_model(os.path.abspath("siamese_resnet_model.h5"))
print("Model loaded in : {} seconds.".format(time.time() - start))
result = evaluate(train_dir, test_dir)
data, header = get_result_details(result)
util.print_table(data, header)
