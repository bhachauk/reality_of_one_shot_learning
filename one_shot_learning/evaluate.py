from one_shot_learning import util
import os, time
logging = util.get_logging()


def print_this_dict(indict, name):
    result = list()
    for key in indict.keys():
        result.append([key, len(indict[key])])
    util.print_table(result, [name, 'size'])


def evaluate(model_name, train_embed_dict, test_embed_dict):
    global logging
    result = dict()
    logging.info("Initing Evaluation for model type : {}".format(model_name))
    for class_name in test_embed_dict.keys():
        result[class_name] = dict()
        distances = list()
        count = 0
        actual_count = 0
        cpc = 0
        for test_data_row in test_embed_dict[class_name]:
            actual_count = actual_count + 1
            distance, predicted_class = util.minimum_distance_classifier(train_embed_dict, test_data_row)
            if distance <= util.models_info[model_name]:
                count = count + 1
                if class_name == predicted_class:
                    cpc = cpc + 1
                distances.append(distance)
            logging.info("For Class : {}, Count : {}, Distance : {}".format(class_name, count, distance))
        accuracy = cpc / actual_count
        false_prediction = (count - cpc) / actual_count
        avg_distance = -1
        max_distance = -1
        min_distance = -1
        if count != 0:
            avg_distance = sum(distances) / count
            max_distance = max(distances)
            min_distance = min(distances)
        logging.info("class: {}, accuracy: {:.3f}, distances: {:.3f}-{:.3f}-{:.3f}, total: {}".format(class_name,
                                                                                                      accuracy,
                                                                                                      min_distance,
                                                                                                      avg_distance,
                                                                                                      max_distance,
                                                                                                      actual_count))
        if false_prediction:
            logging.info(
                "Warning. False prediction found in class {} : {}".format(class_name, str(false_prediction)))
        result[class_name]['accuracy'] = accuracy
        result[class_name]['avg_distance'] = avg_distance
        result[class_name]['min_distance'] = min_distance
        result[class_name]['max_distance'] = max_distance
        result[class_name]['false_prediction_rate'] = false_prediction
    return result


def get_val_div_correct(f_count, t_count):
    ret_val = 0
    if t_count == 0:
        ret_val = -1
    else:
        ret_val = f_count / t_count
    return ret_val


def Sort(sub_li):
    return sorted(sub_li, key=lambda x: x[0])


def show_result(final_metrics, trained_classes):
    final_result = []
    for metric in final_metrics:
        print("Model Results for : {}".format(metric.name))
        data = []
        gar = 0
        far = 0
        frr_count = 0
        rr_count = 0
        gar_count = 0
        for k in metric.result.keys():
            row = list(metric.result[k].values())
            row.insert(0, k)
            data.append(row)
            if k in trained_classes:
                gar_count = gar_count + 1
                gar = gar + row[1]
                if row[1] == 0 and row[-1] == 0:
                    frr_count = frr_count + 1
            if row[1] == 0 and row[-1] == 0:
                rr_count = rr_count + 1
            far = far + row[-1]
        frr_val = get_val_div_correct(frr_count, rr_count)
        gar_val = get_val_div_correct(gar, gar_count)
        far_val = get_val_div_correct(far , (len(data)-rr_count))
        util.print_table(Sort(data), ['class', 'gar', 'avg', 'min', 'max', 'false_positive'])
        final_result.append([metric.name, metric.feature_size, metric.total_time, metric.total_images_size, gar_val, frr_val,
                             far_val])
    print("Final Results of {} --> {}:".format(os.path.basename(util.train_dir), os.path.basename(util.test_dir)))
    util.print_table(final_result, ['Model', 'Feature Size', 'Training Time in (seconds)', 'Images Size', 'GAR', 'FRR', 'FAR'])


def main():
    final_metrics = list()
    print("Evaluating One shot learning for the models :")
    print_this_dict(util.train_dir_dict, "Train classes")
    print_this_dict(util.test_dir_dict, "Test classes")
    for model_name in util.models_info.keys():
        print("Model name : ", model_name)
        model_metric = util.ModelMetrics(model_name)
        start = time.time()
        train_embed_dict = util.get_emb_arrays(util.train_dir_dict, model_name)
        model_metric.total_time = time.time() - start
        model_metric.total_images_size = util.count_lists_indict(train_embed_dict)
        model_metric.feature_size = util.current_feature_size
        test_embed_dict = util.get_emb_arrays(util.test_dir_dict, model_name)
        result = evaluate(model_name, train_embed_dict, test_embed_dict)
        model_metric.result = result
        final_metrics.append(model_metric)
    print("collected all evaluation metrics size : {}".format(len(final_metrics)))
    show_result(final_metrics, util.train_dir_dict.keys())


if __name__ == "__main__":
    main()
