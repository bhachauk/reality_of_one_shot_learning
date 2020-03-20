# Evaluating One shot learning for the models :

### Training Detail

| Train classes   |   size |
|-----------------|--------|
| ben_afflek      |     13 |
| elton_john      |     15 |
| jerry_seinfeld  |     21 |
| madonna         |     18 |
| mindy_kaling    |     21 |


### Testing Details

| Test classes   |   size |
|----------------|--------|
| ben_afflek     |      1 |
| elton_john     |      1 |
| jerry_seinfeld |      1 |
| madonna        |      1 |
| mindy_kaling   |      1 |

### Models details

Model name :  dlib
Model name :  keras_openface
Model name :  vgg_resnet_50
collected all evaluation metrics size : 3

### Model Results for : dlib

| class          |   gar |      avg |      min |      max |   false_positive |
|----------------|-------|----------|----------|----------|------------------|
| ben_afflek     |     1 | 0.472255 | 0.472255 | 0.472255 |                0 |
| elton_john     |     1 | 0.522505 | 0.522505 | 0.522505 |                0 |
| jerry_seinfeld |     1 | 0.393405 | 0.393405 | 0.393405 |                0 |
| madonna        |     1 | 0.501737 | 0.501737 | 0.501737 |                0 |
| mindy_kaling   |     1 | 0.411828 | 0.411828 | 0.411828 |                0 |


### Model Results for : keras_openface


| class          |   gar |       avg |       min |       max |   false_positive |
|----------------|-------|-----------|-----------|-----------|------------------|
| ben_afflek     |     0 | 0.0532961 | 0.0532961 | 0.0532961 |                1 |
| elton_john     |     0 | 0.0750824 | 0.0750824 | 0.0750824 |                1 |
| jerry_seinfeld |     0 | 0.0517428 | 0.0517428 | 0.0517428 |                1 |
| madonna        |     1 | 0.0416994 | 0.0416994 | 0.0416994 |                0 |
| mindy_kaling   |     1 | 0.0728799 | 0.0728799 | 0.0728799 |                0 |


### Model Results for : vgg_resnet_50


| class          |   gar |      avg |      min |      max |   false_positive |
|----------------|-------|----------|----------|----------|------------------|
| ben_afflek     |     1 | 101.858  | 101.858  | 101.858  |                0 |
| elton_john     |     1 | 113.426  | 113.426  | 113.426  |                0 |
| jerry_seinfeld |     1 |  86.6292 |  86.6292 |  86.6292 |                0 |
| madonna        |     1 | 110.299  | 110.299  | 110.299  |                0 |
| mindy_kaling   |     1 | 103.847  | 103.847  | 103.847  |                0 |


Final Results of train --> test:

| Model          |   Feature Size |   Training Time in (seconds) |   Images Size |   GAR |   FRR |   FAR |
|----------------|----------------|------------------------------|---------------|-------|-------|-------|
| dlib           |            128 |                      3.38861 |            84 |   1   |    -1 |   0   |
| keras_openface |            128 |                      8.63753 |            88 |   0.4 |    -1 |   0.6 |
| vgg_resnet_50  |           2048 |                     27.3885  |            88 |   1   |    -1 |   0   |

