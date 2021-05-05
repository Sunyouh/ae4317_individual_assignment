# ae4317_individual_assignment

# Environment
Codes are tested on Ubuntu 20.04 LTS
Python 3.8.5
numpy 1.19.5
pillow 7.0.0
Tensorflow 2.4.1 (with CUDA 11.3, CuDNN 8.2)
Tensorflow object detection API (https://github.com/tensorflow/models/tree/master/research)

# Dataset
WashingtonOBRace
Total 308 images
train:eval = 218:90 (~= 7:3)

# tf_model
SSD-MobileNet trained on WashingtonOBRace dataset
AP 0.971(IoU>0.5), 0.888(IoU>0.75)
Necessary parameters are defined in pipeline.config


# To make tfrecord:
$ python generate_tfrecord.py \
    --dataset_path=${DATASET_PATH} \
    --output_path=${WHERE_TO_CREATE_TFRECORD}


# To run training:
Requires TF2 and TF Object detection API
$ PIPELINE_CONFIG_PATH={YOUR CONFIG FILE PATH}   (use pipeline.config in tf_model/saved_model)
$ MODEL_DIR={YOUR MODEL PATH}
$ cd ~/tensorflow/models/research 
$ python object_detection/model_main_tf2.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --model_dir=${MODEL_DIR} \
    --alsologtostderr \
    --eval_on_train_data=True


# To run inference
Requires TF2
$ python run_inference.py --img_path=${PATH_TO_EVAL_IMGS}
