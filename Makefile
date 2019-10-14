SHELL := /bin/bash

TRAIN_DIR = "training"
EVAL_DIR = "eval"
PIPELINE_CONFIG = "pipeline.config"
INFERENCE_GRPAHS_DIR = "inference-graphs"

download_model:
	@ wget -N --continue "http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz"
	@ tar xvf ssd_inception_v2_coco_2018_01_28.tar.gz

init:
	@ if [ ! -f train.py ]; then ln -s models/research/object_detection/legacy/train.py; fi
	@ if [ ! -f eval.py ]; then ln -s models/research/object_detection/legacy/eval.py; fi
	@ if [ ! -f export_inference_graph.py ]; then ln -s models/research/object_detection/export_inference_graph.py; fi

compile_protobuf:
	@ cd models/research && protoc object_detection/protos/*.proto --python_out=.

xml_to_csv:
	@ ./xml_to_csv.py

generate_tfrecord:
	@ source ./env.sh && ./generate_tfrecord.py


.PHONY: train
train:
	@ source ./env.sh && python3 train.py \
		--logtostderr \
		--train_dir=$(TRAIN_DIR) \
		--pipeline_config_path=$(PIPELINE_CONFIG)
log_train:
	@ tensorboard --logdir $(TRAIN_DIR)

.PHONY: eval
eval:
	@ source ./env.sh && python3 eval.py --logtostderr \
        --checkpoint_dir=$(TRAIN_DIR) \
        --eval_dir=$(EVAL_DIR)\
        --pipeline_config_path=$(PIPELINE_CONFIG)


log_eval:
	@ tensorboard --logdir $(EVAL_DIR)

.PHONY: export
export:
	@ rm -rf $(INFERENCE_GRPAHS_DIR)
	@ source ./env.sh && python3 export_inference_graph.py \
		--input_type image_tensor \
		--pipeline_config_path $(PIPELINE_CONFIG) \
		--trained_checkpoint_prefix $$(ls training/model.ckpt-*.meta | tail -n 1 | sed -e 's/.meta//') \
		--output_directory $(INFERENCE_GRPAHS_DIR)

model_optimize:
	@ /opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py \
		--input_model="$(INFERENCE_GRPAHS_DIR)/frozen_inference_graph.pb" \
		--tensorflow_use_custom_operations_config ssd_v2_support.json \
		--tensorflow_object_detection_api_pipeline_config "$(INFERENCE_GRPAHS_DIR)/pipeline.config" \
		--output_dir="IR" \
		--reverse_input_channels \
		--data_type=FP16

detect_sample:
	@ ./detect.py --gui

detect_cam:
	@ ./detect.py --gui --input-type 'camera'

demo:
	@ ./detect_realsense.py --gui

detect_rs:
	@ ./detect_realsense.py --gui --show-depth

clean:
	@ rm -rvf $(TRAIN_DIR) $(EVAL_DIR)
