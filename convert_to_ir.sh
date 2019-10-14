#!/usr/bin/env bash

OPTIMIZER_DIR=/opt/intel/openvino/deployment_tools/model_optimizer
$OPTIMIZER_DIR/mo_tf.py \
    --input_model="./inference-graphs/frozen_inference_graph.pb" \
    --tensorflow_use_custom_operations_config ssd_v2_support.json \
    --tensorflow_object_detection_api_pipeline_config "./inference-graphs/pipeline.config" \
    --output_dir="IR" \
    --reverse_input_channels \
    --data_type=FP16

