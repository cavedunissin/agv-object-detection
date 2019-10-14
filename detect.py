#!/usr/bin/env python3

import argparse
import cv2
import logging
import sys
from openvino.inference_engine import IENetwork, IEPlugin
import time
import numpy as np
import json

parser = argparse.ArgumentParser()
parser.add_argument(
    '--model',
    default='IR/frozen_inference_graph.xml',
    help='trained model topology (.xml)'
)
parser.add_argument(
    '--weights',
    default='IR/frozen_inference_graph.bin',
    help='trained model weights (.bin)'
)
parser.add_argument(
    '--input-type',
    default='file',
    choices=['file', 'camera'],
    help='video from file or camera'
)
parser.add_argument(
    '--input',
    default='sample.mp4',
    help='video input'
)
parser.add_argument(
    '--device',
    default='MYRIAD',
    choices=['MYRIAD', 'GPU'],
    help='Computing device'
)
parser.add_argument(
    '--labels',
    default='labels/label_map.json',
    help='labels mapping file(json)'
)
parser.add_argument(
    '--threshold',
    default=0.5,
    type=float,
    help='probability threshold of predictions'
)
parser.add_argument(
    '--output',
    default='./output.mp4',
    help='save prediction into mp4 file'
)
parser.add_argument(
    '--gui',
    default=False,
    action='store_true',
    help='toggle GUI'
)
args = parser.parse_args()

# Setup video source
if args.input_type == 'file':
    video_capture = cv2.VideoCapture(args.input)
elif args.input_type == 'camera':
    video_capture = cv2.VideoCapture(-1)

# Prepare labels map
with open(args.labels) as f:
    labels_map = json.load(f)

# switch keys and values
labels_map = dict((y, x) for (x, y)in labels_map.items())

# setup logger
logging.basicConfig(
    format='[ %(levelname)s ] %(message)s',
    level=logging.INFO,
    stream=sys.stdout
)

# Load model into inference engine
plugin = IEPlugin(args.device)
net = IENetwork(model=args.model, weights=args.weights)
input_blob = next(iter(net.inputs))
out_blob = next(iter(net.outputs))
exec_net = plugin.load(network=net, num_requests=2)

# Input shape: [n_samples, n_channels, height, width]
input_shape = net.inputs[input_blob].shape
input_height = input_shape[2]
input_width = input_shape[3]
assert input_shape[0] == 1


def preprocess(frame):
    frame = cv2.resize(frame, (input_width, input_height))
    frame = frame.astype(np.float32)
    frame = np.moveaxis(frame, -1, 0)  # change layout from HWC to CHW
    batch = np.expand_dims(frame, 0)  # convert into batch data with size 1
    return batch


video_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

if args.output is not None:
    video_writer = cv2.VideoWriter(
        args.output,
        fourcc=cv2.VideoWriter_fourcc(*'mp4v'),
        fps=10,
        frameSize=(video_width, video_height)
    )


def plot_bbox(
    frame,
    bbox,
    label,
    prob,
    font=cv2.FONT_HERSHEY_DUPLEX,
    font_size=0.8,
    font_thickness=1,
    text_color=(0, 0, 255)
):
    # box_color = (0, max(255 - class_id * 5, 0), 0)
    box_color = (0, 255, 0)
    text = '%s: %.1f%%' % (label, round(prob * 100, 1))
    x_min, y_min, x_max, y_max = bbox
    text_size = cv2.getTextSize(
        text,
        fontFace=font,
        fontScale=font_size,
        thickness=font_thickness
    )
    cv2.rectangle(
        frame,
        pt1=(x_min, y_min),
        pt2=(x_max, y_max),
        color=box_color,
        thickness=2
    )
    cv2.rectangle(
        frame,
        pt1=(x_min, y_min),
        pt2=(x_min+text_size[0][0], y_min-text_size[0][1] - 7),
        color=box_color,
        thickness=cv2.FILLED
    )
    cv2.putText(
        frame,
        text=text,
        org=(x_min, y_min - 7),
        fontFace=font,
        fontScale=font_size,
        color=text_color,
        thickness=1
    )


logging.info('Start inference ...')
logging.info(
    'Press Esc/<Ctrl+C> to terminate '
    'or Tab to switch async/sync mode.'
)
async_mode = True
cur_request_id = 0
next_request_id = 1


while video_capture.isOpened():
    try:
        retrieved, frame = video_capture.read()
        if not retrieved:
            break

        timer = time.time()
        exec_net.start_async(
            request_id=next_request_id if async_mode else cur_request_id,
            inputs={input_blob: preprocess(frame)}
        )
        inferece_time = time.time() - timer

        if exec_net.requests[cur_request_id].wait(-1) == 0:
            result = exec_net.requests[cur_request_id].outputs[out_blob]
            for obj in result[0][0]:
                prob = obj[2]

                if prob > args.threshold:
                    class_id = int(obj[1])
                    bbox = (
                        int(obj[3] * video_width),
                        int(obj[4] * video_height),
                        int(obj[5] * video_width),
                        int(obj[6] * video_height),
                    )

                    # Exclude object of too large size
                    if (obj[5] - obj[3]) < 0.5 and (obj[6] - obj[4]) < 0.5:
                        plot_bbox(
                            frame,
                            bbox,
                            label=labels_map[class_id],
                            prob=prob,
                        )
        # print inference time message
        if async_mode:
            inf_time_msg = "Inference time: N\\A for async mode"
        else:
            inf_time_msg = "Inference time: %.3f ms" % (inferece_time * 1000)
        cv2.putText(
            frame,
            inf_time_msg,
            (10, 30),
            cv2.FONT_HERSHEY_DUPLEX,
            1,
            (0, 255, 0),
            1
        )

        if args.gui:
            cv2.imshow('Detection Results', frame)
        if async_mode:
            cur_request_id, next_request_id = next_request_id, cur_request_id
        if args.output is not None:
            video_writer.write(frame)

        key = cv2.waitKey(1)
        if key == 27:
            break
        if (9 == key):
            async_mode = not async_mode
            logging.info(
                'Switched to %s mode'
                % ('async' if async_mode else 'sync')
            )
            time.sleep(0.1)

    except KeyboardInterrupt:
        break

video_capture.release()
if args.output is not None:
    video_writer.release()
cv2.destroyAllWindows()
