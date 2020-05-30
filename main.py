"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import socket
import json
import cv2

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser

def draw_output(frame, width, height, output, prob_threshold):
    """
    Draw model output on a frame.
    :param frame: image frame
    :param output: output data to be drawed
    :return: person count and frame
    """
    counter = 0
    for box in output[0][0]: # Output shape is 1x1x100x7
        conf = box[2]
        if conf >= prob_threshold:
            xmin = int(box[3] * width)
            ymin = int(box[4] * height)
            xmax = int(box[5] * width)
            ymax = int(box[6] * height)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
            counter += 1
    return frame, counter

def connect_mqtt():
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    return client


def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    curr_req_id = 0
    next_req_id = 1
    num_requests = 2
    single_image_mode = False
    input_file = args.input
    prev_counter = 0
    total_counter = 0
    delay_counter = 0
    max_delay = 2
    start_time = None
    
    # Initialise the class
    infer_network = Network()
    
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold

    ### Load the model through `infer_network` ###
    infer_network.load_model(args.model, num_requests, args.device, args.cpu_extension)
    input_shape = infer_network.get_input_shape()

    ### Handle the input stream ###
    
    if not os.path.isfile(args.input):
        log.error("input doesn't exist: {}".format(args.input))
        sys.ext(1)
    
    if input_file.lower()=="cam":
        input_file = 0
    elif input_file.split(".")[-1].lower() in ['jpg', 'png', 'bmp']:
        single_image_mode = True
                
    cap = cv2.VideoCapture(input_file)
    
    if input_file:
        cap.open(input_file)
        
    # Grab the shape of the input 
    width = int(cap.get(3))
    height = int(cap.get(4))
      
        
    ### Loop until stream is over ###
    while cap.isOpened():
        
        ### Read from the video capture ###
        flag, frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)

        ### Pre-process the image as needed ###
        p_frame = cv2.resize(frame, (input_shape[3], input_shape[2]))
        p_frame = p_frame.transpose((2,0,1))
        p_frame = p_frame.reshape(1, *p_frame.shape)

        ### Start asynchronous inference for specified request ###
        infer_start_time = time.time()

        infer_network.exec_net(curr_req_id, p_frame)
        
        ### Wait for the result ###
        if infer_network.wait(curr_req_id) == 0:
            
            infer_duration = time.time() - infer_start_time
            ### Get the results of the inference request ###
            output = infer_network.get_output(curr_req_id)
            
            frame, curr_counter = draw_output(frame, width, height, output, prob_threshold)
            
            infer_time_text = "Inference : {:.3f}ms".format(infer_duration * 1000)                      
            cv2.putText(frame, infer_time_text, (20, 380), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 1)
            
            if start_time is not None:
                duration_text = "Duration : {}s".format(int(time.time() - start_time))
                cv2.putText(frame, duration_text, (20, 400), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 1)
            
            ### Extract any desired stats from the results ###
            if curr_counter > prev_counter:
                start_time = time.time()
                increment = curr_counter - prev_counter
                total_counter += increment
                client.publish("person", json.dumps({"total": total_counter}))                
            elif curr_counter < prev_counter:
                if delay_counter < max_delay:
                    delay_counter += 1
                    curr_counter = prev_counter
                else:
                    duration = int(time.time() - start_time)
                    client.publish("person/duration", json.dumps({"duration": duration}))
                    delay_counter = 0 
                    start_time = None
                
            client.publish("person", json.dumps({"count": curr_counter}))
            prev_counter = curr_counter

        ### TODO: Send the frame to the FFMPEG server ###
        sys.stdout.buffer.write(frame)  
        sys.stdout.flush()
    
        ### TODO: Write an output image if `single_image_mode` ###
        if single_image_mode:
            cv2.imwrite('output_image.jpg', frame)


def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
