# People Counter App

A people counter app is an application to count number of people in a stored video, streaming video, or single image. The application is utilizing Intel Open Vino toolkit to detect the people.


## Explaining Custom Layers

Custom layers are layers which are not actually included in the known layers list of Open Vino toolkit.

If you want to implement the custom layers for your pretrained model, you need to add extensions to both the Model Optimizer and Inference Engine. Note that different supported framework will has some different steps for registering the custom layers. You can read here for more details: https://github.com/david-drew/OpenVINO-Custom-Layers

One of the reason to implement the custom layers is because you really need to convert a specific model to IR, but the model cannot be converted without converting some of the layers to the custom layers. The layers which previously cannot be converted can be a really important layer for you, and the layers cannot be ignored because it might be the main part of your research or works.


## Comparing Model Performance

Following are the performance of three tensorflow model before and after conversion:

### Size

| |Before Conversion|After Conversion|
|-|-|-|
|ssd_inception_v2_coco|98MB|96MB|
|ssdlite_mobilenet_v2_coco|19MB|18MB|
|ssd_mobilenet_v2_coco|67MB|65MB|

### Inference Time

| |Before Conversion|After Conversion|
|-|-|-|
|ssd_inception_v2_coco|157ms|166ms|
|ssdlite_mobilenet_v2_coco|35ms|32ms|
|ssd_mobilenet_v2_coco|83ms|70ms|

### Accuracy

All three models failed to correctly detect the people in the video. So I decided to use Open Vino Pretrained model to get a better accuracy, which is **person-detection-retail-0013**. The person-detection-retail-0013 performed quite good with 45ms inference time and consistently accurate. The size of the model is only 2MB.

Compare to cloud services, one of the advantages in deploying model at edge is it doesn't really depend on network connection. While cloud services could be a problem when the network connection is slow, edge doesn't have this problem because the inference can be done at the edge. 


## Assess Model Use Cases

### 1. Covid-19 Prevention
By calculating the number of people in a room, we can avoid "too many people" in the room by giving an alert if the number of people reach the maximum allowable capacity. Similarly, this also can be applied in an elevator. This use case will be useful to avoid the crowded room that can increase the spread of the virus.
### 2. Queue Monitoring
By collecting data from the queuing people, we will know that the "serving" duration for each person in the queue. Using the data, we can decide to add additional staff or not. Besides, we also can detect how many people actually leave the queue because they are bored for waiting. This use case is useful so we can optimize order of our customer, and of course we will get more revenue.
### 3. Security
When we leave our house and lock the door, there shouldn't be anyone detected inside the house. But if there is someone detected, it can be a signed of some has broken into our house. This will be useful so we can act quickly before the robber steal anything inside.


## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a
deployed edge model. The potential effects of each of these are as follows:

- Lighting: a dark room may cause a serious problem to the performance of a model. The model may failing to detect anyone in a frame. Depends on the use case, if a really dark room is one of the requirements, using infrared cam can be a solution in that case. But of course the model need to be trained on the infrared video. 

- Model Accuracy: To get a better speed, we will lose some accuracies because of the conversion. So we need to make sure the accuracy of the model is still within "usable" range after the conversion, which is not too low, and still can make a good detection. So basically it's normal to get a lower accuracy after converting the model by using the OpenVINO toolkit.

- Camera focal length/image size: different resolution of an image can be a problem for a model to detect a person correctly. Training the model with a lot of possible range of resolution or image size can be one of the solutions to handle this issue. Augmentation can be applied to generate the training data.

## Model Research

In investigating potential people counter models, I tried each of the following three models:

- Model 1: [ssd_inception_v2_coco]
  - [http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz]
  - I converted the model to an Intermediate Representation with the following arguments...
  ```python /opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json --reverse_input_channel```
  - The model was insufficient for the app because the model often failed to detect the people. Besides, the inference of the model was too slow.
  - I tried to improve the model for the app by reducing the probability threshold in hope that the model could detect the people, but unfortunately it didn't work. Using a delay of frame will also not be useful because the second person in the video couldn't be detected for a long period of time.
  
- Model 2: [ssdlite_mobilenet_v2_coco]
  - [http://download.tensorflow.org/models/object_detection/ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz]
  - I converted the model to an Intermediate Representation with the following arguments...
    ```python /opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json --reverse_input_channel```
  - Even though the inference of this model was quite fast, but it is still insufficient for the app because the model often failed to detect people.
  - I also tried to improve the model by reducing the probability threshold, but unfortunately it didn't work. Using a delay of frame will also not be useful because the second person in the video couldn't be detected for a long period of time.

- Model 3: [ssd_mobilenet_v2_coco]
  - [http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz]
  - I converted the model to an Intermediate Representation with the following arguments...
    ```python /opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json --reverse_input_channel```
  - The model was insufficient for the app because it was not fast enough and most importanly it was not accurate.
  - I also tried to improve the model by reducing the probability threshold, but unfortunately it didn't work. Using a delay of frame will also not be useful because the second person in the video couldn't be detected for a long period of time.
