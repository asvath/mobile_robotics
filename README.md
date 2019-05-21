# Mobile Robotics Course Project
## Goal: 2D object detection of pedestrians, cyclists and cars
## A. Dataset Selection




The nuScenes dataset is a publicly available multimodal dataset by nuTonomy. The data was gathered in Boston and Singapore; two mega cities with busy traffic, thus ensuring a diverse scenario of traffic situations. The initial release of the dataset comprises of 23,772, 1600 x 900, images with 3D annotations of 23 classes. Objects were annotated by considering the full suite of sensors, 6 cameras, 1 Lidar and 5 Radar. Each annotated object was covered by at least one lidar or radar point; hence even objects with low visibility: 0% to 40% visibility were annotated. The annotations were done by expert annotators and numerous validation steps were performed to ensure the quality of the annotations. The diversity and quality of the annotations is why nuScenes was selected for our project.
For the purposes of 2D object detection, we converted the given 3D bounding boxes into 2D bounding boxes. The global coordinates of the 8 corners of the 3D bounding boxes were provided and were converted into camera coordinates via the get_sample_data function provided by nuTonomy. The given functions can be accessed at www.nuscenes.org. We wrote our own function, all_3d_to_2d to convert the camera coordinates to the image coordinates by utilizing the intrinsic camera calibration matrices. The 2D bounding boxes were then extracted by taking the minimum and maximum of the x and y coordinates of the 3D bounding boxes via our extract_bounding_box function. These coordinates form the corners of our resulting 2D bounding boxes. All of our function cane be accessed in "nuscenes extract and write out 2d annotation boxes-revised to truncate bb.ipynb" (the bounding boxes are within the image frame). The figures below shows an example of 3D bounding boxes and the corresponding extracted 2D bounding boxes.

![alt text](https://github.com/asvath/mobile_robotics/blob/master/final%20results/3d.png)
![alt text](https://github.com/asvath/mobile_robotics/blob/master/final%20results/2dbb.png)



We only acquired the 2D bounding boxes for objects whose visibility exceeded 40% and whose center fall within the image boundaries. This is to ensure that the extracted bounding box annotations were similar to that of data only acquired only via cameras. We also combined the 'adult', ‘child’, ‘police officer’ and ‘construction worker’ classes together to form our pedestrian class. The final dataset consists of 20,273 pedestrian annotations, 26,202 car annotations and 1,588 cyclist annotations. This amounts to 48,063 annotations in total.
We generated the train dataset, validation dataset and test dataset by randomly splitting the nuScenes dataset into 70% for training, 15% for validation and 15% for testing. The train dataset consists of 16,640 images, the validation dataset and test dataset consist of 3,566 images respectively. (Code to split dataset: train validation test -Copy2.ipynb)


## B. Tiny YOLO v3

Tiny YOLO Version 3
The Tiny You Only Look Once (Tiny YOLO) algorithm utilizes features learned by a deep convolution neural network (CNN) to detect objects. It is a fully convolutional network(FCN), thus making it invariant to the size of the input image. The input image is firstly resized to the network resolution. It is then divided into S x S grid cells, and each of these grid cells is “responsible” for predicting objects whose center falls within it. In practice, a grid cell might detect an object even though the center of the object does not fall within it. This leads to multiple detections of the same object by different grid cells. Non-max suppression cleans up the detections and ensures that each object is detected once. This is done by selecting the bounding box with the highest object detection probability as the output bounding box and suppressing bounding boxes that have a high IoU with the output bounding
box. In addition, predefined shapes called anchor boxes enable the detection of multiple objects whose centers fall within the same grid cell. Each object is associated with the anchor box with the highest IoU. The K-means clustering algorithm isused to determine the height and width of the anchor boxes. Each bounding box prediction is a vector. The components of the vectors are the following: confidence score of object detection, x,y coordinates of the center of the bounding box,the height and width h,w of the bounding box and C class probabilities. If there are A anchor boxes, the vector is A(5+C) in dimension.

The figure below shows the results from the K-means algorithm (k means clustering.ipynb)
<img src="https://github.com/asvath/mobile_robotics/blob/master/final%20results/IOU_clusters.png" width="500" height="500">

We chose to use 6 anchor boxes as the average IOU was reasonable value of approx 60% and also the default number of anchor boxes used by Tiny YOLO v3 is 6. Increasing the number of anchor boxes will increase the number of parameters used.


## C. Hardware
We trained the Tiny YOLO v3 model on the train dataset consisting of 16,640 images. The base model and the initial pre-trained weights were acquired via the official YOLO website (https://pjreddie.com/darknet/yolo/). In particular, we utilized yolov3-tiny_obj.cfg as our base model and yolov3-tiny.conv.15 as our initial weights. We trained 4 different versions of the base model by tuning the following
hyperparameters: resolution and subdivision. In addition, we changed the default anchor box values to that generated by the K-means clustering algorithm. The resolution and subdivision of the base model are 416 x 416 and 8 respectively. The model resizes any input data to the resolution value. The subdivision value refers to number of mini-batches that is sent to the GPU for processing. We trained all our models for 12,000 iterations.

## D. Results
## 1. Validation of the trained models
The 4 different versions of the Tiny YOLO v3 model were trained by tuning the following hyperparameters: resolution and subdivision. The trained models were then validated using the validation dataset. The results from the validation is shown below.
### TABLE VALIDATION OF TRAINED MODELS
| Resolution  | Batch | Subdivision | Highest mAP(%) at IoU Threshold (50%) |
| ------------- | ------------- | ------------- | ------------- |
| 416  | 64  | 2  | 48.32  |
| 416  | 64  | 8  | 48.51  |
| 832  | 64  | 8  | 61.76  |
| 832  | 64  | 32  | 61.46  |
 

The mean average precision value is the area under the
precision and recall curve. It is a metric that is used to
compare the performance of various models. The model with
the input resolution of 832 and subdivision value of 8 was
selected as the best performing model as it has the highest
mAP score of 61.76% at the IoU threshold of 50%. The
Precision and Recall curve of the selected model is shown at
different confidence score thresholds the figure below. 
<img src="https://github.com/asvath/mobile_robotics/blob/master/final%20results/nuval_50%20(1).png" width="500" height="500">

The loss during the training of the model with resolution 832 and subdivision 8, declined rapidly before stagnating at 1.2 as shown in the figure below. Further training will probably not improve the model’s performance. The mAP value reached a maximum of 61.76% at iteration 11, 000. Hence the weights from this iteration was used as our final weights. The mAP score declined after iteration 11,000. The decline could be due to overfitting. This could be verified by training the model for several more iterations and determining if the declining mAP trend continues. This model had the highest mAP score out of all trained models; and was thus chosen as the model for further analysis.
<img src="https://github.com/asvath/mobile_robotics/blob/master/final%20results/plot_832_64_8%20(1).png" width="500" height="500">
### Selection of Confidence Score Threshold
The default confidence score threshold of Tiny YOLO v3 during detection is 25%. At this threshold, the precision, recall and F1-scores are 0.81, 0.57 and 0.67 respectively.
The high precision of 0.81, indicates low false positives and the low recall value of 0.57 indicates high false negatives.The figure below shows an instance of detection at the threshold of 25%. A pedestrian at the crosswalk was not detected despite their proximity to the car. The pedestrian was thus a false negative. Scenarios such as this must be avoided as it could lead to dangerous driving by the autonomous vehicle. Hence a confidence score threshold needs to be selected with a high recall value.

![alt text](https://github.com/asvath/mobile_robotics/blob/master/final%20results/ped_25.png)
![alt text](https://github.com/asvath/mobile_robotics/blob/master/final%20results/ped_10.png)

Fig. 4 in the Results Section A. shows the precision recall curve at various confidence score thresholds. Fig. 7 shows the F1 scores vs. confidence score threshold, where a high F1 indicates a high precision and high recall value. This occurs at threshold 20% with F1 score of 0.68, precision of 0.78 and recall of 0.60. Fig. 8 shows the recall vs confidence score threshold. The highest recall value of 0.73 occurs at a threshold of 5%, however the F1-score is 0.62 and the precision is 0.54. While having a high recall is paramount for purposes of autonomous driving, we also want to ensure that the tradeoff between precision and recall is low, as too many false positives could potentially lead to situations where the autonomous vehicle is unable to function. Hence, we chose a confidence threshold of 10%, where the precision and recall are high and also comparable in values; in addition, the F1 score of 0.67 is close to the highest F1 score of 0.68. At 10% confidence the precision is 0.66, recall is 0.68, and F1-score is 0.67 as mentioned in Results Section A
Lowering the threshold to 10% allowed the pedestrian to be detected as highlighted in Fig. 6
The average precision (ap) values for the classes being detected were also mentioned in Results Section A. The car class has the highest ap value of 76.17%. This could be due to the symmetric nature of cars, thus enabling the model to learn the features better. In addition, we had over 20, 000 car annotations. Despite having a comparable number of pedestrian annotations, the ap for pedestrians was much lower. This could be due to the diversity in pedestrian features. We also found that most of the pedestrians in our training set were in the background as opposed to the foreground. This is shown in Fig. 9
The Confidence score threshold of 10% was selected for
detection. At this threshold, the precision is 0.66, recall is
0.68, and F-1 score is 0.67 respectively.
The average precision values for pedestrians, cars and
cyclists are 54.78%, 76.17% and 54.33% respectively.

##YouTube Video
[![Alt text](https://github.com/asvath/mobile_robotics/blob/master/final%20results/Capture.JPG)](https://www.youtube.com/watch?v=hmpNFlYn0yo&feature=youtu.be&fbclid=IwAR167HZ5qLn4Co63pQxlnsFPsgUeM3Pq84B0FmO7yLNVyffIRLjVCSNJv9w)
