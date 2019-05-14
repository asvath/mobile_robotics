# mobile_robotics
##Goal: 2D object detection of pedestrians, cyclists and cars
##A. Dataset Selection




The nuScenes dataset is a publicly available multimodal dataset by nuTonomy. The data was gathered in Boston and Singapore; two mega cities with busy traffic, thus ensuring a diverse scenario of traffic situations. The initial release of the dataset comprises of 23,772, 1600 x 900, images with 3D annotations of 23 classes. Objects were annotated by considering the full suite of sensors, 6 cameras, 1 Lidar and 5 Radar. Each annotated object was covered by at least one lidar or radar point; hence even objects with low visibility: 0% to 40% visibility were annotated. The annotations were done by expert annotators and numerous validation steps were performed to ensure the quality of the annotations. The diversity and quality of the annotations is why nuScenes was selected for our project.
For the purposes of 2D object detection, we converted the given 3D bounding boxes into 2D bounding boxes. The global coordinates of the 8 corners of the 3D bounding boxes were provided and were converted into camera coordinates via the get_sample_data function provided by nuTonomy. The given functions can be accessed at www.nuscenes.org. We wrote our own function, all_3d_to_2d to convert the camera coordinates to the image coordinates by utilizing the intrinsic camera calibration matrices. The 2D bounding boxes were then extracted by taking the minimum and maximum of the x and y coordinates of the 3D bounding boxes via our extract_bounding_box function. These coordinates form the corners of our resulting 2D bounding boxes. All of our function cane be accessed in "nuscenes extract and write out 2d annotation boxes-revised to truncate bb.ipynb" (the bounding boxes are within the image frame). The figures below shows an example of 3D bounding boxes and the corresponding extracted 2D bounding boxes.

![alt text](https://github.com/asvath/mobile_robotics/blob/master/final%20results/3d.png)
![alt text](https://github.com/asvath/mobile_robotics/blob/master/final%20results/2dbb.png)



We only acquired the 2D bounding boxes for objects whose visibility exceeded 40% and whose center fall within the image boundaries. This is to ensure that the extracted bounding box annotations were similar to that of data only acquired only via cameras. We also combined the 'adult', ‘child’, ‘police officer’ and ‘construction worker’ classes together to form our pedestrian class. The final dataset consists of 20,273 pedestrian annotations, 26,202 car annotations and 1,588 cyclist annotations. This amounts to 48,063 annotations in total.
We generated the train dataset, validation dataset and test dataset by randomly splitting the nuScenes dataset into 70% for training, 15% for validation and 15% for testing. The train dataset consists of 16,640 images, the validation dataset and test dataset consist of 3,566 images respectively.
