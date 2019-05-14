# CarND-Capstone

This is the project repo for the final project of the Udacity Self-Driving Car Nanodegree: Programming a Real Self-Driving Car.

## Individual Submission

This is an individual submission for the project by:

| Name | Email
|---------------- | ---------------------|
| Pouyan Azari | science(at)pouyanazari.com |

## Workflow

The project has 3 main parts:

- The waypoints of the car are given and with help of PID controller and low pass the car is kept on the waypoints.
- There is a classification algorithm that finds the traffic light color.
- Using the classification car stops at red light and starts again on green light.

## Tensorflow

The Hardware specs for the Tensorflow training:

|OS| CPU | RAM | GPU
|---|---------|------- | ---------------------|
|Linux| Intel Xeon E5 3.4GHZ | 32 GB | Nvidia 750 TI (2GB Memory) |

To have the tensorflow classify the traffic lights the dataset that is created kindly by 'coldKnight' are used [link](https://github.com/coldKnight/TrafficLight_Detection-TensorFlowAPI).

The following models were tried:

- Faster-RCNN
- Inception SSD
- MobileNet SSD

### Faster-RCNN

This model needed a very high (11GB GPU Memory) end graphic card that did not run on my specs.

### Inception SSD

This model worked with some tweaks (like lowering the batch size from 24 to 2 and making the images smaller). On the Virtual Machine it was still slow and training with this model was also very slow. It is more accurate than the MobileNet.

### MobileNet SSD

This was the suggested model from most of the people to use on low end machine. The batch size was still to high, so I lowered the number again to 2. This model was the only model that also gave acceptable latency on the VM. It is less accurate than the Faster-RCNN or MobileNet SSD, but the results are still acceptable for the simulation.

#### How Tensorflow training works

This is the description of how the tensorflow model was built.

- Install the pre-requirements

```bash
sudo apt-get install protobuf-compiler
sudo pip install pillow
sudo pip install lxml
sudo pip install jupyter
sudo pip install matplotlib
```

- The tensorflow model library should be downloaded:

```bash
git clone https://github.com/tensorflow/models.git
cd tensorflow/models/research
```

- Configure the tensorflow model

```bash
protoc object_detection/protos/*.proto --python_out=.
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
```

- Then the coco-zoo files for each of the pre-trained models should be fetched.

```bash
wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz
wget http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz
wget http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_2018_01_28.tar.gz
```

- Extract the models

```bash
tar -xvzf ssd_mobilenet_v1_coco_2018_01_28.tar.gz
tar -xvzf ssd_inception_v2_coco_2018_01_28.tar.gz
tar -xvzf faster_rcnn_resnet101_coco_2018_01_28.tar.gz
```

- The classified images were downloaded from [link](https://github.com/coldKnight/TrafficLight_Detection-TensorFlowAPI). Specifically from here: [Google Drive Link](https://drive.google.com/file/d/0B-Eiyn-CUQtxdUZWMkFfQzdObUE/view?usp=sharing) and save them inside the data folder.

- Create tf records from the images. Tensorflow needs this records for the training:

```bash
python object_detection/dataset_tools/create_pascal_tf_record.py --data_dir=data/sim_training_data/sim_data_capture --output_path=sim_data.record
```

- use the configuration in `tensorflow_config` folder, and change to use the coco files

```bash
cp tensorflow_config tensorflow/model/research/config
```

- Run the training (The same done for each model)

```bash
python object_detection/train.py --pipeline_config_path=config/ssd_mobilenet_v1.config --train_dir=data/sim_training_data/sim_data_capture
```

- Freeze the result of training so it the created graph can be used by the ROS classifier.

```bash
python object_detection/export_inference_graph.py --pipeline_config_path=config/ssd_mobilenet_v1.config --trained_checkpoint_prefix=data/sim_training_data/sim_data_capture/model.ckpt-6000 --output_directory=model_frozen_sim/
```

- Use the created `.pb` file in your ROS classifier.

## Running

As it was suggested the VM runs the ROS with all the needed nodes. The simulator was running on the host it self. The machine for simulation had the following specs:

|OS| CPU | RAM | GPU
|--|---------|------- | ---------------------|
|MacOS| 2,5 GHz Intel Core i7 | 16 GB | Intel Iris Pro 1536 MB |

## Challenges

There were several challenges in this project:

- VirtualMachine/Simulator is too slow for having the images go through the tensorflow all the time. As I know the location of each traffic light I decided to have the images classifier 150 waypoints to the traffic light. With this the car has enough time to brake and the PID that is used to keep the car on the lane also works without latency (or minimal latency).

- The version mismatch between my own tensorflow on training machine and the tensorflow of the VM made some problems, but I was able to solve them with some upgrade
and downgrades.

## Udacity ReadMe

This is the project repo for the final project of the Udacity Self-Driving Car Nanodegree: Programming a Real Self-Driving Car. For more information about the project, see the project introduction [here](https://classroom.udacity.com/nanodegrees/nd013/parts/6047fe34-d93c-4f50-8336-b70ef10cb4b2/modules/e1a23b06-329a-4684-a717-ad476f0d8dff/lessons/462c933d-9f24-42d3-8bdc-a08a5fc866e4/concepts/5ab4b122-83e6-436d-850f-9f4d26627fd9).

Please use **one** of the two installation options, either native **or** docker installation.

### Native Installation

* Be sure that your workstation is running Ubuntu 16.04 Xenial Xerus or Ubuntu 14.04 Trusty Tahir. [Ubuntu downloads can be found here](https://www.ubuntu.com/download/desktop).
* If using a Virtual Machine to install Ubuntu, use the following configuration as minimum:
  * 2 CPU
  * 2 GB system memory
  * 25 GB of free hard drive space

  The Udacity provided virtual machine has ROS and Dataspeed DBW already installed, so you can skip the next two steps if you are using this.

* Follow these instructions to install ROS
  * [ROS Kinetic](http://wiki.ros.org/kinetic/Installation/Ubuntu) if you have Ubuntu 16.04.
  * [ROS Indigo](http://wiki.ros.org/indigo/Installation/Ubuntu) if you have Ubuntu 14.04.
* [Dataspeed DBW](https://bitbucket.org/DataspeedInc/dbw_mkz_ros)
  * Use this option to install the SDK on a workstation that already has ROS installed: [One Line SDK Install (binary)](https://bitbucket.org/DataspeedInc/dbw_mkz_ros/src/81e63fcc335d7b64139d7482017d6a97b405e250/ROS_SETUP.md?fileviewer=file-view-default)
* Download the [Udacity Simulator](https://github.com/udacity/CarND-Capstone/releases).

### Docker Installation
[Install Docker](https://docs.docker.com/engine/installation/)

Build the docker container
```bash
docker build . -t capstone
```

Run the docker file
```bash
docker run -p 4567:4567 -v $PWD:/capstone -v /tmp/log:/root/.ros/ --rm -it capstone
```

### Port Forwarding
To set up port forwarding, please refer to the [instructions from term 2](https://classroom.udacity.com/nanodegrees/nd013/parts/40f38239-66b6-46ec-ae68-03afd8a601c8/modules/0949fca6-b379-42af-a919-ee50aa304e6a/lessons/f758c44c-5e40-4e01-93b5-1a82aa4e044f/concepts/16cf4a78-4fc7-49e1-8621-3450ca938b77)

### Usage

1. Clone the project repository
```bash
git clone https://github.com/udacity/CarND-Capstone.git
```

2. Install python dependencies
```bash
cd CarND-Capstone
pip install -r requirements.txt
```
3. Make and run styx
```bash
cd ros
catkin_make
source devel/setup.sh
roslaunch launch/styx.launch
```
4. Run the simulator

### Real world testing
1. Download [training bag](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/traffic_light_bag_file.zip) that was recorded on the Udacity self-driving car.
2. Unzip the file
```bash
unzip traffic_light_bag_file.zip
```
3. Play the bag file
```bash
rosbag play -l traffic_light_bag_file/traffic_light_training.bag
```
4. Launch your project in site mode
```bash
cd CarND-Capstone/ros
roslaunch launch/site.launch
```
5. Confirm that traffic light detection works on real life images

### Other library/driver information
Outside of `requirements.txt`, here is information on other driver/library versions used in the simulator and Carla:

Specific to these libraries, the simulator grader and Carla use the following:

|        | Simulator | Carla  |
| :-----------: |:-------------:| :-----:|
| Nvidia driver | 384.130 | 384.130 |
| CUDA | 8.0.61 | 8.0.61 |
| cuDNN | 6.0.21 | 6.0.21 |
| TensorRT | N/A | N/A |
| OpenCV | 3.2.0-dev | 2.4.8 |
| OpenMP | N/A | N/A |

We are working on a fix to line up the OpenCV versions between the two.
