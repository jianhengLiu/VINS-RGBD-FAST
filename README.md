# VINS-RGBD-FAST

**VINS-RGBD-FAST** is a SLAM system based on VINS-RGBD. We do some refinements to accelerate the system's performance in resource-constrained embedded paltform, like [HUAWEI Atlas200 DK](https://e.huawei.com/en/products/cloud-computing-dc/atlas/atlas-200), [NVIDIA Jetson AGX Xavier](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-agx-xavier/).

## Refinements
* grid-based feature detection
* extract FAST feature instead of Harris feature
* added stationary initialization
* added IMU-aided feature tracking
* added extracted-feature area's quality judgement
* solved feature clusttering problem result frome FAST feature
* use "sensor_msg::CompressedImage" as image topic type
  
## Related Paper:
```
@ARTICLE{9830851,  
  author={Liu, Jianheng and Li, Xuanfu and Liu, Yueqian and Chen, Haoyao},  
  journal={IEEE Robotics and Automation Letters},  
  title={RGB-D Inertial Odometry for a Resource-Restricted Robot in Dynamic Environments},   
  year={2022},  
  volume={7},  
  number={4},  
  pages={9573-9580},  
  doi={10.1109/LRA.2022.3191193}}
```

## RGBD-Inertial Trajectory Estimation and Mapping for Small Ground Rescue Robot

Based one open source SLAM framework [VINS-Mono](https://github.com/HKUST-Aerial-Robotics/VINS-Mono).

The approach contains
+ Depth-integrated visual-inertial initialization process.
+ Visual-inertial odometry by utilizing depth information while avoiding the limitation is working for 3D pose estimation.
+ Noise elimination map which is suitable for path planning and navigation.

However, the proposed approach can also be applied to other application like handheld and wheeled robot.

## 1. Prerequisites
1.1. **Ubuntu** 16.04 or 18.04.

1.2. **ROS** version Kinetic or Melodic fully installation

1.3. **Ceres Solver**
Follow [Ceres Installation](http://ceres-solver.org/installation.html)

1.4. **Sophus**
```
  git clone http://github.com/strasdat/Sophus.git
  git checkout a621ff
```

1.3. **Atlas 200 DK环境配置**

[https://blog.csdn.net/qq_42703283/article/details/110389270](https://blog.csdn.net/qq_42703283/article/details/110389270)

1.4. **ROS多机通信**

[https://blog.csdn.net/qq_42703283/article/details/110408848](

## 2. Datasets

Recording by RealSense D435i. Contain 9 bags in three different applicaions:
+ [Handheld](https://star-center.shanghaitech.edu.cn/seafile/d/0ea45d1878914077ade5/)
+ [Wheeled robot](https://star-center.shanghaitech.edu.cn/seafile/d/78c0375114854774b521/) ([Jackal](https://www.clearpathrobotics.com/jackal-small-unmanned-ground-vehicle/))
+ [Tracked robot](https://star-center.shanghaitech.edu.cn/seafile/d/f611fc44df0c4b3d936d/)

Note the rosbags are in compressed format. Use "rosbag decompress" to decompress.

Topics:
+ depth topic: /camera/aligned_depth_to_color/image_raw
+ color topic: /camera/color/image_raw
+ imu topic: /camera/imu

我们使用的是压缩图像节点：

+ depth topic: /camera/aligned_depth_to_color/image_raw
+ color topic: /camera/color/image_raw/compressed
+ imu topic: /camera/imu

如何录制一个数据包

1. 运行d435i
2. `rosbag record /camera/imu /camera/color/image_raw /camera/aligned_depth_to_color/image_raw /camera/color/camera_info /camera/color/image_raw/compressed`


## 3. Run with Docker

make Dockerfile like below
```c
FROM ros:melodic-ros-core-bionic

# apt-get update
RUN apt-get update

# install essentials
RUN apt install -y gcc
RUN apt install -y g++
RUN apt-get install -y cmake
RUN apt-get install -y wget
RUN apt install -y git

# install ceres
WORKDIR /home
RUN apt-get install -y libgoogle-glog-dev libgflags-dev
RUN apt-get install -y libatlas-base-dev
RUN apt-get install -y libeigen3-dev
RUN apt-get install -y libsuitesparse-dev
RUN wget http://ceres-solver.org/ceres-solver-2.1.0.tar.gz
RUN tar zxf ceres-solver-2.1.0.tar.gz
WORKDIR /home/ceres-solver-2.1.0
RUN mkdir build
WORKDIR /home/ceres-solver-2.1.0/build
RUN cmake ..
RUN make
RUN make install

# install sophus
WORKDIR /home
RUN git clone https://github.com/demul/Sophus.git
WORKDIR /home/Sophus
RUN git checkout fix/unit_complex_eror
RUN mkdir build
WORKDIR /home/Sophus/build
RUN cmake ..
RUN make
RUN make install

# install ros dependencies
WORKDIR /home
RUN mkdir ros_ws
WORKDIR /home/ros_ws
RUN apt-get -y install ros-melodic-cv-bridge
RUN apt-get -y install ros-melodic-nodelet
RUN apt-get -y install ros-melodic-tf
RUN apt-get -y install ros-melodic-image-transport
RUN apt-get -y install ros-melodic-rviz

# build vins-rgbd-fast
RUN mkdir src
WORKDIR /home/ros_ws/src
RUN git clone https://github.com/jianhengLiu/VINS-RGBD-FAST
WORKDIR /home/ros_ws
RUN /bin/bash -c ". /opt/ros/melodic/setup.bash; cd /home/ros_ws; catkin_make"
RUN echo "source /home/ros_ws/devel/setup.bash" >> ~/.bashrc
```
docker build .

## 4. Licence
The source code is released under [GPLv3](http://www.gnu.org/licenses/) license.

