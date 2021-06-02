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


## 3. Licence
The source code is released under [GPLv3](http://www.gnu.org/licenses/) license.
