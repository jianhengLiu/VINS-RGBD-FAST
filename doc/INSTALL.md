# Dynamic-VINS

## 1. Prerequisites

**ROS**
```
sudo apt-get install ros-melodic-cv-bridge ros-melodic-tf ros-melodic-message-filters ros-melodic-image-transport ros-melodic-nav-msgs ros-melodic-visualization-msgs
```

**Ceres-Solver**
```
# CMake
sudo apt-get install cmake
# google-glog + gflags
sudo apt-get install libgoogle-glog-dev libgflags-dev
# BLAS & LAPACK
sudo apt-get install libatlas-base-dev
# Eigen3
sudo apt-get install libeigen3-dev
# SuiteSparse and CXSparse (optional)
sudo apt-get install libsuitesparse-dev
```
```
git clone https://ceres-solver.googlesource.com/ceres-solver
cd ceres-solver
git checkout 2.0.0
mkdir ceres-bin
cd ceres-bin
cmake ..
make -j3
sudo make install
```

**Sophus**
```
git clone https://github.com/strasdat/Sophus.git
cd Sophus
git checkout a621ff  #版本回溯
```
`gedit sophus/so2.cpp` modify `sophus/so2.cpp` as
```
SO2::SO2()
{
  unit_complex_.real(1.0);
  unit_complex_.imag(0.0);
}
```
build
```
mkdir build && cd build && cmake .. && sudo make install
```



## 2. Prerequisites for object detection 

We offer two kinds of device for tests, please follow the instruction for your match device.

### 2.1. NVIDIA devices

Clone the repository and catkin_make:

```
cd {YOUR_WORKSPACE}/src
git clone https://github.com/jianhengLiu/Dynamic-VINS.git

# build
cd ..
catkin_make
```



### 2.2. HUAWEI Atlas200


**!!!Note:**
 It is recommended to use high write speeds's microSD card(TF card), and a low write speeds' microSD card may result in Dyanmic-VINS not perferming in real time.(at least 40Mbs)

0. prequisities

```
  sudo apt install ros-melodic-image-transport-plugins
```

1. Clone the repository:

```
  cd {YOUR_WORKSPACE}/src
  git clone https://github.com/jianhengLiu/Dynamic-VINS.git

  git clone https://github.com/jianhengLiu/compressedimg2img.git
```

1. allocate core 
```
  sudo npu-smi set -t aicpu-config -i 0 -c 0 -d 2
```

4.  build
```   
  cd ../..
  catkin_make
```