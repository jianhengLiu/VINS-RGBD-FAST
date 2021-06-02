#pragma once
#include <ros/ros.h>
#include <eigen3/Eigen/Dense>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/eigen.hpp>

extern int ROW;
extern int COL;
extern int FOCAL_LENGTH;
const int NUM_OF_CAM = 1;


extern std::string IMAGE_TOPIC;
extern std::string DEPTH_TOPIC;
extern std::string IMU_TOPIC;
extern std::string FISHEYE_MASK;
extern std::vector<std::string> CAM_NAMES;
extern int MAX_CNT;
extern int MIN_DIST;
extern int WINDOW_SIZE;
extern int FREQ;
extern double F_THRESHOLD;
extern int SHOW_TRACK;
extern int STEREO_TRACK;
extern int EQUALIZE;
extern int FISHEYE;
extern bool PUB_THIS_FRAME;

extern double DEPTH_MIN_DIST;
extern double DEPTH_MAX_DIST;

extern Eigen::Matrix3d Ric;
extern Eigen::Vector3d Tic;

void readParameters(ros::NodeHandle &n);

void readParameters(std::string config_file,std::string VINS_FOLDER_PATH);