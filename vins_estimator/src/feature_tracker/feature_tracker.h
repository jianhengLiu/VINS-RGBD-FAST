#pragma once

#include <cstdio>
#include <iostream>
#include <queue>
#include <execinfo.h>
#include <csignal>

#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>

#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"

#include "../parameters.h"
#include "../utility/tic_toc.h"

using namespace std;
using namespace camodocal;
using namespace Eigen;

bool inBorder(const cv::Point2f &pt);

void reduceVector(vector<cv::Point2f> &v, vector<uchar> status);

void reduceVector(vector<int> &v, vector<uchar> status);

class FeatureTracker {
public:
    FeatureTracker();

    void readImage(const cv::Mat &_img, double _cur_time, Matrix3d relative_R);

    void setMask();

    void addPoints();

    void addPoints(int n_max_cnt);

    bool updateID(unsigned int i);

    void readIntrinsicParameter(const string &calib_file);

    void showUndistortion(const string &name);

    void rejectWithF();


    void undistortedPoints();

    void predictPtsInNextFrame(Matrix3d relative_R);

    void rejectWithPlane(const cv::Mat &depth);

    Eigen::Vector3d get3dPt(const cv::Mat &_depth, const cv::Point2f &pt);

    void readImage(const cv::Mat &_img, const cv::Mat &_depth, double _cur_time, Matrix3d relative_R);

    cv::Mat mask, mask_exp;
    cv::Mat fisheye_mask;
    cv::Mat prev_img, cur_img, forw_img;
    vector<cv::Point2f> n_pts;
    vector<cv::KeyPoint> Keypts;
    vector<cv::Point2f> prev_pts, cur_pts, forw_pts, predict_pts, unstable_pts;
    vector<cv::Point2f> prev_un_pts, cur_un_pts;
    vector<cv::Point2f> pts_velocity;
    vector<int> ids;
    vector<int> track_cnt;
    map<int, cv::Point2f> cur_un_pts_map;
    map<int, cv::Point2f> prev_un_pts_map;
    camodocal::CameraPtr m_camera;
    double cur_time;
    double prev_time;

    static int n_id;

    bool hasPrediction = false;
};
