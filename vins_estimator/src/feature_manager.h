#ifndef FEATURE_MANAGER_H
#define FEATURE_MANAGER_H

#include <list>
#include <algorithm>
#include <vector>
#include <numeric>

using namespace std;

#include <eigen3/Eigen/Dense>

using namespace Eigen;

#include <ros/console.h>
#include <ros/assert.h>

#include "parameters.h"

#include "utility/tic_toc.h"

class FeaturePerFrame {
public:
    FeaturePerFrame(const Eigen::Matrix<double, 8, 1> &_point, double td) {
        point.x() = _point(0);
        point.y() = _point(1);
        point.z() = _point(2);
        uv.x() = _point(3);
        uv.y() = _point(4);
        velocity.x() = _point(5);
        velocity.y() = _point(6);
        depth = _point(7);
        cur_td = td;
    }

    double cur_td;
    Vector3d point;
    Vector2d uv;
    Vector2d velocity;
    double z;
    bool is_used;
    double parallax;
    MatrixXd A;
    VectorXd b;
    double dep_gradient;
    double depth;
};

class FeaturePerId {
public:
    const int feature_id;
    int start_frame;
    vector<FeaturePerFrame> feature_per_frame;

    int used_num;
    bool is_outlier;
    bool is_margin;
    double estimated_depth;
    double measured_depth;
    int estimate_flag;//0 initial; 1 by depth image; 2 by triangulate
    int solve_flag; // 0 haven't solve yet; 1 solve succ; 2 solve fail;

    double first_t;// feature first appear time

    Vector3d gt_p;

    FeaturePerId(int _feature_id, int _start_frame, double _measured_depth, double _first_t)
            : feature_id(_feature_id), start_frame(_start_frame),
              used_num(0), estimated_depth(-1.0), measured_depth(_measured_depth), first_t(_first_t), solve_flag(0),
              estimate_flag(0) {
    }

    int endFrame();
};

class FeatureManager {
public:
    FeatureManager(Matrix3d _Rs[]);

    void setRic(Matrix3d _ric[]);

    void clearState();

    int getFeatureCount();

    bool addFeatureCheckParallax(int frame_count, const map<int, Eigen::Matrix<double, 8, 1>> &image,
                                 double td, double img_t);

    void debugShow();

    vector<pair<Vector3d, Vector3d>> getCorresponding(int frame_count_l, int frame_count_r);

    vector<pair<Vector3d, Vector3d>> getCorrespondingWithDepth(int frame_count_l, int frame_count_r);

    //void updateDepth(const VectorXd &x);
    void setDepth(const VectorXd &x);

    void removeFailures();

    void clearDepth(const VectorXd &x);

    VectorXd getDepthVector();

    void triangulate(Vector3d Ps[], Vector3d tic[], Matrix3d ric[]);

    void triangulateWithDepth(Vector3d Ps[], Vector3d tic[], Matrix3d ric[]);

    void initFramePoseByPnP(int frameCnt, Vector3d Ps[], Matrix3d Rs[], Vector3d tic[], Matrix3d ric[]);

    bool solvePoseByPnP(Eigen::Matrix3d &R_initial, Eigen::Vector3d &P_initial,
                        vector<cv::Point2f> &pts2D, vector<cv::Point3f> &pts3D);

    void
    removeBackShiftDepth(Eigen::Matrix3d marg_R, Eigen::Vector3d marg_P, Eigen::Matrix3d new_R, Eigen::Vector3d new_P);

    void removeBack();

    void removeFront(int frame_count);

    void removeOutlier();

    list<FeaturePerId> feature;//cl:Lists将元素按顺序储存在链表中. 与 向量(vectors)相比, 它允许快速的插入和删除，但是随机访问却比较慢.
    int last_track_num;

private:
    double compensatedParallax2(const FeaturePerId &it_per_id, int frame_count);

    const Matrix3d *Rs;
    Matrix3d ric[NUM_OF_CAM];
};

#endif
