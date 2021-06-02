#include "feature_tracker.h"

int FeatureTracker::n_id = 0;

bool inBorder(const cv::Point2f &pt) {
    const int BORDER_SIZE = 1;
    int img_x = cvRound(pt.x);
    int img_y = cvRound(pt.y);
    return BORDER_SIZE <= img_x && img_x < COL - BORDER_SIZE && BORDER_SIZE <= img_y && img_y < ROW - BORDER_SIZE;
}

void reduceVector(vector<cv::Point2f> &v, vector<uchar> status) {
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

void reduceVector(vector<int> &v, vector<uchar> status) {
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

FeatureTracker::FeatureTracker() {
    p_fast_feature_detector = cv::FastFeatureDetector::create();
}

void FeatureTracker::setMask() {
    if (FISHEYE)
        mask = fisheye_mask.clone();
    else
        mask = cv::Mat(ROW, COL, CV_8UC1, cv::Scalar(255));


    // prefer to keep features that are tracked for long time
    vector<pair<int, pair<cv::Point2f, int>>> cnt_pts_id;

    for (unsigned int i = 0; i < forw_pts.size(); i++)
        cnt_pts_id.push_back(make_pair(track_cnt[i], make_pair(forw_pts[i], ids[i])));

    sort(cnt_pts_id.begin(), cnt_pts_id.end(),
         [](const pair<int, pair<cv::Point2f, int>> &a, const pair<int, pair<cv::Point2f, int>> &b) {
             return a.first > b.first;
         });

    forw_pts.clear();
    ids.clear();
    track_cnt.clear();

    for (auto &it : cnt_pts_id) {
        if (mask.at<uchar>(it.second.first) == 255) {
            forw_pts.push_back(it.second.first);
            ids.push_back(it.second.second);
            track_cnt.push_back(it.first);
            cv::circle(mask, it.second.first, MIN_DIST, 0, -1);
        }
    }
    mask_exp = mask.clone();
    for (auto &pt : unstable_pts) {
        cv::circle(mask, pt, MIN_DIST, 0, -1);
    }
}

void FeatureTracker::addPoints() {
    for (auto &p : n_pts) {
        forw_pts.push_back(p);
        ids.push_back(-1);
        track_cnt.push_back(1);
    }
}

void FeatureTracker::addPoints(int n_max_cnt) {
    if (Keypts.empty()) {
        return;
    }

    sort(Keypts.begin(), Keypts.end(),
         [](const cv::KeyPoint &a, const cv::KeyPoint &b) {
             return a.response > b.response;
         });

    int n_add = 0;
    for (int i = 0; i < Keypts.size(); i++) {
        if (mask.at<uchar>(Keypts[i].pt) == 255) {

            forw_pts.push_back(Keypts[i].pt);
            ids.push_back(-1);
            track_cnt.push_back(1);
            cv::circle(mask, Keypts[i].pt, MIN_DIST, 0, -1);
            n_add++;
            if (n_add == n_max_cnt) {
                break;
            }
        }
    }
}

void FeatureTracker::readImage(const cv::Mat &_img, double _cur_time, Matrix3d relative_R) {
    cv::Mat img;
    TicToc t_r;
    cur_time = _cur_time;
    // too dark or too bright: histogram
    if (EQUALIZE) {
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        TicToc t_c;
        clahe->apply(_img, img);
        ROS_DEBUG("CLAHE costs: %fms", t_c.toc());
    } else
        img = _img;

    if (forw_img.empty()) {
        //curr_img<--->forw_img
        prev_img = cur_img = forw_img = img;
    } else {
        forw_img = img;
    }

    forw_pts.clear();
    unstable_pts.clear();

    if (cur_pts.size() > 0) {
        TicToc t_o;
        vector<uchar> status;
        vector<float> err;

        predictPtsInNextFrame(relative_R);
        forw_pts = predict_pts;
        cv::calcOpticalFlowPyrLK(cur_img, forw_img, cur_pts, forw_pts, status, err, cv::Size(21, 21), 3,
                                 cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01),
                                 cv::OPTFLOW_USE_INITIAL_FLOW);

        for (int i = 0; i < int(forw_pts.size()); i++) {
            if (!status[i] && inBorder(forw_pts[i])) {
                unstable_pts.push_back(forw_pts[i]);
            } else if (status[i] && !inBorder(forw_pts[i])) {
                status[i] = 0;
            }
        }

        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(forw_pts, status);
        reduceVector(ids, status);
        reduceVector(cur_un_pts, status);
        reduceVector(track_cnt, status);

        ROS_DEBUG("temporal optical flow costs: %fms", t_o.toc());
    }

    for (auto &n : track_cnt)
        n++;

    if (PUB_THIS_FRAME) {
        //对prev_pts和forw_pts做ransac剔除outlier.
        rejectWithF();
        ROS_DEBUG("set mask begins");
        TicToc t_m;
        setMask();
        ROS_DEBUG("set mask costs %fms", t_m.toc());

        ROS_DEBUG("detect feature begins");
        TicToc t_t;
        int n_max_cnt = MAX_CNT - static_cast<int>(forw_pts.size());
        if (n_max_cnt > 0) {
            if (mask.empty())
                cout << "mask is empty " << endl;
            if (mask.type() != CV_8UC1)
                cout << "mask type wrong " << endl;
            if (mask.size() != forw_img.size())
                cout << "wrong size " << endl;

            TicToc t_t_fast;

            p_fast_feature_detector->detect(forw_img, Keypts, mask);
        } else {
            n_pts.clear();
            Keypts.clear();
        }
        ROS_DEBUG("detect feature costs: %fms", t_t.toc());

        ROS_DEBUG("add feature begins");
        TicToc t_a;
        addPoints(n_max_cnt);
        ROS_DEBUG("selectFeature costs: %fms", t_a.toc());
    }
    prev_img = cur_img;
    prev_pts = cur_pts;
    prev_un_pts = cur_un_pts;
    cur_img = forw_img;
    cur_pts = forw_pts;
    //  去畸变，投影至归一化平面，计算特征点速度(pixel/s)
    undistortedPoints();
    prev_time = cur_time;
}

void FeatureTracker::rejectWithF() {
    if (forw_pts.size() >= 8) {
        ROS_DEBUG("FM ransac begins");
        TicToc t_f;
        vector<cv::Point2f> un_cur_pts(cur_pts.size()), un_forw_pts(forw_pts.size());
        for (unsigned int i = 0; i < cur_pts.size(); i++) {
            Eigen::Vector3d tmp_p;
            m_camera->liftProjective(Eigen::Vector2d(cur_pts[i].x, cur_pts[i].y), tmp_p);
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
            un_cur_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());

            m_camera->liftProjective(Eigen::Vector2d(forw_pts[i].x, forw_pts[i].y), tmp_p);
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
            un_forw_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());
        }

        vector<uchar> status;
        cv::findFundamentalMat(un_cur_pts, un_forw_pts, cv::FM_RANSAC, F_THRESHOLD, 0.99, status);
        int size_a = cur_pts.size();
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(forw_pts, status);
        reduceVector(cur_un_pts, status);
        reduceVector(ids, status);
        reduceVector(track_cnt, status);
        ROS_DEBUG("FM ransac: %d -> %lu: %f", size_a, forw_pts.size(), 1.0 * forw_pts.size() / size_a);
        ROS_DEBUG("FM ransac costs: %fms", t_f.toc());
    }
}

Eigen::Vector3d FeatureTracker::get3dPt(const cv::Mat &depth, const cv::Point2f &pt) {
    Eigen::Vector3d tmp_P;
    m_camera->liftProjective(Eigen::Vector2d(pt.x, pt.y), tmp_P);
    Eigen::Vector3d P = tmp_P.normalized() * (((int) depth.at<unsigned short>(round(pt.y), round(pt.x))) / 1000.0);

    return P;
}

const int r = 3;

void FeatureTracker::rejectWithPlane(const cv::Mat &depth) {
    vector<cv::KeyPoint> selectedKeypts;
    for (auto &Keypt : Keypts) {
        cv::Point2f p0 = Keypt.pt;
        Eigen::Vector3d P0 = get3dPt(depth, p0);
        if (P0.z() < DEPTH_MIN_DIST) {
            continue;
        }
        cv::Point2f p1 = p0 + cv::Point2f(-r, -r);
        if (!inBorder(p1))
            continue;
        Eigen::Vector3d P1 = get3dPt(depth, p1);
        if (P1.z() < DEPTH_MIN_DIST)
            continue;

        cv::Point2f p2 = p0 + cv::Point2f(r, -r);
        if (!inBorder(p2))
            continue;
        Eigen::Vector3d P2 = get3dPt(depth, p2);
        if (P2.z() < DEPTH_MIN_DIST)
            continue;

        cv::Point2f p3 = p0 + cv::Point2f(r, r);
        if (!inBorder(p3))
            continue;
        Eigen::Vector3d P3 = get3dPt(depth, p3);
        if (P3.z() < DEPTH_MIN_DIST)
            continue;

        cv::Point2f p4 = p0 + cv::Point2f(-r, r);
        if (!inBorder(p4))
            continue;
        Eigen::Vector3d P4 = get3dPt(depth, p4);
        if (P4.z() < DEPTH_MIN_DIST)
            continue;

        Eigen::Vector3d v12 = P2 - P1;
        Eigen::Vector3d v23 = P3 - P2;
        Eigen::Vector3d v34 = P4 - P3;
        Eigen::Vector3d v41 = P1 - P4;

        Eigen::Vector3d v13 = P3 - P1;
        Eigen::Vector3d v24 = P4 - P2;

        Eigen::Vector3d n0 = (v24.cross(v13)).normalized();

        Eigen::Vector3d n1 = (v12.cross(v41)).normalized();
        Eigen::Vector3d n2 = (v23.cross(v12)).normalized();
        Eigen::Vector3d n3 = (v34.cross(v23)).normalized();
        Eigen::Vector3d n4 = (v41.cross(-v12)).normalized();

        double e1 = acos(n0.dot(n1)) * 57.3;
        double e2 = acos(n0.dot(n2)) * 57.3;
        double e3 = acos(n0.dot(n3)) * 57.3;
        double e4 = acos(n0.dot(n4)) * 57.3;

        double e = (e1 + e2 + e3 + e4);

        if (e < 40) {
            continue;
        }

        Keypt.response += e;
        selectedKeypts.push_back(Keypt);
    }

    Keypts.clear();
    if (!selectedKeypts.empty()) {
        Keypts = selectedKeypts;
    }
}

bool FeatureTracker::updateID(unsigned int i) {
    if (i < ids.size()) {
        if (ids[i] == -1)
            ids[i] = n_id++;
        return true;
    } else
        return false;
}

void FeatureTracker::readIntrinsicParameter(const string &calib_file) {
    ROS_INFO("reading paramerter of camera %s", calib_file.c_str());
    m_camera = CameraFactory::instance()->generateCameraFromYamlFile(calib_file);
}

void FeatureTracker::showUndistortion(const string &name) {
    cv::Mat undistortedImg(ROW + 600, COL + 600, CV_8UC1, cv::Scalar(0));
    vector<Eigen::Vector2d> distortedp, undistortedp;
    for (int i = 0; i < COL; i++)
        for (int j = 0; j < ROW; j++) {
            Eigen::Vector2d a(i, j);
            Eigen::Vector3d b;
            m_camera->liftProjective(a, b);
            distortedp.push_back(a);
            undistortedp.push_back(Eigen::Vector2d(b.x() / b.z(), b.y() / b.z()));
            //printf("%f,%f->%f,%f,%f\n)\n", a.x(), a.y(), b.x(), b.y(), b.z());
        }
    for (int i = 0; i < int(undistortedp.size()); i++) {
        cv::Mat pp(3, 1, CV_32FC1);
        pp.at<float>(0, 0) = undistortedp[i].x() * FOCAL_LENGTH + COL / 2;
        pp.at<float>(1, 0) = undistortedp[i].y() * FOCAL_LENGTH + ROW / 2;
        pp.at<float>(2, 0) = 1.0;
        //cout << trackerData[0].K << endl;
        //printf("%lf %lf\n", p.at<float>(1, 0), p.at<float>(0, 0));
        //printf("%lf %lf\n", pp.at<float>(1, 0), pp.at<float>(0, 0));
        if (pp.at<float>(1, 0) + 300 >= 0 && pp.at<float>(1, 0) + 300 < ROW + 600 && pp.at<float>(0, 0) + 300 >= 0 &&
            pp.at<float>(0, 0) + 300 < COL + 600) {
            undistortedImg.at<uchar>(pp.at<float>(1, 0) + 300, pp.at<float>(0, 0) + 300) = cur_img.at<uchar>(
                    distortedp[i].y(), distortedp[i].x());
        } else {
            //ROS_ERROR("(%f %f) -> (%f %f)", distortedp[i].y, distortedp[i].x, pp.at<float>(1, 0), pp.at<float>(0, 0));
        }
    }
    cv::imshow(name, undistortedImg);
    cv::waitKey(0);
}

void FeatureTracker::undistortedPoints() {
    cur_un_pts.clear();
    cur_un_pts_map.clear();
    //cv::undistortPoints(cur_pts, un_pts, K, cv::Mat());
    for (unsigned int i = 0; i < cur_pts.size(); i++) {
        Eigen::Vector2d a(cur_pts[i].x, cur_pts[i].y);
        Eigen::Vector3d b;
        //https://github.com/HKUST-Aerial-Robotics/VINS-Mono/blob/0d280936e441ebb782bf8855d86e13999a22da63/camera_model/src/camera_models/PinholeCamera.cc
        //brief Lifts a point from the image plane to its projective ray
        m_camera->liftProjective(a, b);
        // 特征点在相机坐标系的归一化坐标
        cur_un_pts.push_back(cv::Point2f(b.x() / b.z(), b.y() / b.z()));
        cur_un_pts_map.insert(make_pair(ids[i], cv::Point2f(b.x() / b.z(), b.y() / b.z())));
        //printf("cur pts id %d %f %f", ids[i], cur_un_pts[i].x, cur_un_pts[i].y);
    }
    // caculate points velocity
    if (!prev_un_pts_map.empty()) {
        double dt = cur_time - prev_time;
        pts_velocity.clear();
        for (unsigned int i = 0; i < cur_un_pts.size(); i++) {
            if (ids[i] != -1) {
                std::map<int, cv::Point2f>::iterator it;
                it = prev_un_pts_map.find(ids[i]);
                if (it != prev_un_pts_map.end()) {
                    double v_x = (cur_un_pts[i].x - it->second.x) / dt;
                    double v_y = (cur_un_pts[i].y - it->second.y) / dt;
                    pts_velocity.push_back(cv::Point2f(v_x, v_y));
                } else
                    pts_velocity.push_back(cv::Point2f(0, 0));
            } else {
                pts_velocity.push_back(cv::Point2f(0, 0));
            }
        }
    } else {
        for (unsigned int i = 0; i < cur_pts.size(); i++) {
            pts_velocity.push_back(cv::Point2f(0, 0));
        }
    }
    prev_un_pts_map = cur_un_pts_map;
}

void FeatureTracker::predictPtsInNextFrame(Matrix3d relative_R) {
    predict_pts.resize(cur_pts.size());
    for (int i = 0; i < cur_pts.size(); ++i) {
        Eigen::Vector3d tmp_P;
        m_camera->liftProjective(Eigen::Vector2d(cur_pts[i].x, cur_pts[i].y), tmp_P);
        Eigen::Vector3d predict_P = relative_R * tmp_P;
        Eigen::Vector2d tmp_p;
        m_camera->spaceToPlane(predict_P, tmp_p);
        predict_pts[i].x = tmp_p.x();
        predict_pts[i].y = tmp_p.y();
    }
}