#define BACKWARD_HAS_DW 1

#include "backward.hpp"

namespace backward {
    backward::SignalHandling sh;
}

#include <stdio.h>
#include <queue>
#include <map>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

#include "estimator.h"
#include "parameters.h"
#include "utility/visualization.h"

#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>

#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

Estimator estimator;

std::condition_variable con;
std::condition_variable con_tracker;
double current_time = -1;
queue<sensor_msgs::ImuConstPtr> imu_buf;
queue<pair<sensor_msgs::CompressedImageConstPtr, sensor_msgs::CompressedImageConstPtr>> img_depth_buf;
queue<sensor_msgs::PointCloudConstPtr> feature_buf;
queue<sensor_msgs::PointCloudConstPtr> relo_buf;
int sum_of_wait = 0;

std::mutex m_buf;
std::mutex m_state;
std::mutex m_estimator;
std::mutex m_tracker;
std::mutex m_vis;

double latest_time;
Eigen::Vector3d tmp_P;
Eigen::Quaterniond tmp_Q;
Eigen::Vector3d tmp_V;
Eigen::Vector3d tmp_Ba;
Eigen::Vector3d tmp_Bg;
Eigen::Vector3d acc_0;
Eigen::Vector3d gyr_0;
bool init_feature = 0;
bool init_imu = 1;
double last_imu_t = 0;

void predict(const sensor_msgs::ImuConstPtr &imu_msg) {
    double t = imu_msg->header.stamp.toSec();
    if (init_imu) {
        latest_time = t;
        init_imu = 0;
        return;
    }
    double dt = t - latest_time;
    latest_time = t;

    double dx = imu_msg->linear_acceleration.x;
    double dy = imu_msg->linear_acceleration.y;
    double dz = imu_msg->linear_acceleration.z;
    Eigen::Vector3d linear_acceleration{dx, dy, dz};

    double rx = imu_msg->angular_velocity.x;
    double ry = imu_msg->angular_velocity.y;
    double rz = imu_msg->angular_velocity.z;
    Eigen::Vector3d angular_velocity{rx, ry, rz};

    Eigen::Vector3d un_acc_0 = tmp_Q * (acc_0 - tmp_Ba) - estimator.g;

    Eigen::Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - tmp_Bg;
    tmp_Q = tmp_Q * Utility::deltaQ(un_gyr * dt);

    Eigen::Vector3d un_acc_1 = tmp_Q * (linear_acceleration - tmp_Ba) - estimator.g;

    Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);


    tmp_P = tmp_P + dt * tmp_V + 0.5 * dt * dt * un_acc;
    tmp_V = tmp_V + dt * un_acc;

    acc_0 = linear_acceleration;
    gyr_0 = angular_velocity;
}

void update() {
    TicToc t_predict;
    latest_time = current_time;
    tmp_P = estimator.Ps[WINDOW_SIZE];
    tmp_Q = estimator.Rs[WINDOW_SIZE];
    tmp_V = estimator.Vs[WINDOW_SIZE];
    tmp_Ba = estimator.Bas[WINDOW_SIZE];
    tmp_Bg = estimator.Bgs[WINDOW_SIZE];
    acc_0 = estimator.acc_0;
    gyr_0 = estimator.gyr_0;

    queue<sensor_msgs::ImuConstPtr> tmp_imu_buf = imu_buf;
    for (sensor_msgs::ImuConstPtr tmp_imu_msg; !tmp_imu_buf.empty(); tmp_imu_buf.pop())
        predict(tmp_imu_buf.front());

}


std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>>
getMeasurements() {
    std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>> measurements;
    while (true) {
        if (imu_buf.empty() || feature_buf.empty()) {
            return measurements;
        }


        if (!(imu_buf.back()->header.stamp.toSec() > feature_buf.front()->header.stamp.toSec() + estimator.td)) {
            ROS_WARN("wait for imu, only should happen at the beginning");
            sum_of_wait++;
            return measurements;
        }

        if (!(imu_buf.front()->header.stamp.toSec() < feature_buf.front()->header.stamp.toSec() + estimator.td)) {
            ROS_WARN("throw img, only should happen at the beginning");
            feature_buf.pop();
            continue;
        }

        sensor_msgs::PointCloudConstPtr feature_msg = feature_buf.front();
        feature_buf.pop();

        std::vector<sensor_msgs::ImuConstPtr> IMUs;
        while (imu_buf.front()->header.stamp.toSec() < feature_msg->header.stamp.toSec() + estimator.td) {
            IMUs.emplace_back(imu_buf.front());
            imu_buf.pop();
        }
        IMUs.emplace_back(imu_buf.front());
        if (IMUs.empty())
            ROS_WARN("no imu between two image");

        measurements.emplace_back(IMUs, feature_msg);
    }
    return measurements;
}

bool first_image_flag = true;
double first_image_time;
double last_image_time = 0;
double prev_image_time = 0;
bool init_pub = 0;
int pub_count = 1;
queue<sensor_msgs::CompressedImageConstPtr> vis_img_buf;

void imu_callback(const sensor_msgs::ImuConstPtr &imu_msg) {
    if (imu_msg->header.stamp.toSec() <= last_imu_t) {
        ROS_WARN("imu message in disorder! %f", imu_msg->header.stamp.toSec());
        return;
    }
    last_imu_t = imu_msg->header.stamp.toSec();

    m_buf.lock();
    imu_buf.push(imu_msg);
    m_buf.unlock();
    con.notify_one();

    last_imu_t = imu_msg->header.stamp.toSec();

    // cl:pub "imu_propagate"
    {
        std::lock_guard<std::mutex> lg(m_state);
        //predict imu (no residual error)
        predict(imu_msg);
        std_msgs::Header header = imu_msg->header;
        header.frame_id = "world";
        if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
            pubLatestOdometry(tmp_P, tmp_Q, tmp_V, header);
    }
}

Matrix3d integrateImuData() {
    queue<sensor_msgs::ImuConstPtr> tmp_imu_buf = imu_buf;
    if (tmp_imu_buf.empty() || prev_image_time == 0 || last_image_time == 0)
        return Matrix3d::Identity();

    // Compute the mean angular velocity in the IMU frame.
    Vector3d mean_ang_vel(0.0, 0.0, 0.0);
    int cnt = 0;
    while (!tmp_imu_buf.empty()) {
        double t = tmp_imu_buf.front()->header.stamp.toSec();
        if ((t - prev_image_time) > -0.01 && (t - last_image_time) < 0.005) {
            mean_ang_vel += Vector3d(tmp_imu_buf.front()->angular_velocity.x, tmp_imu_buf.front()->angular_velocity.y,
                                     tmp_imu_buf.front()->angular_velocity.z);
            cnt++;
        }
        tmp_imu_buf.pop();
    }
    if (cnt > 0)
        mean_ang_vel *= 1.0f / cnt;
    else
        return Matrix3d::Identity();

    mean_ang_vel -= tmp_Bg;
    // Transform the mean angular velocity from the IMU
    // frame to the cam0 frames.
    Vector3d cam0_mean_ang_vel = Ric.transpose() * mean_ang_vel;

    // Compute the relative rotation.
    double dtime = last_image_time - prev_image_time;
    Vector3d cam0_angle_axisd = cam0_mean_ang_vel * dtime;

    return AngleAxisd(cam0_angle_axisd.norm(), cam0_angle_axisd.normalized()).toRotationMatrix().transpose();

}

void img_callback(const sensor_msgs::CompressedImageConstPtr &color_msg,
                  const sensor_msgs::CompressedImageConstPtr &depth_msg) {
    m_tracker.lock();
    img_depth_buf.emplace(color_msg, depth_msg);
    m_tracker.unlock();
    con_tracker.notify_one();
}

void relocalization_callback(const sensor_msgs::PointCloudConstPtr &points_msg) {
    //printf("relocalization callback! \n");
    m_buf.lock();
    relo_buf.push(points_msg);
    m_buf.unlock();
}

// thread: feature tracker
void process_tracker() {
    while (1) {
        sensor_msgs::CompressedImageConstPtr color_msg = NULL;
        sensor_msgs::CompressedImageConstPtr depth_msg = NULL;
        std::unique_lock<std::mutex> lk_tracker(m_tracker);
        con_tracker.wait(lk_tracker, [&] {
            return !img_depth_buf.empty();
        });
        color_msg = img_depth_buf.front().first;
        depth_msg = img_depth_buf.front().second;
        img_depth_buf.pop();
        lk_tracker.unlock();

        if (first_image_flag) {
            first_image_flag = false;
            first_image_time = color_msg->header.stamp.toSec();
            last_image_time = color_msg->header.stamp.toSec();
            continue;
        }
        // detect unstable camera stream
        if (color_msg->header.stamp.toSec() - last_image_time > 1.0 ||
            color_msg->header.stamp.toSec() < last_image_time) {
            ROS_WARN("image discontinue! reset the feature tracker!");
            first_image_flag = true;
            last_image_time = 0;
            pub_count = 1;

            ROS_WARN("restart the estimator!");
            m_buf.lock();
            while (!feature_buf.empty())
                feature_buf.pop();
            while (!imu_buf.empty())
                imu_buf.pop();
            m_buf.unlock();
            m_estimator.lock();
            estimator.clearState();
            estimator.setParameter();
            m_estimator.unlock();
            current_time = -1;
            last_imu_t = 0;

            continue;
        }
        prev_image_time = last_image_time;
        last_image_time = color_msg->header.stamp.toSec();
        // frequency control
        if (round(1.0 * pub_count / (color_msg->header.stamp.toSec() - first_image_time)) <= FREQ) {
            PUB_THIS_FRAME = true;
            // reset the frequency control
            if (abs(1.0 * pub_count / (color_msg->header.stamp.toSec() - first_image_time) - FREQ) < 0.01 * FREQ) {
                first_image_time = color_msg->header.stamp.toSec();
                pub_count = 0;
            }
        } else
            PUB_THIS_FRAME = false;
        // encodings in ros: http://docs.ros.org/diamondback/api/sensor_msgs/html/image__encodings_8cpp_source.html
        //color has encoding RGB8
        cv_bridge::CvImageConstPtr ptr;
        ptr = cv_bridge::toCvCopy(color_msg, sensor_msgs::image_encodings::MONO8);

        // Get compressed image data
        // https://answers.ros.org/question/51490/sensor_msgscompressedimage-decompression/
        // https://github.com/heleidsn/heleidsn.github.io/blob/c02ed13cb4ffe85ee8f03a9ad93fa55336f84f7c/source/_posts/realsense-depth-image.md
        // https://sourcegraph.com/github.com/ros-perception/image_transport_plugins/-/blob/compressed_depth_image_transport/include/compressed_depth_image_transport/codec.h
        const std::vector<uint8_t> imageData(depth_msg->data.begin() + 12, depth_msg->data.end());
        cv::Mat depth_img = cv::imdecode(imageData, cv::IMREAD_UNCHANGED);

        cv::Mat show_img = ptr->image;
        TicToc t_r;
        ROS_DEBUG("processing camera");
        Matrix3d relative_R = integrateImuData();
        estimator.featureTracker.readImage(ptr->image.rowRange(0, ROW),
                                           color_msg->header.stamp.toSec(),
                                           relative_R);
        //always 0
#if SHOW_UNDISTORTION
        trackerData[i].showUndistortion("undistrotion_" + std::to_string(i));
#endif
// update all id in ids[]
//If has ids[i] == -1 (newly added pts by cv::goodFeaturesToTrack), substitute by gloabl id counter (n_id)
        for (unsigned int i = 0;; i++) {
            bool completed = false;
            completed |= estimator.featureTracker.updateID(i);
            if (!completed)
                break;
        }
        if (PUB_THIS_FRAME) {
            vector<int> test;
            pub_count++;
            //http://docs.ros.org/api/sensor_msgs/html/msg/PointCloud.html
            sensor_msgs::PointCloudPtr feature_points(new sensor_msgs::PointCloud);
            sensor_msgs::ChannelFloat32 id_of_point;
            sensor_msgs::ChannelFloat32 u_of_point;
            sensor_msgs::ChannelFloat32 v_of_point;
            sensor_msgs::ChannelFloat32 velocity_x_of_point;
            sensor_msgs::ChannelFloat32 velocity_y_of_point;
            //Use round to get depth value of corresponding points
            sensor_msgs::ChannelFloat32 depth_of_point;

            feature_points->header = color_msg->header;
            feature_points->header.frame_id = "world";

            auto &un_pts = estimator.featureTracker.cur_un_pts;
            auto &cur_pts = estimator.featureTracker.cur_pts;
            auto &ids = estimator.featureTracker.ids;
            auto &pts_velocity = estimator.featureTracker.pts_velocity;
            for (unsigned int j = 0; j < ids.size(); j++) {
                if (estimator.featureTracker.track_cnt[j] > 1) {
                    int p_id = ids[j];
                    geometry_msgs::Point32 p;
                    p.x = un_pts[j].x;
                    p.y = un_pts[j].y;
                    p.z = 1;
                    // push normalized point to pointcloud
                    feature_points->points.push_back(p);
                    // push other info
                    id_of_point.values.push_back(p_id * NUM_OF_CAM);
                    u_of_point.values.push_back(cur_pts[j].x);
                    v_of_point.values.push_back(cur_pts[j].y);
                    velocity_x_of_point.values.push_back(pts_velocity[j].x);
                    velocity_y_of_point.values.push_back(pts_velocity[j].y);

                    //nearest neighbor....fastest  may be changed
                    // show_depth: 480*640   y:[0,480]   x:[0,640]
                    depth_of_point.values.push_back(
                            (int) depth_img.at<unsigned short>(round(cur_pts[j].y), round(cur_pts[j].x)));
                }
            }


            feature_points->channels.push_back(id_of_point);
            feature_points->channels.push_back(u_of_point);
            feature_points->channels.push_back(v_of_point);
            feature_points->channels.push_back(velocity_x_of_point);
            feature_points->channels.push_back(velocity_y_of_point);
            feature_points->channels.push_back(depth_of_point);
            ROS_DEBUG("publish %f, at %f", feature_points->header.stamp.toSec(), ros::Time::now().toSec());
            // skip the first image; since no optical speed on frist image
            if (!init_pub) {
                init_pub = 1;
            } else {
                if (!init_feature) {
                    //skip the first detected feature, which doesn't contain optical flow speed
                    init_feature = 1;
                    continue;
                }
                m_buf.lock();
                feature_buf.push(feature_points);
                m_buf.unlock();
                con.notify_one();
            }
            if (SHOW_TRACK) {
                m_vis.lock();
                vis_img_buf.emplace(color_msg);
                m_vis.unlock();
            }

            // Show image with tracked points in rviz (by topic pub_match)
            if (SHOW_TRACK) {
                ptr = cv_bridge::cvtColor(ptr, sensor_msgs::image_encodings::BGR8);
                //cv::Mat stereo_img(ROW * NUM_OF_CAM, COL, CV_8UC3);
                cv::Mat stereo_img = ptr->image;

                cv::Mat tmp_img = stereo_img.rowRange(0, ROW);
                cv::cvtColor(show_img, tmp_img, CV_GRAY2RGB);//??seems useless?

                for (unsigned int j = 0; j < estimator.featureTracker.cur_pts.size(); j++) {
                    double len = std::min(1.0, 1.0 * estimator.featureTracker.track_cnt[j] / WINDOW_SIZE);
                    cv::circle(tmp_img, estimator.featureTracker.cur_pts[j], 2,
                               cv::Scalar(255 * (1 - len), 0, 255 * len),
                               2);
                    //draw speed line
                    /*
                    Vector2d tmp_cur_un_pts (trackerData[i].cur_un_pts[j].x, trackerData[i].cur_un_pts[j].y);
                    Vector2d tmp_pts_velocity (trackerData[i].pts_velocity[j].x, trackerData[i].pts_velocity[j].y);
                    Vector3d tmp_prev_un_pts;
                    tmp_prev_un_pts.head(2) = tmp_cur_un_pts - 0.10 * tmp_pts_velocity;
                    tmp_prev_un_pts.z() = 1;
                    Vector2d tmp_prev_uv;
                    trackerData[i].m_camera->spaceToPlane(tmp_prev_un_pts, tmp_prev_uv);
                    cv::line(tmp_img, trackerData[i].cur_pts[j], cv::Point2f(tmp_prev_uv.x(), tmp_prev_uv.y()), cv::Scalar(255 , 0, 0), 1 , 8, 0);
                    */
                    //char name[10];
                    //sprintf(name, "%d", trackerData[i].ids[j]);
                    //cv::putText(tmp_img, name, trackerData[i].cur_pts[j], cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
                }
                for (unsigned int j = 0; j < estimator.featureTracker.predict_pts.size(); j++) {
                    cv::circle(tmp_img, estimator.featureTracker.predict_pts[j], 1, cv::Scalar(0, 255, 0), 1);
                }
                //                for (unsigned int j = 0; j < trackerData[i].Keypts.size(); j++) {
                //                    cv::Mat e_img = stereo_img.rowRange(i * ROW, (i + 1) * ROW).clone();
                //                    cv::circle(e_img, trackerData[i].Keypts[j].pt, 1, cv::Scalar(0, 0, 255), 1);
                //                    cv::putText(e_img, std::to_string(trackerData[i].Keypts[j].response), trackerData[i].Keypts[j].pt,
                //                                cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(0, 255, 0));
                //                    cv::imshow("e_img", e_img);
                //                    cv::waitKey();
                //                }
                pubTrackImg(ptr);
            }
        }
//    ROS_INFO("whole feature tracker processing costs: %f", t_r.toc());
        std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
    }
}

// thread: visual-inertial odometry
void process() {
    while (true) {
        std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>> measurements;
        std::unique_lock<std::mutex> lk(m_buf);
        con.wait(lk, [&] {
            return (measurements = getMeasurements()).size() != 0;
        });
        lk.unlock();
        m_estimator.lock();
        for (auto &measurement: measurements) {
            auto feature_msg = measurement.second;//measurement.second;
            double dx = 0, dy = 0, dz = 0, rx = 0, ry = 0, rz = 0;
            for (auto &imu_msg: measurement.first) {
                double t = imu_msg->header.stamp.toSec();
                double img_t = feature_msg->header.stamp.toSec() + estimator.td;
                if (t <= img_t) {
                    if (current_time < 0)
                        current_time = t;
                    double dt = t - current_time;
                    ROS_ASSERT(dt >= 0);
                    current_time = t;
                    dx = imu_msg->linear_acceleration.x;
                    dy = imu_msg->linear_acceleration.y;
                    dz = imu_msg->linear_acceleration.z;
                    rx = imu_msg->angular_velocity.x;
                    ry = imu_msg->angular_velocity.y;
                    rz = imu_msg->angular_velocity.z;
                    estimator.processIMU(dt, Vector3d(dx, dy, dz), Vector3d(rx, ry, rz));
                    //printf("imu: dt:%f a: %f %f %f w: %f %f %f\n",dt, dx, dy, dz, rx, ry, rz);

                } else {
                    double dt_1 = img_t - current_time;//cl:当前帧与前一刻imu的时间差
                    double dt_2 = t - img_t;//cl:后一刻imu与当前帧的时间差
                    current_time = img_t;
                    ROS_ASSERT(dt_1 >= 0);
                    ROS_ASSERT(dt_2 >= 0);
                    ROS_ASSERT(dt_1 + dt_2 > 0);
                    double w1 = dt_2 / (dt_1 + dt_2);
                    double w2 = dt_1 / (dt_1 + dt_2);
                    dx = w1 * dx + w2 * imu_msg->linear_acceleration.x;
                    dy = w1 * dy + w2 * imu_msg->linear_acceleration.y;
                    dz = w1 * dz + w2 * imu_msg->linear_acceleration.z;
                    rx = w1 * rx + w2 * imu_msg->angular_velocity.x;
                    ry = w1 * ry + w2 * imu_msg->angular_velocity.y;
                    rz = w1 * rz + w2 * imu_msg->angular_velocity.z;
                    estimator.processIMU(dt_1, Vector3d(dx, dy, dz), Vector3d(rx, ry, rz));
                    //printf("dimu: dt:%f a: %f %f %f w: %f %f %f\n",dt_1, dx, dy, dz, rx, ry, rz);
                }
            }
            // set relocalization frame
            sensor_msgs::PointCloudConstPtr relo_msg = NULL;
            while (!relo_buf.empty()) {
                relo_msg = relo_buf.front();
                relo_buf.pop();
            }
            if (relo_msg != NULL) {
                vector<Vector3d> match_points;
                double frame_stamp = relo_msg->header.stamp.toSec();
                for (unsigned int i = 0; i < relo_msg->points.size(); i++) {
                    Vector3d u_v_id;
                    u_v_id.x() = relo_msg->points[i].x;
                    u_v_id.y() = relo_msg->points[i].y;
                    u_v_id.z() = relo_msg->points[i].z;
                    match_points.push_back(u_v_id);
                }
                Vector3d relo_t(relo_msg->channels[0].values[0], relo_msg->channels[0].values[1],
                                relo_msg->channels[0].values[2]);
                Quaterniond relo_q(relo_msg->channels[0].values[3], relo_msg->channels[0].values[4],
                                   relo_msg->channels[0].values[5], relo_msg->channels[0].values[6]);
                Matrix3d relo_r = relo_q.toRotationMatrix();
                int frame_index;
                frame_index = relo_msg->channels[0].values[7];
                estimator.setReloFrame(frame_stamp, frame_index, match_points, relo_t, relo_r);
            }

            ROS_DEBUG("processing vision data with stamp %f \n", feature_msg->header.stamp.toSec());

            TicToc t_s;
            map<int, Eigen::Matrix<double, 8, 1>> image;
            for (unsigned int i = 0; i < feature_msg->points.size(); i++) {
                int v = feature_msg->channels[0].values[i] + 0.5;
                int feature_id = v / NUM_OF_CAM;

                int camera_id = v % NUM_OF_CAM;
                double x = feature_msg->points[i].x;
                double y = feature_msg->points[i].y;
                double z = feature_msg->points[i].z;
                double p_u = feature_msg->channels[1].values[i];
                double p_v = feature_msg->channels[2].values[i];
                double velocity_x = feature_msg->channels[3].values[i];
                double velocity_y = feature_msg->channels[4].values[i];
                double depth = feature_msg->channels[5].values[i] / 1000.0;

                ROS_ASSERT(z == 1);
                Eigen::Matrix<double, 8, 1> xyz_uv_velocity_depth;
                xyz_uv_velocity_depth << x, y, z, p_u, p_v, velocity_x, velocity_y, depth;
                image[feature_id] = xyz_uv_velocity_depth;
            }

            estimator.processImage(image, feature_msg->header);

            double whole_t = t_s.toc();

            printStatistics(estimator, whole_t);
            std_msgs::Header header = feature_msg->header;
            header.frame_id = "world";
            // utility/visualization.cpp
            pubOdometry(estimator, header);
            pubKeyPoses(estimator, header);
            pubCameraPose(estimator, header);
            pubIMUPose(estimator, header);
            pubPointCloud(estimator, header);
            pubTF(estimator, header);
            pubKeyframe(estimator);
            if (relo_msg != NULL)
                pubRelocalization(estimator);
            //ROS_ERROR("end: %f, at %f", feature_msg->header.stamp.toSec(), ros::Time::now().toSec());
        }
        m_estimator.unlock();
        m_buf.lock();
        m_state.lock();
        if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
            update();
        m_state.unlock();
        m_buf.unlock();
        std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
    }

}

int main(int argc, char **argv) {
    ros::init(argc, argv, "vins_estimator");
    ros::NodeHandle n("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);
    readParameters(n);
//std::string config_file = "/home/chrisliu/ROSws_nuc/AtalsSLAM_ws/src/vins-rgbd-atlas/config/realsense/lifelong_config.yaml";
//    readParameters(config_file);

    estimator.setParameter();
#ifdef EIGEN_DONT_PARALLELIZE
    ROS_DEBUG("EIGEN_DONT_PARALLELIZE");
#endif
    ROS_WARN("waiting for image and imu...");

    registerPub(n);

    //ref: http://docs.ros.org/api/message_filters/html/c++/classmessage__filters_1_1TimeSynchronizer.html#a9e58750270e40a2314dd91632a9570a6
    //     https://blog.csdn.net/zyh821351004/article/details/47758433
    message_filters::Subscriber<sensor_msgs::CompressedImage> sub_image(n, IMAGE_TOPIC + "/compressed", 1000);
    message_filters::Subscriber<sensor_msgs::CompressedImage> sub_depth(n, DEPTH_TOPIC + "/compressedDepth", 1000);

    // use ApproximateTime to fit fisheye camera
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::CompressedImage, sensor_msgs::CompressedImage> syncPolicy;
    message_filters::Synchronizer<syncPolicy> sync(syncPolicy(100), sub_image, sub_depth);
    sync.registerCallback(boost::bind(&img_callback, _1, _2));

    ros::Subscriber sub_imu = n.subscribe(IMU_TOPIC, 1000, imu_callback, ros::TransportHints().tcpNoDelay());
    //topic from pose_graph, notify if there's relocalization
    ros::Subscriber sub_relo_points = n.subscribe("/pose_graph/match_points", 1000, relocalization_callback);

    std::thread trackThread{process_tracker};
    std::thread processThread{process};
    ros::MultiThreadedSpinner spinner(4);
    spinner.spin();

    return 0;
}
