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


Estimator estimator;

std::condition_variable con;
double current_time = -1;
queue<sensor_msgs::ImuConstPtr> imu_buf;
queue<sensor_msgs::PointCloudConstPtr> feature_buf;
queue<sensor_msgs::PointCloudConstPtr> relo_buf;
int sum_of_wait = 0;

std::mutex m_buf;
std::mutex m_state;
std::mutex i_buf;
std::mutex m_estimator;

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

// 从IMU测量值imu_msg和上一个PVQ递推得到当前PVQ
// 对单次的IMU测量值做积分得到位移和姿态.
/**
*从两帧IMU数据中获得当前位姿的预测思路非常简单:
无非是求出当前时刻𝑡与下一时刻𝑡+1加速度的均值，
 把它作为Δ𝑡时间内的平均加速度，
 有了这个平均加速度及当前时刻的初始速度和初始位置，
 就可以近似的求出𝑡+1时刻的速度和位置
*/
void predict(const sensor_msgs::ImuConstPtr &imu_msg)
{
    double t = imu_msg->header.stamp.toSec();
    if (init_imu)
    {
        latest_time = t;
        init_imu = 0;
        return;
    }

    //计算当前imu_msg距离上一个imu_msg的时间间隔
    double dt = t - latest_time;
    latest_time = t;

    //获取x,y,z三个方向上的线加速度
    double dx = imu_msg->linear_acceleration.x;
    double dy = imu_msg->linear_acceleration.y;
    double dz = imu_msg->linear_acceleration.z;
    Eigen::Vector3d linear_acceleration{dx, dy, dz};

    //获取x,y,z三个方向上的角速度
    double rx = imu_msg->angular_velocity.x;
    double ry = imu_msg->angular_velocity.y;
    double rz = imu_msg->angular_velocity.z;
    Eigen::Vector3d angular_velocity{rx, ry, rz};

    // acc_0：body坐标系下的加速度——t时刻
    // tmp_Q * (acc_0 - tmp_Ba) : body坐标系下的加速度 - bias ,再通过矩阵转到世界坐标系下
    // un_acc_0: 世界坐标系下的加速度 = 融合了重力加速度的值 - 重力加速度 
    Eigen::Vector3d un_acc_0 = tmp_Q * (acc_0 - tmp_Ba) - estimator.g;  //t时刻的加速度

    // gyr_0: 
    // angular_velocity: 角速度--t+1时刻
    // tmp_Bg: g的bias 
    // un_gyr = 0.5 * (gyr_0 + angular_velocity) - tmp_Bg; 平均角速度
    Eigen::Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - tmp_Bg;
    // tmp_Q是t时刻的姿态， 需要计算出t+1时刻的姿态， 用角速度来近似
    // Qt+1 = Qt *(平均角速度* delta t)
    tmp_Q = tmp_Q * Utility::deltaQ(un_gyr * dt);

    Eigen::Vector3d un_acc_1 = tmp_Q * (linear_acceleration - tmp_Ba) - estimator.g; //t+1时刻的加速度

    //平均加速度
    Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);

    // 运动学公式
    // s = s0 + v*t + 1/2 * a *t*t
    // v = v0 + a*t
    tmp_P = tmp_P + dt * tmp_V + 0.5 * dt * dt * un_acc;
    tmp_V = tmp_V + dt * un_acc;

    // 更新存储 加速度 和 角速度
    acc_0 = linear_acceleration;
    gyr_0 = angular_velocity;
}

void update()
{
    //得到窗口最后一个图像帧的imu项[P,Q,V,ba,bg,a,g]，对imu_buf中剩余imu_msg进行PVQ递推
    // ! 了解predict ，但是estimator的的Ps，Rs，Vs等等是怎么得到的还不清楚 
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
getMeasurements()
{
    std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>> measurements;

    while (true)
    {
        //imu信息接收到后会在imu_callback回调中存入imu_buf，feature消息收到后会在feature_callback中存入feature_buf中
        if (imu_buf.empty() || feature_buf.empty())
            return measurements;

        //imu_buf中最后一个imu消息的时间戳比feature_buf中第一个feature消息的时间戳还要小，说明imu数据发出来太早了
        if (!(imu_buf.back()->header.stamp.toSec() > feature_buf.front()->header.stamp.toSec() + estimator.td))
        {   
            //imu数据比图像数据要早，所以返回，等待时间上和图像对齐的imu数据到来
            // 外层是一个while，所以会一直等待，直到imu时间戳和图像时间戳差不多对齐。
            //ROS_WARN("wait for imu, only should happen at the beginning");
            sum_of_wait++;
            return measurements;
        }
        
        //imu_buf中第一个数据的时间戳比feature_buf的第一个数据的时间戳还要大
        if (!(imu_buf.front()->header.stamp.toSec() < feature_buf.front()->header.stamp.toSec() + estimator.td))
        {
            //imu数据比图像数据滞后，所以图像数据要出队列， 
            // pop  直到大致对齐
            ROS_WARN("throw img, only should happen at the beginning");
            feature_buf.pop();
            continue;
        }

        // !一下讨论是对齐后的情况了 

        // 取图像队列头部数据
        sensor_msgs::PointCloudConstPtr img_msg = feature_buf.front();
        feature_buf.pop();

        // 取imu队列中时间戳小于图像时间戳的数据 
        std::vector<sensor_msgs::ImuConstPtr> IMUs;
        while (imu_buf.front()->header.stamp.toSec() < img_msg->header.stamp.toSec() + estimator.td)
        {
            IMUs.emplace_back(imu_buf.front());
            imu_buf.pop();
        }

        // 再放一个数据，是大于图像时间戳的第一个imu ,  但是没有pop，因为下一次是从这个数据取 
        IMUs.emplace_back(imu_buf.front());
        if (IMUs.empty())
            ROS_WARN("no imu between two image");
        measurements.emplace_back(IMUs, img_msg);
    }
    return measurements;
}

// imu订阅的回调函数为imu_callback，在imu_callback函数中对接收到的imu消息进行处理。
void imu_callback(const sensor_msgs::ImuConstPtr &imu_msg)
{   
    //用时间戳来判断IMU message是否乱序
    if (imu_msg->header.stamp.toSec() <= last_imu_t)
    {
        ROS_WARN("imu message in disorder!");
        return;
    }
    last_imu_t = imu_msg->header.stamp.toSec();
    
    
    //在修改多个线程共享的变量的时候要进行上锁，防止多个线程同时访问该变量
    //新来的imu_msg放入imu_buf队列当中
    m_buf.lock();
    imu_buf.push(imu_msg);
    m_buf.unlock();
    con.notify_one();

    last_imu_t = imu_msg->header.stamp.toSec();

    {
        std::lock_guard<std::mutex> lg(m_state);

        // 预测函数，这里推算的是tmp_P,tmp_Q,tmp_V
        // 从IMU测量值imu_msg和上一个PVQ递推得到当前PVQ
        // tmp_P(s:位置) , tep_V(v速度)， tmp_Q ： body->world的变换矩阵
        predict(imu_msg);
        std_msgs::Header header = imu_msg->header;
        header.frame_id = "world";

        // 发布imu_propagate topic
        //  发布最新的由imu直接递推得到的PQV
        if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
            pubLatestOdometry(tmp_P, tmp_Q, tmp_V, header);
    }
}


void feature_callback(const sensor_msgs::PointCloudConstPtr &feature_msg)
{
    if (!init_feature)
    {
        //skip the first detected feature, which doesn't contain optical flow speed
        init_feature = 1;
        return;
    }
    m_buf.lock();
    feature_buf.push(feature_msg);
    m_buf.unlock();
    con.notify_one();
}

void restart_callback(const std_msgs::BoolConstPtr &restart_msg)
{
    if (restart_msg->data == true)
    {
        ROS_WARN("restart the estimator!");
        m_buf.lock();
        while(!feature_buf.empty())
            feature_buf.pop();
        while(!imu_buf.empty())
            imu_buf.pop();
        m_buf.unlock();
        m_estimator.lock();
        estimator.clearState();
        estimator.setParameter();
        m_estimator.unlock();
        current_time = -1;
        last_imu_t = 0;
    }
    return;
}

void relocalization_callback(const sensor_msgs::PointCloudConstPtr &points_msg)
{
    //printf("relocalization callback! \n");
    m_buf.lock();
    relo_buf.push(points_msg);
    m_buf.unlock();
}

// thread: visual-inertial odometry
// process线程作为estimator中的主要线程，对接收到的imu消息和image消息进行处理，进而通过这些信息估计出相机位姿。
void process()
{
    while (true)
    {
        std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>> measurements;
        std::unique_lock<std::mutex> lk(m_buf);
        con.wait(lk, [&]
                 {
            //1.获取在时间上“对齐”的IMU和图像数据的组合
            return (measurements = getMeasurements()).size() != 0;
                 });
        lk.unlock();
        m_estimator.lock();

        // 2.遍历measurements，其实就是遍历获取每一个img_msg和其对应的imu_msg对数据进行处理
        // first 是imu数据
        // second 是图像数据
        for (auto &measurement : measurements)
        {
            auto img_msg = measurement.second;
            double dx = 0, dy = 0, dz = 0, rx = 0, ry = 0, rz = 0;


            // 2.1 遍历和当前img_msg时间上“对齐”的IMU数据
            // 目的是对每个imu数据做预积分
            for (auto &imu_msg : measurement.first)
            {
                double t = imu_msg->header.stamp.toSec();
                double img_t = img_msg->header.stamp.toSec() + estimator.td;


                //  case 1： imu数据比图像数据早到
                if (t <= img_t)
                { 
                    if (current_time < 0)
                        current_time = t;

                    // dt是imu数据的时间间隔
                    double dt = t - current_time;

                    
                    ROS_ASSERT(dt >= 0);
                    current_time = t;
                    dx = imu_msg->linear_acceleration.x;
                    dy = imu_msg->linear_acceleration.y;
                    dz = imu_msg->linear_acceleration.z;
                    rx = imu_msg->angular_velocity.x;
                    ry = imu_msg->angular_velocity.y;
                    rz = imu_msg->angular_velocity.z;
                    
                    //对每一个IMU值进行预积分
                    estimator.processIMU(dt, Vector3d(dx, dy, dz), Vector3d(rx, ry, rz));
                    //printf("imu: dt:%f a: %f %f %f w: %f %f %f\n",dt, dx, dy, dz, rx, ry, rz);

                }
                //  case 2： imu数据比图像数据晚到
                else
                {
                    // ! ???? 
                    double dt_1 = img_t - current_time;
                    double dt_2 = t - img_t;
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
            while (!relo_buf.empty())
            {
                relo_msg = relo_buf.front();
                relo_buf.pop();
            }
            if (relo_msg != NULL)
            {
                vector<Vector3d> match_points;
                double frame_stamp = relo_msg->header.stamp.toSec();
                for (unsigned int i = 0; i < relo_msg->points.size(); i++)
                {
                    Vector3d u_v_id;
                    u_v_id.x() = relo_msg->points[i].x;
                    u_v_id.y() = relo_msg->points[i].y;
                    u_v_id.z() = relo_msg->points[i].z;
                    match_points.push_back(u_v_id);
                }
                Vector3d relo_t(relo_msg->channels[0].values[0], relo_msg->channels[0].values[1], relo_msg->channels[0].values[2]);
                Quaterniond relo_q(relo_msg->channels[0].values[3], relo_msg->channels[0].values[4], relo_msg->channels[0].values[5], relo_msg->channels[0].values[6]);
                Matrix3d relo_r = relo_q.toRotationMatrix();
                int frame_index;
                frame_index = relo_msg->channels[0].values[7];

                // set relocalization frame 设置重定位帧
                estimator.setReloFrame(frame_stamp, frame_index, match_points, relo_t, relo_r);
            }

            ROS_DEBUG("processing vision data with stamp %f \n", img_msg->header.stamp.toSec());


            TicToc t_s;
            map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> image;
            // 2.2 遍历每一个img_msg中的特征点信息，所有的特征点信息构成了一个image信息
            for (unsigned int i = 0; i < img_msg->points.size(); i++)
            {
                int v = img_msg->channels[0].values[i] + 0.5;
                int feature_id = v / NUM_OF_CAM;
                int camera_id = v % NUM_OF_CAM;
                
                
                // img_msg 是继承自PointCloud： 常见属性有：
                // /** \brief A list of optional point cloud properties. See \ref CloudProperties for more information. */
                        //   pcl::CloudProperties properties;
                        //   Eigen::MatrixXf points;
                        //   std::map<std::string, pcl::ChannelProperties> channels;
                        //   uint32_t width;
                        //   uint32_t height;
                        //   bool is_dense;
                
                //获取img_msg中第i个点的x,y,z坐标，这个是归一化后的坐标值
                double x = img_msg->points[i].x;
                double y = img_msg->points[i].y;
                double z = img_msg->points[i].z;
                
                //获取像素的坐标值
                double p_u = img_msg->channels[1].values[i];
                double p_v = img_msg->channels[2].values[i];

                //获取像素点在x,y方向上的速度
                double velocity_x = img_msg->channels[3].values[i];
                double velocity_y = img_msg->channels[4].values[i];
                ROS_ASSERT(z == 1);

                //建立每个特征点的image map,索引为feature_id
                //image中每个特征点在帧中的位置信息和坐标轴上的速度信息按照feature_id为索引存入image中

                Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
                xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
                image[feature_id].emplace_back(camera_id,  xyz_uv_velocity);
            }

            // !图像特征处理，包括初始化和非线性优化 #####重点######
            estimator.processImage(image, img_msg->header);

            double whole_t = t_s.toc();
            printStatistics(estimator, whole_t);
            std_msgs::Header header = img_msg->header;
            header.frame_id = "world";

            //给Rviz发送信息
            pubOdometry(estimator, header);
            pubKeyPoses(estimator, header);
            pubCameraPose(estimator, header);
            pubPointCloud(estimator, header);
            pubTF(estimator, header);
            pubKeyframe(estimator);
            if (relo_msg != NULL)
                pubRelocalization(estimator);
            //ROS_ERROR("end: %f, at %f", img_msg->header.stamp.toSec(), ros::Time::now().toSec());
        }
        m_estimator.unlock();
        m_buf.lock();
        m_state.lock();
        if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
            // ! 
            update();
        m_state.unlock();
        m_buf.unlock();
    }
}

int main(int argc, char **argv)
{
    //1.相关初始化

    ros::init(argc, argv, "vins_estimator");
    ros::NodeHandle n("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);

    //2.读取参数
    readParameters(n);


    //3.设置状态估计器的参数
    //! ????? 
    estimator.setParameter();
#ifdef EIGEN_DONT_PARALLELIZE
    ROS_DEBUG("EIGEN_DONT_PARALLELIZE");
#endif
    ROS_WARN("waiting for image and imu...");
   
    //4.注册发布器
    registerPub(n);

    //5.订阅topic
    ros::Subscriber sub_imu = n.subscribe(IMU_TOPIC, 2000, imu_callback, ros::TransportHints().tcpNoDelay());
    ros::Subscriber sub_image = n.subscribe("/feature_tracker/feature", 2000, feature_callback);
    ros::Subscriber sub_restart = n.subscribe("/feature_tracker/restart", 2000, restart_callback);
    ros::Subscriber sub_relo_points = n.subscribe("/pose_graph/match_points", 2000, relocalization_callback);

    //6.创建process线程，这个是主线程 重要
    std::thread measurement_process{process};
    ros::spin();

    return 0;
}
