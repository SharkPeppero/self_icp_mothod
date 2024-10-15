//
// Created by xu on 24-9-9.
//

#ifndef EIGEN_ICP_POINT2POINT_SVD_H
#define EIGEN_ICP_POINT2POINT_SVD_H

#include <vector>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/transforms.h>
#include <pcl/kdtree/kdtree_flann.h>


/**
 * @brief ICPSVD实现
 * @tparam PointT
 */
template<typename PointT>
class ICP_SVD {
public:

    // 构造函数
    ICP_SVD();

    // 析构函数
    ~ICP_SVD();

    // 进行SVD求解
    void SVDHandle();

    // 设置目标点云数据
    void setTargetCloud(const typename pcl::PointCloud<PointT>::Ptr &target_cloud_ptr);

    // 设置source点云数据
    void setSourceCloud(const typename pcl::PointCloud<PointT>::Ptr &source_cloud_ptr);

    // 设置迭代次数（不设置的话默认为2）
    void setIterCnts(int iter_cnts);

    // 设置精度阈值（不设置的话默认为1e-6）
    void setEpsilon(double epsilon);

    // 设置初始外参（不设置的话默认初值）
    void setInitExternalParam(Eigen::Matrix3d R_target_source, Eigen::Vector3d t_target_source);


    // 获取计算的外参
    Eigen::Matrix4d getExternalParam();

private:
    int iter_cnts_ = 5;             // 最大的迭代次数
    double epsilon_ = 1e-6;         // 平移量的epsilon阈值
    double nearest_dist_ = 5.0;     // 最近邻迭代是否控制最近邻搜索距离

    typename pcl::PointCloud<PointT>::Ptr target_cloud_ptr_ = nullptr;
    typename pcl::PointCloud<PointT>::Ptr source_cloud_ptr_ = nullptr;

    Eigen::Matrix3d R_target_source_ = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t_target_source_ = Eigen::Vector3d::Zero();
};


/**
 * @brief ICP_SVD构造函数
 * @tparam PointT
 */
template<typename PointT>
ICP_SVD<PointT>::ICP_SVD() {
    target_cloud_ptr_ = boost::make_shared<pcl::PointCloud<PointT>>();
    source_cloud_ptr_ = boost::make_shared<pcl::PointCloud<PointT>>();
}


/**
 * @brief ICP_SVD析构函数
 * @tparam PointT
 */
template<typename PointT>
ICP_SVD<PointT>::~ICP_SVD() {}

/**
 * @brief 进行SVD求解
 * @tparam PointT
 */
template<typename PointT>
void ICP_SVD<PointT>::SVDHandle() {

    // 配置target点云的kdtree
    pcl::KdTreeFLANN<PointT> target_KDtree;
    target_KDtree.setInputCloud(target_cloud_ptr_);

    typename pcl::PointCloud<PointT>::Ptr optimized_cloud_ptr(new pcl::PointCloud<PointT>);

    // 开始迭代SVD计算外参
    for (int iter_cnt = 0; iter_cnt < iter_cnts_; ++iter_cnt) {

        // Step1: 计算基于当前外参的source转换后的点云
        Eigen::Matrix4d T_target_source = Eigen::Matrix4d::Identity();
        T_target_source.block<3, 3>(0, 0) = R_target_source_;
        T_target_source.block<3, 1>(0, 3) = t_target_source_;
        pcl::transformPointCloud(*source_cloud_ptr_, *optimized_cloud_ptr, T_target_source);


        // Step2：最近邻查询组织数据
        std::vector<Eigen::Vector3d> P_temp;
        std::vector<Eigen::Vector3d> Q_temp;

        for (size_t i = 0; i < optimized_cloud_ptr->points.size(); ++i) {
            std::vector<int> idx;
            std::vector<float> dist;
            target_KDtree.nearestKSearch(optimized_cloud_ptr->points[i], 1, idx, dist);

            if (dist.front() > nearest_dist_) {
                continue;
            }

            // 存储有效点对
            P_temp.push_back(Eigen::Vector3d(source_cloud_ptr_->points[i].x,
                                             source_cloud_ptr_->points[i].y,
                                             source_cloud_ptr_->points[i].z));
            Q_temp.push_back(Eigen::Vector3d(target_cloud_ptr_->points[idx[0]].x,
                                             target_cloud_ptr_->points[idx[0]].y,
                                             target_cloud_ptr_->points[idx[0]].z));
        }

        Eigen::MatrixXd P(3, P_temp.size());
        Eigen::MatrixXd Q(3, Q_temp.size());
        for (size_t i = 0; i < P_temp.size(); ++i) {
            P.col(i) = P_temp[i];
            Q.col(i) = Q_temp[i];
        }

        // 去质心             计算 Rt
        Eigen::Vector3d p_mean = P.rowwise().mean();
        Eigen::Vector3d q_mean = Q.rowwise().mean();
        Eigen::MatrixXd one_matrix(1, P.cols());
        one_matrix.setOnes();
        auto p_means = p_mean * one_matrix;
        auto q_means = q_mean * one_matrix;
        P = P - p_means;
        Q = Q - q_means;




        // 计算协方差   W = target * source
        Eigen::Matrix3d W = Q * P.transpose();
        Eigen::JacobiSVD<Eigen::Matrix3d> svd(W, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Matrix3d U = svd.matrixU();
        Eigen::Matrix3d V = svd.matrixV();


        // 进行旋转矩阵求解，这里考虑到vUT的行列式小于0的问题，
        // https://www.liuxiao.org/2019/08/%e4%bd%bf%e7%94%a8-svd-%e6%96%b9%e6%b3%95%e6%b1%82%e8%a7%a3-icp-%e9%97%ae%e9%a2%98/
        Eigen::Matrix3d R = Eigen::Matrix3d::Identity();       // 设为对角矩阵初始值
        R(2, 2) = (V * U.transpose()).determinant();          // 调整最后一个元素使行列式为1
        R = V * R * U.transpose();

        // 进行t矩阵计算
        Eigen::Vector3d t = q_mean - R * p_mean;

        // 计算 epsilon
        double epsilon_t = (t - t_target_source_).norm();

        // 更新外参
        R_target_source_ = R;
        t_target_source_ = t;

        // 收件判断
        if (epsilon_t < epsilon_) {
            std::cout << "当前是 " << iter_cnt << "次迭代:" << " delta_t: " << epsilon_t << " < " << epsilon_
                      << std::endl;
            break; // 收敛，停止迭代
        }
    }
}

/**
 * @brief 设置迭代次数
 * @tparam PointT
 * @param iter_cnts
 */
template<typename PointT>
void ICP_SVD<PointT>::setIterCnts(int iter_cnts) {
    iter_cnts_ = iter_cnts;
}

/**
 * @brief 设置收敛阈值
 * @tparam PointT
 * @param epsilon
 */
template<typename PointT>
void ICP_SVD<PointT>::setEpsilon(double epsilon) {
    epsilon_ = epsilon;
}

/**
 * @brief 设置初始外参
 * @tparam PointT
 * @param R_target_source
 * @param t_target_source
 */
template<typename PointT>
void ICP_SVD<PointT>::setInitExternalParam(Eigen::Matrix3d R_target_source, Eigen::Vector3d t_target_source) {
    R_target_source_ = R_target_source;
    t_target_source_ = t_target_source;
}

/**
 * @brief 设置目标点云数据
 * @tparam PointT
 * @param target_cloud_ptr
 */
template<typename PointT>
void ICP_SVD<PointT>::setTargetCloud(const typename pcl::PointCloud<PointT>::Ptr &target_cloud_ptr) {
    target_cloud_ptr_ = target_cloud_ptr;
}

/**
 * @brief 设置source点云数据
 * @tparam PointT
 * @param source_cloud_ptr
 */
template<typename PointT>
void ICP_SVD<PointT>::setSourceCloud(const typename pcl::PointCloud<PointT>::Ptr &source_cloud_ptr) {
    source_cloud_ptr_ = source_cloud_ptr;
}

/**
 * @brief 获取计算的外参
 * @tparam PointT
 * @return
 */
template<typename PointT>
Eigen::Matrix4d ICP_SVD<PointT>::getExternalParam() {
    Eigen::Matrix4d final_T_target_source = Eigen::Matrix4d::Identity();
    final_T_target_source.block<3, 3>(0, 0) = R_target_source_;
    final_T_target_source.block<3, 1>(0, 3) = t_target_source_;
    return final_T_target_source;
}

#endif //EIGEN_ICP_POINT2POINT_SVD_H
