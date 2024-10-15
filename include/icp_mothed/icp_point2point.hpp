//
// Created by xu on 24-9-9.
//

#ifndef EIGEN_ICP_ICP_POINT2POINT_H
#define EIGEN_ICP_ICP_POINT2POINT_H

#include <execution>

#include "so3_math.h"

#include "Eigen/Core"
#include "Eigen/Dense"
#include "Eigen/Geometry"

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/transforms.h>
#include <pcl/kdtree/kdtree_flann.h>


/**
 * @brief GN实现点点非线性优化
 * @tparam PointT
 */
template<typename PointT>
class ICP_Point2Point {
public:

    // 构造函数
    ICP_Point2Point();

    // 析构函数
    ~ICP_Point2Point();

    // 点2点的GN非线性优化计算外参
    void Point2PointHandle();

    // 设置目标点云数据,并构建kdtree
    void setTargetCloud(const typename pcl::PointCloud<PointT>::Ptr &target_cloud_ptr);

    // 设置source点云数据
    void setSourceCloud(const typename pcl::PointCloud<PointT>::Ptr &source_cloud_ptr);

    // 设置迭代次数（不设置的话默认为2）
    void setIterCnts(int iter_cnts);

    // 设置精度阈值（不设置的话默认为1e-6）
    void setEpsilon(double epsilon);

    // 设置kdtree查询的最近邻
    void setNearestDistance(double nearest_distance);

    // 设置初始外参（不设置的话默认初值）
    void setInitExternalParam(Eigen::Matrix3d R_target_source, Eigen::Vector3d t_target_source);

    // 获取计算的外参
    Eigen::Matrix4d getExternalParam();

    // 打印参数
    void logParam();


private:
    int iter_cnts_ = 5;             // 最大的迭代次数
    double epsilon_ = 1e-6;         // 平移量的epsilon阈值
    double nearest_dist_ = 1.0;     // 最近邻迭代是否控制最近邻搜索距离
    double last_cost_ = std::numeric_limits<double>::max(); // 设置上一次的cost

    typename pcl::PointCloud<PointT>::Ptr target_cloud_ptr_ = nullptr;
    typename pcl::PointCloud<PointT>::Ptr source_cloud_ptr_ = nullptr;

    bool gt_set_flag_ = false;
    Eigen::Matrix3d R_target_source_ = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t_target_source_ = Eigen::Vector3d::Zero();

    pcl::KdTreeFLANN<PointT> target_KDtree_;
    Eigen::Vector3d target_center_;
    Eigen::Vector3d source_center_;
};



template<typename PointT>
ICP_Point2Point<PointT>::ICP_Point2Point() {
    target_cloud_ptr_ = boost::make_shared<pcl::PointCloud<PointT>>();
    source_cloud_ptr_ = boost::make_shared<pcl::PointCloud<PointT>>();
}

template<typename PointT>
ICP_Point2Point<PointT>::~ICP_Point2Point() {

}

/*
template<typename PointT>
void ICP_Point2Point<PointT>::Point2PointHandle() {

    // 如果没有传入优化初值，将质心的偏移量作为平移的初值
    if(!gt_set_flag_){
        std::cout << "没有初始位姿，计算点云质心，作为平移的初始外参" << std::endl;
        t_target_source_ = target_center_ - source_center_;
    }


    for (int iter = 0; iter < iter_cnts_; iter++) {



        // 对source中的每一个点进行变换，并开始最近邻查询

        std::vector<Eigen::Vector3d> source_points;
        std::vector<Eigen::Vector3d> target_points;
        std::vector<Eigen::Vector3d> errors;
        for(int index = 0; index < source_cloud_ptr_->points.size(); index++){

            // source中的每一个点
            PointT point = source_cloud_ptr_->points[index];

            // 进行变换
            Eigen::Vector3d transformed_point = R_target_source_ * Eigen::Vector3d(point.x, point.y, point.z) + t_target_source_;

            // 进行最近邻查询
            PointT pcl_point;
            pcl_point.x = transformed_point.x();
            pcl_point.y = transformed_point.y();
            pcl_point.z = transformed_point.z();

            std::vector<int> idx;
            std::vector<float> dist;
            int num_found = target_KDtree_.nearestKSearch(pcl_point, 1, idx, dist);

            // 确保找到了最近邻点 判断dist是否大于阈值
            if (num_found == 0 || dist.front() > nearest_dist_) {
                continue;
            }

            // 存储有效点对
            source_points.push_back(Eigen::Vector3d(point.x, point.y, point.z));
            target_points.push_back(Eigen::Vector3d(target_cloud_ptr_->points[idx[0]].x, target_cloud_ptr_->points[idx[0]].y, target_cloud_ptr_->points[idx[0]].z));

            // 计算 error = target - R * source - t
            Eigen::Vector3d error = Eigen::Vector3d(target_cloud_ptr_->points[idx[0]].x, target_cloud_ptr_->points[idx[0]].y, target_cloud_ptr_->points[idx[0]].z) -
                                    transformed_point;
            errors.push_back(error);
        }

        // Step2: 组织GN求解 JJ^T delta_x = - J * e

        // 进行非线性的GN求解
        // J * JT * delta_x = -1 * J * ei
        // e_i = target_i - (R * source_i + t)

        // 计算残差与旋转以及平移的雅阁比矩阵
        // J_rotation = R * pi^
        // J_translation = -I
//
//                     -1  0   0
//        J = R * pi^   0  -1  0      3行6列
//                      0  0  -1
//

//         雅阁比是列向量
//         * J = f / x1
//         *     f / x2
//         *     f/ x3
//

        Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
        Eigen::Matrix<double, 6, 1> b = Eigen::Matrix<double, 6, 1>::Zero();
        double cur_cost = 0.0;

        // 构建残差
        for (int index = 0; index < source_points.size(); index++) {

            // 计算当前的 ei 3x1
            Eigen::Vector3d error = errors[index];

            // 计算当前的 Ji
            Eigen::Matrix<double, 6, 3> J;
            J.block<3, 3>(0, 0) = R_target_source_ * SO3Math::skew_sym_mat(source_points[index]);
            J.block<3, 3>(3, 0) = -1 * Eigen::Matrix3d::Identity();

            // 计算H、b以及总cost
            //  H = JT * J
            //  b = -1 * J * error
            H += J * J.transpose();
            b += -1 * J * error;
            cur_cost += error.dot(error);
        }

        // 利用 cost 判断是否可以退出迭代
        cur_cost = cur_cost * 1.0 / source_points.size();
        if(cur_cost > last_cost_){
            break;
        }
        last_cost_ = cur_cost;
        std::cout << "iter: " << iter << " cur_cost:" << cur_cost << std::endl;


        // 求解线性方程 H * delta_x = b
        Eigen::Matrix<double, 6, 1> delta_x = H.ldlt().solve(b);
        // 利用delta_x 的下降步长来控制
        if(delta_x.norm() < 1e-4){
            break;
        }


        // 更新部分
        //  这里与上面求导的顺序保持一致，因为上面求导是右乘
        R_target_source_ = R_target_source_ * SO3Math::Exp<double>(delta_x.head<3>());
        t_target_source_ = t_target_source_ + delta_x.tail<3>();

    }

}
*/

template<typename PointT>
void ICP_Point2Point<PointT>::Point2PointHandle(){

    if (!gt_set_flag_) {
        t_target_source_ = target_center_ - source_center_;  // 设置平移初始值
    }

    std::vector<int> index(source_cloud_ptr_->points.size());
    for (int i = 0; i < index.size(); ++i) {
        index[i] = i;
    }

    std::vector<bool> effect_pts(index.size(), false);
    std::vector<Eigen::Matrix<double, 3, 6>> jacobians(index.size());
    std::vector<Eigen::Vector3d> errors(index.size());

    for (int iter = 0; iter < iter_cnts_; ++iter) {
        // gauss-newton 迭代
        // 最近邻，串行处理
        for (int idx : index) {
            Eigen::Vector3d q = Eigen::Vector3d(source_cloud_ptr_->points[idx].x,
                                                source_cloud_ptr_->points[idx].y,
                                                source_cloud_ptr_->points[idx].z);
            Eigen::Vector3d qs = R_target_source_ * q + t_target_source_;  // 转换之后的q

            PointT point_pcl;
            point_pcl.x = qs.x();
            point_pcl.y = qs.y();
            point_pcl.z = qs.z();

            std::vector<int> nn;
            std::vector<float> dist;
            int num_found = target_KDtree_.nearestKSearch(point_pcl, 1, nn, dist);

            if (!nn.empty()) {
                Eigen::Vector3d p = Eigen::Vector3d(target_cloud_ptr_->points[nn[0]].x,
                                                    target_cloud_ptr_->points[nn[0]].y,
                                                    target_cloud_ptr_->points[nn[0]].z);
                double dis2 = (p - qs).squaredNorm();
                if (dis2 > nearest_dist_) {
                    // 点离的太远了不要
                    effect_pts[idx] = false;
                    continue;
                }

                effect_pts[idx] = true;

                // build residual
                Eigen::Vector3d e = p - qs;
                Eigen::Matrix<double, 3, 6> J;
                J.block<3, 3>(0, 0) = R_target_source_ * SO3Math::skew_sym_mat(q);
                J.block<3, 3>(0, 3) = -Eigen::Matrix3d::Identity();

                jacobians[idx] = J;
                errors[idx] = e;
            } else {
                effect_pts[idx] = false;
            }

        }

        // 累加Hessian和error,计算dx
        double total_res = 0;
        int effective_num = 0;
        auto H_and_err = std::accumulate(
                index.begin(), index.end(),
                std::pair<Eigen::Matrix<double, 6, 6>, Eigen::Matrix<double, 6, 1>>(Eigen::Matrix<double, 6, 6>::Zero(), Eigen::Matrix<double, 6, 1>::Zero()),
                [&jacobians, &errors, &effect_pts, &total_res, &effective_num](const std::pair<Eigen::Matrix<double, 6, 6>, Eigen::Matrix<double, 6, 1>> &pre,
                                                                               int idx) -> std::pair<Eigen::Matrix<double, 6, 6>, Eigen::Matrix<double, 6, 1>> {
                    if (!effect_pts[idx]) {
                        return pre;
                    } else {

                        total_res += errors[idx].dot(errors[idx]);

                        effective_num++;

                        return std::pair<Eigen::Matrix<double, 6, 6>, Eigen::Matrix<double, 6, 1>>(
                                pre.first + jacobians[idx].transpose() * jacobians[idx],
                                pre.second - jacobians[idx].transpose() * errors[idx]);


                    }
                });


        Eigen::Matrix<double, 6, 6> H = H_and_err.first;
        Eigen::Matrix<double, 6, 1> err = H_and_err.second;

        Eigen::Matrix<double, 6, 1> dx = H.inverse() * err;
        R_target_source_ = R_target_source_ * SO3Math::Exp<double>(dx.head<3>());
        t_target_source_ += dx.tail<3>();

    }

}



// 配置目标点云
template<typename PointT>
void ICP_Point2Point<PointT>::setTargetCloud(const typename pcl::PointCloud<PointT>::Ptr &target_cloud_ptr) {

    // 设置目标点云
    target_cloud_ptr_ = target_cloud_ptr;

    // 检查 target_cloud_ptr_ 是否为空或是否有点
    if (!target_cloud_ptr_ || target_cloud_ptr_->points.empty()) {
        throw std::runtime_error("Target cloud is empty or not set!");
    }

    // 构建kdtree
    target_KDtree_.setInputCloud(target_cloud_ptr_);

    // 计算质心
    target_center_ = std::accumulate(target_cloud_ptr_->points.begin(), target_cloud_ptr_->points.end(),
                                     Eigen::Vector3d::Zero().eval(),
                                     [](const Eigen::Vector3d &c, const PointT &pt) -> Eigen::Vector3d {
                                         return c + Eigen::Vector3d(pt.x, pt.y, pt.z);
                                     }) / static_cast<double>(target_cloud_ptr_->points.size());

}

// 配置source点云
template<typename PointT>
void ICP_Point2Point<PointT>::setSourceCloud(const typename pcl::PointCloud<PointT>::Ptr &source_cloud_ptr) {
    // 设置原点云数据
    source_cloud_ptr_ = source_cloud_ptr;

    // 检查source_cloud_ptr_是否为空或是否有点
    if (!source_cloud_ptr_ || source_cloud_ptr_->points.empty()) {
        throw std::runtime_error("Source cloud is empty or not set!");
    }

    // 计算source的质心
    source_center_ = std::accumulate(source_cloud_ptr_->points.begin(), source_cloud_ptr_->points.end(), Eigen::Vector3d::Zero().eval(),
                                     [](const Eigen::Vector3d &c, const PointT &pt) -> Eigen::Vector3d {
                                         return c + Eigen::Vector3d(pt.x, pt.y, pt.z);
                                     }) / static_cast<double>(source_cloud_ptr_->points.size());

}


// 配置初始外参
template<typename PointT>
void ICP_Point2Point<PointT>::setInitExternalParam(Eigen::Matrix3d R_target_source, Eigen::Vector3d t_target_source) {
    gt_set_flag_ = true;
    R_target_source_ = R_target_source;
    t_target_source_ = t_target_source;
}

// 设置epsilon参数
template<typename PointT>
void ICP_Point2Point<PointT>::setEpsilon(double epsilon) {
    epsilon_ = epsilon;
}

// 配置迭代次数
template<typename PointT>
void ICP_Point2Point<PointT>::setIterCnts(int iter_cnts) {
    iter_cnts_ = iter_cnts;
}

// 设置kdtree最近邻查询
template<typename PointT>
void ICP_Point2Point<PointT>::setNearestDistance(double nearest_distance) {
    nearest_dist_ = nearest_distance;
}

// 进行参数打印
template<typename PointT>
void ICP_Point2Point<PointT>::logParam() {
    std::cout << "icp_point2point Param: " << std::endl;
    std::cout << "\titer_cnts: " << iter_cnts_ << std::endl <<
                 "\tepsilon: " << epsilon_ << std::endl <<
                 "\tnearest_distance: " << nearest_dist_ << std::endl <<
                 "\ttarget_center: " << target_center_.transpose() << std::endl <<
                 "\tsource_center: " << source_center_.transpose() << std::endl;
}

// 获取优化后的外参
template<typename PointT>
Eigen::Matrix4d ICP_Point2Point<PointT>::getExternalParam() {
    Eigen::Matrix4d final_T_target_source = Eigen::Matrix4d::Identity();
    final_T_target_source.block<3, 3>(0, 0) = R_target_source_;
    final_T_target_source.block<3, 1>(0, 3) = t_target_source_;
    return final_T_target_source;
}


#endif //EIGEN_ICP_ICP_POINT2POINT_H
