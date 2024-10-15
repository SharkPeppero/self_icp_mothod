//
// Created by xu on 24-9-10.
//

#ifndef EIGEN_ICP_ICP_MOTHED_H
#define EIGEN_ICP_ICP_MOTHED_H


#include <vector>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "pcl/point_types.h"
#include "pcl/point_cloud.h"
#include <pcl/kdtree/kdtree_flann.h>

#include "so3_math.h"

/**
 * @brief 基于SVD的点云配准方法
 */

namespace SVDAligned {
    template<typename PointT>
    class SVDAligned {

    public:
        SVDAligned() = default;

        ~SVDAligned() = default;

        // 配置目标点云
        void setTargetCloud(const typename pcl::PointCloud<PointT>::Ptr &target_cloud_ptr);

        // 配置原始点云
        void setSourceCloud(const typename pcl::PointCloud<PointT>::Ptr &source_cloud_ptr);

        // 配置初始参数
        void setInitPose(const Eigen::Matrix3d &R_target_source, const Eigen::Vector3d &t_target_source);

        // 设置迭代次数（不设置的话默认为2）
        void setMaxIterations(int max_iterations);

        // 设置精度阈值（不设置的话默认为1e-6）
        void setEpsilon(double epsilon);

        // 设置kdtree查询的距离值
        void setNearestDistance(double nearest_dis);

        // Handle函数
        void Handle();

        // 获取最后的结果
        std::pair<Eigen::Matrix3d, Eigen::Vector3d> getOptimizedPose();

    private:

        int max_iterations_ = 10;   // 最大的迭代次数
        double epsilon_ = 1e-6;     // 平移量的epsilon阈值
        double nearest_dist_ = 10;  // 最近邻迭代是否控制最近邻搜索距离

        typename pcl::PointCloud<PointT>::Ptr target_cloud_ptr_ = boost::make_shared<pcl::PointCloud<PointT>>();
        typename pcl::PointCloud<PointT>::Ptr source_cloud_ptr_ = boost::make_shared<pcl::PointCloud<PointT>>();

        Eigen::Matrix3d R_target_source_ = Eigen::Matrix3d::Identity();
        Eigen::Vector3d t_target_source_ = Eigen::Vector3d::Zero();

        bool gt_set_ = false;
        Eigen::Vector3d target_center_;
        Eigen::Vector3d source_center_;

        pcl::KdTreeFLANN<PointT> target_kd_tree_; // target点云的kdtree

    };

// 设置kdtree最近邻查询距离
    template<typename PointT>
    void SVDAligned<PointT>::setNearestDistance(double nearest_dis) {
        nearest_dist_ = nearest_dis;
    }

// SVD求解外参的代码
    template<typename PointT>
    void SVDAligned<PointT>::Handle() {

        typename pcl::PointCloud<PointT>::Ptr optimized_cloud_ptr(new pcl::PointCloud<PointT>);

        for (int iter = 0; iter < max_iterations_; ++iter) {

            // 点云变换
            Eigen::Matrix4d T_target_source = Eigen::Matrix4d::Identity();
            T_target_source.block<3, 3>(0, 0) = R_target_source_;
            T_target_source.block<3, 1>(0, 3) = t_target_source_;
            pcl::transformPointCloud(*source_cloud_ptr_, *optimized_cloud_ptr, T_target_source);


            // STEP1: 进行最近邻查询，组织数据
            std::vector<Eigen::Vector3d> source_vec;
            std::vector<Eigen::Vector3d> target_vec;
            source_vec.reserve(source_cloud_ptr_->points.size());  // 预分配内存
            target_vec.reserve(source_cloud_ptr_->points.size());  // 预分配内存

            for (size_t index = 0; index < source_cloud_ptr_->points.size(); ++index) {

                // 计算最近邻
                int candidate = 1;
                std::vector<int> idx;
                std::vector<float> dist;
                target_kd_tree_.nearestKSearch(optimized_cloud_ptr->points[index], candidate, idx, dist);

                if (dist.front() > nearest_dist_) {
                    continue;
                }

                source_vec.push_back(Eigen::Vector3d(source_cloud_ptr_->points[index].x,
                                                     source_cloud_ptr_->points[index].y,
                                                     source_cloud_ptr_->points[index].z));
                target_vec.push_back(Eigen::Vector3d(target_cloud_ptr_->points[idx[0]].x,
                                                     target_cloud_ptr_->points[idx[0]].y,
                                                     target_cloud_ptr_->points[idx[0]].z));
            }

            // 制作 3xN维度的数据
            Eigen::MatrixXd source_matrix(3, source_vec.size());
            Eigen::MatrixXd target_matrix(3, target_vec.size());
            for (size_t i = 0; i < source_vec.size(); ++i) {
                source_matrix.col(i) = source_vec[i];
                target_matrix.col(i) = target_vec[i];
            }

            // 数据去除质心
            Eigen::Vector3d source_mean = source_matrix.rowwise().mean();
            Eigen::Vector3d target_mean = target_matrix.rowwise().mean();
            source_matrix.colwise() -= source_mean;
            target_matrix.colwise() -= target_mean;

            // 计算协方差   W = target * source.transpose() 关键注意顺序
            Eigen::Matrix3d W = target_matrix * source_matrix.transpose();
            Eigen::JacobiSVD<Eigen::Matrix3d> svd(W, Eigen::ComputeFullU | Eigen::ComputeFullV);
            Eigen::Matrix3d U = svd.matrixU();
            Eigen::Matrix3d V = svd.matrixV();

            // 进行旋转矩阵求解，这里考虑到vUT的行列式小于0的问题，
            // https://www.liuxiao.org/2019/08/%e4%bd%bf%e7%94%a8-svd-%e6%96%b9%e6%b3%95%e6%b1%82%e8%a7%a3-icp-%e9%97%ae%e9%a2%98/
            Eigen::Matrix3d R = Eigen::Matrix3d::Identity();       // 设为对角矩阵初始值
            R(2, 2) = (V * U.transpose()).determinant();          // 调整最后一个元素使行列式为1
            R = V * R * U.transpose();

            // 进行t矩阵计算
            Eigen::Vector3d t = target_mean - R * source_mean;

            // 计算平移量的epsilon
            double cur_epsilon = (t - t_target_source_).norm();
            std::cout << iter << " " << cur_epsilon << std::endl;

            // 更新pose
            R_target_source_ = R;
            t_target_source_ = t;

            // 计算位移量迭代的epsilon

            if (cur_epsilon < epsilon_) {
                break;
            }

        }
    }

// 获取SVD求解的结果
    template<typename PointT>
    std::pair<Eigen::Matrix3d, Eigen::Vector3d> SVDAligned<PointT>::getOptimizedPose() {
        std::pair<Eigen::Matrix3d, Eigen::Vector3d> optimizedPose(R_target_source_, t_target_source_);
        return optimizedPose;
    }

// 设置迭代步长Epsilon阈值
    template<typename PointT>
    void SVDAligned<PointT>::setEpsilon(double epsilon) { epsilon_ = epsilon; }

// 设置最大的迭代次数
    template<typename PointT>
    void SVDAligned<PointT>::setMaxIterations(int max_iterations) { max_iterations_ = max_iterations; }

// 设置迭代的初始位姿
    template<typename PointT>
    void
    SVDAligned<PointT>::setInitPose(const Eigen::Matrix3d &R_target_source, const Eigen::Vector3d &t_target_source) {
        gt_set_ = true;
        R_target_source_ = R_target_source;
        t_target_source_ = t_target_source;
    }

// 设置输入的source点云
    template<typename PointT>
    void SVDAligned<PointT>::setSourceCloud(const typename pcl::PointCloud<PointT>::Ptr &source_cloud_ptr) {
        // 设置原点云数据
        source_cloud_ptr_ = source_cloud_ptr;

        // 检查source_cloud_ptr_是否为空或是否有点
        if (!source_cloud_ptr_ || source_cloud_ptr_->points.empty()) {
            throw std::runtime_error("Source cloud is empty or not set!");
        }

        // 计算source的质心
        source_center_ = std::accumulate(source_cloud_ptr_->points.begin(), source_cloud_ptr_->points.end(),
                                         Eigen::Vector3d::Zero().eval(),
                                         [](const Eigen::Vector3d &c, const PointT &pt) -> Eigen::Vector3d {
                                             return c + Eigen::Vector3d(pt.x, pt.y, pt.z);
                                         }) / static_cast<double>(source_cloud_ptr_->points.size());
    }

// 设置输入的target点云
    template<typename PointT>
    void SVDAligned<PointT>::setTargetCloud(const typename pcl::PointCloud<PointT>::Ptr &target_cloud_ptr) {

        // 设置目标点云
        target_cloud_ptr_ = target_cloud_ptr;

        // 检查 target_cloud_ptr_ 是否为空或是否有点
        if (!target_cloud_ptr_ || target_cloud_ptr_->points.empty()) {
            throw std::runtime_error("Target cloud is empty or not set!");
        }

        // 构建kdtree
        target_kd_tree_.setInputCloud(target_cloud_ptr_);

        // 计算质心
        target_center_ = std::accumulate(target_cloud_ptr_->points.begin(), target_cloud_ptr_->points.end(),
                                         Eigen::Vector3d::Zero().eval(),
                                         [](const Eigen::Vector3d &c, const PointT &pt) -> Eigen::Vector3d {
                                             return c + Eigen::Vector3d(pt.x, pt.y, pt.z);
                                         }) / static_cast<double>(target_cloud_ptr_->points.size());

    }
}




/**
 * @brief 基于点点ICP的点云配准
 */

namespace P2PointAligned {
    template<typename PointT>
    class P2PointAligned {

    public:
        P2PointAligned() = default;

        ~P2PointAligned() = default;

        // 配置目标点云
        void setTargetCloud(const typename pcl::PointCloud<PointT>::Ptr &target_cloud_ptr);

        // 配置原始点云
        void setSourceCloud(const typename pcl::PointCloud<PointT>::Ptr &source_cloud_ptr);

        // 配置初始参数
        void setInitPose(const Eigen::Matrix3d &R_target_source, const Eigen::Vector3d &t_target_source);

        // 设置迭代次数（不设置的话默认为2）
        void setMaxIterations(int max_iterations);

        // 设置精度阈值（不设置的话默认为1e-6）
        void setEpsilon(double epsilon);

        // 设置kdtree查询的距离值
        void setNearestDistance(double nearest_dis);

        // Handle函数
        void Handle();

        // 获取最后的结果
        std::pair<Eigen::Matrix3d, Eigen::Vector3d> getOptimizedPose();

    private:

        int max_iterations_ = 10;   // 最大的迭代次数
        double epsilon_ = 1e-6;     // 平移量的epsilon阈值
        double nearest_dist_ = 1.0;  // 最近邻迭代是否控制最近邻搜索距离

        typename pcl::PointCloud<PointT>::Ptr target_cloud_ptr_ = boost::make_shared<pcl::PointCloud<PointT>>();
        typename pcl::PointCloud<PointT>::Ptr source_cloud_ptr_ = boost::make_shared<pcl::PointCloud<PointT>>();

        Eigen::Matrix3d R_target_source_ = Eigen::Matrix3d::Identity();
        Eigen::Vector3d t_target_source_ = Eigen::Vector3d::Zero();

        bool gt_set_ = false;
        Eigen::Vector3d target_center_;
        Eigen::Vector3d source_center_;

        pcl::KdTreeFLANN<PointT> target_kd_tree_; // target点云的kdtree

    };

    template<typename PointT>
    void P2PointAligned<PointT>::setNearestDistance(double nearest_dis) {
// 设置kdtree最近邻的查询距离
        nearest_dist_ = nearest_dis;
    }

// 点点ICP求解外参的代码
    template<typename PointT>
    void P2PointAligned<PointT>::Handle() {

        // 设置平移初始值
        if (!gt_set_) {
            t_target_source_ = target_center_ - source_center_;
        }

        // 初始化变换后的点云
        typename pcl::PointCloud<PointT>::Ptr optimized_cloud_ptr(new pcl::PointCloud<PointT>);

        double last_cost = std::numeric_limits<double>::max();

        // 进行迭代计算
        for (int iter = 0; iter < max_iterations_; ++iter) {

            // 点云变换
            Eigen::Matrix4d T_target_source = Eigen::Matrix4d::Identity();
            T_target_source.block<3, 3>(0, 0) = R_target_source_;
            T_target_source.block<3, 1>(0, 3) = t_target_source_;
            pcl::transformPointCloud(*source_cloud_ptr_, *optimized_cloud_ptr, T_target_source);

            // STEP1: 进行最近邻查询，组织数据
            std::vector<Eigen::Vector3d> source_vec;
            std::vector<Eigen::Vector3d> target_vec;
            std::vector<Eigen::Vector3d> error_vec;
            source_vec.reserve(source_cloud_ptr_->points.size());  // 预分配内存
            target_vec.reserve(source_cloud_ptr_->points.size());  // 预分配内存
            error_vec.reserve(source_cloud_ptr_->points.size());   // 预分配内存  target - R * source - t

            for (size_t index = 0; index < source_cloud_ptr_->points.size(); ++index) {

                // 计算最近邻
                int candidate = 5;
                std::vector<int> idx;
                std::vector<float> dist;
                target_kd_tree_.nearestKSearch(optimized_cloud_ptr->points[index], candidate, idx, dist);

                if (dist.empty() || dist.front() > nearest_dist_) {
                    continue;
                }

                source_vec.emplace_back(source_cloud_ptr_->points[index].x, source_cloud_ptr_->points[index].y,
                                        source_cloud_ptr_->points[index].z);
                target_vec.emplace_back(target_cloud_ptr_->points[idx[0]].x, target_cloud_ptr_->points[idx[0]].y,
                                        target_cloud_ptr_->points[idx[0]].z);
                error_vec.emplace_back(target_cloud_ptr_->points[idx[0]].x - optimized_cloud_ptr->points[index].x,
                                       target_cloud_ptr_->points[idx[0]].y - optimized_cloud_ptr->points[index].y,
                                       target_cloud_ptr_->points[idx[0]].z - optimized_cloud_ptr->points[index].z);
            }

            // 组织 J J^T delta_x = -1 * J * e

            Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
            Eigen::Matrix<double, 6, 1> b = Eigen::Matrix<double, 6, 1>::Zero();
            double total_cost = 0.0;
            for (size_t index = 0; index < source_vec.size(); ++index) {
                // 计算error = target - R * source - t
                Eigen::Vector3d error = error_vec[index];

                Eigen::Matrix<double, 6, 3> J;
                J.block<3, 3>(0, 0) = R_target_source_ * SO3Math::skew_sym_mat(source_vec[index]);
                J.block<3, 3>(3, 0) = -1.0 * Eigen::Matrix3d::Identity();

                H += J * J.transpose();
                b += -1 * J * error;

                total_cost = error.dot(error);
            }

            // 进行求解
            double lambda = 1e-6;  // 正则化参数
            Eigen::MatrixXd H_reg = H + lambda * Eigen::MatrixXd::Identity(H.rows(), H.cols());
            Eigen::VectorXd dx = H_reg.colPivHouseholderQr().solve(b);


            // 先判断本次结果是否有效
            //  计算当前的 cur_cost，判断本次和上次的残差关系
            total_cost = std::sqrt(total_cost / source_vec.size());


            std::cout << "iter: " << iter << " total_cost: " << total_cost << " epsilon: " << dx.tail<3>().norm()
                      << std::endl;

            if (total_cost > last_cost) {
                std::cout << "total_cost: " << total_cost << " > " << " last_cost: " << last_cost << std::endl;
                break;
            }
            last_cost = total_cost;

            // 更新Pose
            R_target_source_ = R_target_source_ * SO3Math::Exp<double>(dx.head<3>());
            t_target_source_ += dx.tail<3>();

            // 计算 迭代的步长是否满足epsilon
            if (dx.tail<3>().norm() < epsilon_) {
                break;
            }


        }
    }

// 获取点点ICP求解的结果
    template<typename PointT>
    std::pair<Eigen::Matrix3d, Eigen::Vector3d> P2PointAligned<PointT>::getOptimizedPose() {
        std::pair<Eigen::Matrix3d, Eigen::Vector3d> optimizedPose(R_target_source_, t_target_source_);
        return optimizedPose;
    }

// 设置迭代步长Epsilon阈值
    template<typename PointT>
    void P2PointAligned<PointT>::setEpsilon(double epsilon) { epsilon_ = epsilon; }

// 设置最大的迭代次数
    template<typename PointT>
    void P2PointAligned<PointT>::setMaxIterations(int max_iterations) { max_iterations_ = max_iterations; }

// 设置迭代的初始位姿
    template<typename PointT>
    void
    P2PointAligned<PointT>::setInitPose(const Eigen::Matrix3d &R_target_source,
                                        const Eigen::Vector3d &t_target_source) {
        gt_set_ = true;
        R_target_source_ = R_target_source;
        t_target_source_ = t_target_source;
    }

// 设置输入的source点云
    template<typename PointT>
    void P2PointAligned<PointT>::setSourceCloud(const typename pcl::PointCloud<PointT>::Ptr &source_cloud_ptr) {
        // 设置原点云数据
        source_cloud_ptr_ = source_cloud_ptr;

        // 检查source_cloud_ptr_是否为空或是否有点
        if (!source_cloud_ptr_ || source_cloud_ptr_->points.empty()) {
            throw std::runtime_error("Source cloud is empty or not set!");
        }

        // 计算source的质心
        source_center_ = std::accumulate(source_cloud_ptr_->points.begin(), source_cloud_ptr_->points.end(),
                                         Eigen::Vector3d::Zero().eval(),
                                         [](const Eigen::Vector3d &c, const PointT &pt) -> Eigen::Vector3d {
                                             return c + Eigen::Vector3d(pt.x, pt.y, pt.z);
                                         }) / static_cast<double>(source_cloud_ptr_->points.size());
    }

// 设置输入的target点云
    template<typename PointT>
    void P2PointAligned<PointT>::setTargetCloud(const typename pcl::PointCloud<PointT>::Ptr &target_cloud_ptr) {

        // 设置目标点云
        target_cloud_ptr_ = target_cloud_ptr;

        // 检查 target_cloud_ptr_ 是否为空或是否有点
        if (!target_cloud_ptr_ || target_cloud_ptr_->points.empty()) {
            throw std::runtime_error("Target cloud is empty or not set!");
        }

        // 构建kdtree
        target_kd_tree_.setInputCloud(target_cloud_ptr_);

        // 计算质心
        target_center_ = std::accumulate(target_cloud_ptr_->points.begin(), target_cloud_ptr_->points.end(),
                                         Eigen::Vector3d::Zero().eval(),
                                         [](const Eigen::Vector3d &c, const PointT &pt) -> Eigen::Vector3d {
                                             return c + Eigen::Vector3d(pt.x, pt.y, pt.z);
                                         }) / static_cast<double>(target_cloud_ptr_->points.size());

    }
}


/**
 * @brief 基于点线ICP的点云配准
 */

namespace P2LineAligned {
    template<typename PointT>
    class P2LineAligned {

    public:
        P2LineAligned() = default;

        ~P2LineAligned() = default;

        // 配置目标点云
        void setTargetCloud(const typename pcl::PointCloud<PointT>::Ptr &target_cloud_ptr);

        // 配置原始点云
        void setSourceCloud(const typename pcl::PointCloud<PointT>::Ptr &source_cloud_ptr);

        // 配置初始参数
        void setInitPose(const Eigen::Matrix3d &R_target_source, const Eigen::Vector3d &t_target_source);

        // 设置迭代次数（不设置的话默认为2）
        void setMaxIterations(int max_iterations);

        // 设置精度阈值（不设置的话默认为1e-6）
        void setEpsilon(double epsilon);

        // 设置kdtree查询的距离值
        void setNearestDistance(double nearest_dis);

        // 设置直线拟合的参数阈值
        void setLineFitEpsilon(double line_fit_epsilon);

        // Handle函数
        void Handle();

        // 给定点云拟合直线参数
        template<typename S>
        bool fitLine(std::vector<Eigen::Matrix<S, 3, 1>> &data,
                     Eigen::Matrix<S, 3, 1> &origin,
                     Eigen::Matrix<S, 3, 1> &dir,
                     double eps = 0.2);

        // 获取最后的结果
        std::pair<Eigen::Matrix3d, Eigen::Vector3d> getOptimizedPose();

    private:

        int max_iterations_ = 10;   // 最大的迭代次数
        double epsilon_ = 1e-6;     // 平移量的epsilon阈值
        double nearest_dist_ = 1.0;  // 最近邻迭代是否控制最近邻搜索距离
        double line_fit_epsilon_ = 0.1; // 直线拟合的参数

        typename pcl::PointCloud<PointT>::Ptr target_cloud_ptr_ = boost::make_shared<pcl::PointCloud<PointT>>();
        typename pcl::PointCloud<PointT>::Ptr source_cloud_ptr_ = boost::make_shared<pcl::PointCloud<PointT>>();

        Eigen::Matrix3d R_target_source_ = Eigen::Matrix3d::Identity();
        Eigen::Vector3d t_target_source_ = Eigen::Vector3d::Zero();

        bool gt_set_ = false;
        Eigen::Vector3d target_center_;
        Eigen::Vector3d source_center_;

        pcl::KdTreeFLANN<PointT> target_kd_tree_; // target点云的kdtree

    };

    // 直线参考点 origin 以及 直线方向向量计算
    template<typename PointT>
    template<typename S>
    bool P2LineAligned<PointT>::fitLine(std::vector<Eigen::Matrix<S, 3, 1>> &data,
                                        Eigen::Matrix<S, 3, 1> &origin,
                                        Eigen::Matrix<S, 3, 1> &dir,
                                        double eps) {

        // 判断点的数目必须大于2
        if (data.size() < 2) {
            return false;
        }

        // 计算质心
        origin = std::accumulate(data.begin(), data.end(), Eigen::Matrix<S, 3, 1>::Zero().eval()) / data.size();

        // SVD分解 Nx3的矩阵 Y
        Eigen::MatrixXd Y(data.size(), 3);
        for (int i = 0; i < data.size(); ++i) {
            Y.row(i) = (data[i] - origin).transpose();
        }

        // 对Y进行SVD分解
        Eigen::JacobiSVD svd(Y, Eigen::ComputeFullV);
        dir = svd.matrixV().col(0);

        // check eps
        for (const auto &d: data) {
            if (dir.template cross(d - origin).
                    template squaredNorm() > eps) {
                return false;
            }
        }

        return true;
    }

    // 点线ICP求解外参的代码
    template<typename PointT>
    void P2LineAligned<PointT>::Handle() {

        // 设置平移初始值
        if (!gt_set_) {
            t_target_source_ = target_center_ - source_center_;
        }

        // 初始化变换后的点云
        typename pcl::PointCloud<PointT>::Ptr optimized_cloud_ptr(new pcl::PointCloud<PointT>);

        double last_cost = std::numeric_limits<double>::max();

        // 进行迭代计算
        for (int iter = 0; iter < max_iterations_; ++iter) {

            // 点云变换
            Eigen::Matrix4d T_target_source = Eigen::Matrix4d::Identity();
            T_target_source.block<3, 3>(0, 0) = R_target_source_;
            T_target_source.block<3, 1>(0, 3) = t_target_source_;
            pcl::transformPointCloud(*source_cloud_ptr_, *optimized_cloud_ptr, T_target_source);

            // STEP1: 进行最近邻查询，组织数据
            std::vector<Eigen::Vector3d> source_vec;
            std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> line_param_vec;
            std::vector<Eigen::Vector3d> error_vec;
            source_vec.reserve(source_cloud_ptr_->points.size());
            line_param_vec.reserve(source_cloud_ptr_->points.size());
            error_vec.reserve(source_cloud_ptr_->points.size());

            for (size_t index = 0; index < source_cloud_ptr_->points.size(); ++index) {

                // 计算最近邻
                int candidate = 5;
                std::vector<int> idx;
                std::vector<float> dist;
                target_kd_tree_.nearestKSearch(optimized_cloud_ptr->points[index], candidate, idx, dist);
                if (dist.size() < candidate || dist.back() > nearest_dist_) {
                    continue;
                }

                // 计算直线参数
                //      组织target的点云信息
                std::vector<Eigen::Vector3d> points;
                for (int & index : idx) {
                    Eigen::Vector3d target_point;
                    target_point << target_cloud_ptr_->points[index].x, target_cloud_ptr_->points[index].y, target_cloud_ptr_->points[index].z;
                    points.push_back(target_point);
                }

                //      计算拟合直线的参数
                Eigen::Vector3d origin, direc_vec;
                if(!fitLine(points, origin, direc_vec, line_fit_epsilon_)){
                    continue;
                }

                // 计算点到直线距离的误差
                // e = direc_vec 叉乘 (ps - origin)
                Eigen::Vector3d p_transform = Eigen::Vector3d(optimized_cloud_ptr->points[index].x, optimized_cloud_ptr->points[index].y, optimized_cloud_ptr->points[index].z);
                Eigen::Vector3d error = SO3Math::skew_sym_mat(direc_vec) * (p_transform - origin);
                if(error.norm() > line_fit_epsilon_){
                    continue;
                }

                // 存储相关参数
                source_vec.emplace_back(source_cloud_ptr_->points[index].x, source_cloud_ptr_->points[index].y, source_cloud_ptr_->points[index].z);
                line_param_vec.emplace_back(origin, direc_vec);
                error_vec.emplace_back(error);
            }

            // 组织 J J^T delta_x = -1 * J * e

            Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
            Eigen::Matrix<double, 6, 1> b = Eigen::Matrix<double, 6, 1>::Zero();
            double total_cost = 0.0;
            for (size_t index = 0; index < source_vec.size(); ++index) {

                // 计算error = target - R * source - t
                Eigen::Vector3d error = error_vec[index];

                // e / r =
                Eigen::Matrix<double, 6, 3> J;
                J.block<3, 3>(0, 0) = -1 * SO3Math::Exp(line_param_vec[index].second) * R_target_source_ * SO3Math::skew_sym_mat(source_vec[index]);
                J.block<3, 3>(3, 0) = SO3Math::Exp(line_param_vec[index].second);

                H += J * J.transpose();
                b += -1 * J * error;

                total_cost = error.dot(error);
            }

            // 进行求解
            double lambda = 1e-6;  // 正则化参数
            Eigen::MatrixXd H_reg = H + lambda * Eigen::MatrixXd::Identity(H.rows(), H.cols());
            Eigen::VectorXd dx = H_reg.colPivHouseholderQr().solve(b);


            // 先判断本次结果是否有效
            //  计算当前的 cur_cost，判断本次和上次的残差关系
            total_cost = std::sqrt(total_cost / source_vec.size());


            std::cout << "iter: " << iter << " total_cost: " << total_cost << " epsilon: " << dx.tail<3>().norm()
                      << std::endl;

            if (total_cost > last_cost) {
                std::cout << "total_cost: " << total_cost << " > " << " last_cost: " << last_cost << std::endl;
                break;
            }
            last_cost = total_cost;

            // 更新Pose
            R_target_source_ = R_target_source_ * SO3Math::Exp<double>(dx.head<3>());
            t_target_source_ += dx.tail<3>();

            // 计算 迭代的步长是否满足epsilon
            if (dx.tail<3>().norm() < epsilon_) {
                break;
            }


        }
    }

    // 获取点点ICP求解的结果
    template<typename PointT>
    std::pair<Eigen::Matrix3d, Eigen::Vector3d> P2LineAligned<PointT>::getOptimizedPose() {
        std::pair<Eigen::Matrix3d, Eigen::Vector3d> optimizedPose(R_target_source_, t_target_source_);
        return optimizedPose;
    }


    // 设置直线拟合的epsilon
    template<typename PointT>
    void P2LineAligned<PointT>::setLineFitEpsilon(double line_fit_epsilon) {
        line_fit_epsilon_ = line_fit_epsilon;
    }

    // 设置迭代步长Epsilon阈值
    template<typename PointT>
    void P2LineAligned<PointT>::setEpsilon(double epsilon) { epsilon_ = epsilon; }

    // 设置最大的迭代次数
    template<typename PointT>
    void P2LineAligned<PointT>::setMaxIterations(int max_iterations) { max_iterations_ = max_iterations; }

    // 设置迭代的初始位姿
    template<typename PointT>
    void
    P2LineAligned<PointT>::setInitPose(const Eigen::Matrix3d &R_target_source, const Eigen::Vector3d &t_target_source) {
        gt_set_ = true;
        R_target_source_ = R_target_source;
        t_target_source_ = t_target_source;
    }

    // 设置输入的source点云
    template<typename PointT>
    void P2LineAligned<PointT>::setSourceCloud(const typename pcl::PointCloud<PointT>::Ptr &source_cloud_ptr) {
        // 设置原点云数据
        source_cloud_ptr_ = source_cloud_ptr;

        // 检查source_cloud_ptr_是否为空或是否有点
        if (!source_cloud_ptr_ || source_cloud_ptr_->points.empty()) {
            throw std::runtime_error("Source cloud is empty or not set!");
        }

        // 计算source的质心
        source_center_ = std::accumulate(source_cloud_ptr_->points.begin(), source_cloud_ptr_->points.end(),
                                         Eigen::Vector3d::Zero().eval(),
                                         [](const Eigen::Vector3d &c, const PointT &pt) -> Eigen::Vector3d {
                                             return c + Eigen::Vector3d(pt.x, pt.y, pt.z);
                                         }) / static_cast<double>(source_cloud_ptr_->points.size());
    }

    // 设置输入的target点云
    template<typename PointT>
    void P2LineAligned<PointT>::setTargetCloud(const typename pcl::PointCloud<PointT>::Ptr &target_cloud_ptr) {

        // 设置目标点云
        target_cloud_ptr_ = target_cloud_ptr;

        // 检查 target_cloud_ptr_ 是否为空或是否有点
        if (!target_cloud_ptr_ || target_cloud_ptr_->points.empty()) {
            throw std::runtime_error("Target cloud is empty or not set!");
        }

        // 构建kdtree
        target_kd_tree_.setInputCloud(target_cloud_ptr_);

        // 计算质心
        target_center_ = std::accumulate(target_cloud_ptr_->points.begin(), target_cloud_ptr_->points.end(),
                                         Eigen::Vector3d::Zero().eval(),
                                         [](const Eigen::Vector3d &c, const PointT &pt) -> Eigen::Vector3d {
                                             return c + Eigen::Vector3d(pt.x, pt.y, pt.z);
                                         }) / static_cast<double>(target_cloud_ptr_->points.size());

    }

    // kdtree查询的最近邻距离阈值
    template<typename PointT>
    void P2LineAligned<PointT>::setNearestDistance(double nearest_dis) {
        nearest_dist_ = nearest_dis;
    }

}



#endif //EIGEN_ICP_ICP_MOTHED_H
