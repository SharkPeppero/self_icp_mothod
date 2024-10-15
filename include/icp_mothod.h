//
// Created by xu on 24-10-12.
//

#ifndef EIGEN_ICP_ICP_MOTHOD_H
#define EIGEN_ICP_ICP_MOTHOD_H

#include "Eigen/Core"
#include "Eigen/Dense"
#include "Eigen/Geometry"

#include "pcl/point_types.h"
#include "pcl/point_cloud.h"
#include <pcl/kdtree/kdtree_flann.h>
#include "pcl/common/transforms.h"

#include "math.h"

#include "common.h"

// TBB并行
#include <tbb/tbb.h>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>


template<typename PointT>
class ICP_SVD {
public:

    ICP_SVD() = default;

    ~ICP_SVD()=default;

    /// 设置Target点云
    void setTargetCloud(const typename pcl::PointCloud<PointT>::Ptr &target_cloud_ptr){

        target_cloud_ptr_ = target_cloud_ptr;

        target_center_ = std::accumulate(target_cloud_ptr_->points.begin(),
                                         target_cloud_ptr_->points.end(),
                                         Eigen::Vector3d::Zero().eval(),
                                         [](const Eigen::Vector3d &c, const PointT &pt) -> Eigen::Vector3d {
                                             return c + Eigen::Vector3d(pt.x, pt.y, pt.z);
                                         });
        target_center_ /= static_cast<double>(target_cloud_ptr_->points.size());

        target_kd_tree_.setInputCloud(target_cloud_ptr_);

        if(log_flag_){
            std::cout << "target size: " << target_cloud_ptr_->points.size() << std::endl;
            std::cout << "target center: " << target_center_.transpose() << std::endl;
        }

    }

    /// 设置Source点云
    void setSourceCloud(const typename pcl::PointCloud<PointT>::Ptr &source_cloud_ptr) {

        source_cloud_ptr_ = source_cloud_ptr;

        source_center_ = std::accumulate(source_cloud_ptr_->points.begin(),
                                         source_cloud_ptr_->points.end(),
                                         Eigen::Vector3d::Zero().eval(),
                                         [](const Eigen::Vector3d &c, const PointT &pt) -> Eigen::Vector3d {
                                             return c + Eigen::Vector3d(pt.x, pt.y, pt.z);
                                         });
        source_center_ /= static_cast<double>(source_cloud_ptr_->points.size());

        if(log_flag_){
            std::cout << "source size: " << source_cloud_ptr_->points.size() << std::endl;
            std::cout << "source center: " << source_center_.transpose() << std::endl;
        }

    }

    /**
     * source点云基于初始外参变换成 transformed_cloud
     * transformed_cloud与target_cloud进行kdtree查询
     * ps: 组织 !!!原始点云!!! 数据的矩阵 source_matrix target_matrix
     * 组织协方差矩阵 C = target_matrix * source_matrix.transform()
     * 进行SVD分解 M_cov = U * sig * VT
     * 其中旋转为： R_target_source = V * UT
     * 平移量 t_target_source = C_target - R_target_source * C_source
     */
    void Aligned(){

        typename pcl::PointCloud<PointT>::Ptr transformed_cloud(new pcl::PointCloud<PointT>);

        for (int iter = 0; iter < max_iterations_; ++iter) {

            TicToc solver;

            // STEP1: 点云变换
            pcl::transformPointCloud(*source_cloud_ptr_, *transformed_cloud, T_target_source_);

            // STEP2: 进行最近邻查询，组织 target_3xn * source_3xn.transpose()
            std::vector<Eigen::Vector3d> source_vec;
            std::vector<Eigen::Vector3d> target_vec;
            source_vec.reserve(source_cloud_ptr_->points.size());  // 预分配内存
            target_vec.reserve(source_cloud_ptr_->points.size());  // 预分配内存
            for (size_t index = 0; index < source_cloud_ptr_->points.size(); ++index) {

                // 计算最近邻
                int candidate = 1;
                std::vector<int> idx;
                std::vector<float> dist;
                target_kd_tree_.nearestKSearch(transformed_cloud->points[index], candidate, idx, dist);

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

            assert(source_vec.size() == target_vec.size());
            Eigen::MatrixXd source_matrix(3, source_vec.size());
            Eigen::MatrixXd target_matrix(3, target_vec.size());
            for (size_t i = 0; i < source_vec.size(); ++i) {
                source_matrix.col(i) = source_vec[i];
                target_matrix.col(i) = target_vec[i];
            }

            //  计算匹配点云数据的均值，数据进行去质心
            Eigen::Vector3d source_mean = source_matrix.rowwise().mean();
            Eigen::Vector3d target_mean = target_matrix.rowwise().mean();
            source_matrix.colwise() -= source_mean;
            target_matrix.colwise() -= target_mean;

            //  计算协方差   W = target * source.transpose() 关键注意顺序
            Eigen::Matrix3d W = target_matrix * source_matrix.transpose();
            Eigen::JacobiSVD<Eigen::Matrix3d> svd(W, Eigen::ComputeFullU | Eigen::ComputeFullV);
            Eigen::Matrix3d U = svd.matrixU();
            Eigen::Matrix3d V = svd.matrixV();

            /// 旋转矩阵R_target_source
            /// 平移矩阵t_target_source
            // 进行旋转矩阵求解，这里考虑到vUT的行列式小于0的问题，
            // https://www.liuxiao.org/2019/08/%e4%bd%bf%e7%94%a8-svd-%e6%96%b9%e6%b3%95%e6%b1%82%e8%a7%a3-icp-%e9%97%ae%e9%a2%98/
            Eigen::Matrix3d weight_matrix = Eigen::Matrix3d::Identity();       // 设置旋转矩阵计算权重，防止出现镜像问题
            weight_matrix(2, 2) = (V * U.transpose()).determinant();           // 调整最后一个元素使行列式为1
            Eigen::Matrix3d R = V * weight_matrix * U.transpose();             // 计算调整后的旋转矩阵R

            // 进行t矩阵计算
            Eigen::Vector3d t = target_mean - R * source_mean;

            // 计算平移量的delta量
            double delta_X_translation = (t - T_target_source_.block<3, 1>(0, 3)).norm();

            // 更新pose
            T_target_source_.block<3, 3>(0, 0) = R;
            T_target_source_.block<3, 1>(0, 3) = t;

            if(log_flag_)
                std::cout << std::fixed << std::setprecision(5) <<
                          "iter: " << iter << " | " <<
                          "delta_X(t): " << delta_X_translation << " | " <<
                          "cost time (ms) : " << solver.toc() << std::endl;

            // 根据位移量前后两次迭代的步长，判断是否可以提前终止迭代
            if (delta_X_translation < translation_epsilon_) {
                break;
            }
        }
    }

    /*  不知道初始外参，如果点云的分布轮廓接近（比如点云的模板匹配），可以先利用质心以及分布计算初始外参
     *  但是常见匹配的场景，可能目标点云与source点云数据分布不接近
     */

    /// 获取配准外参
    Eigen::Matrix4d getAlignedExternal() { return T_target_source_; }

    /// 获取 transformed_source_cloud
    typename pcl::PointCloud<PointT>::Ptr getAlignedCloud(){
        typename pcl::PointCloud<PointT>::Ptr transformed_source(new pcl::PointCloud<PointT>);
        pcl::transformPointCloud(*source_cloud_ptr_, *transformed_source, T_target_source_);
        return transformed_source;
    }

    /// 获取配准得分
    double getAlignedScore() { return score_; }


    /// 设置是否需要log
    void setLogFlag(bool log_flag) { this->log_flag_ = log_flag; }

    /// 设置最大迭代次数
    void setMaxIterations(int max_iterations) { this->max_iterations_ = max_iterations; }

    /// 设置最近邻查询的距离阈值
    void setNearestDist(double nearest_dist) { this->nearest_dist_ = nearest_dist; }

    /// 设置旋转的迭代步长阈值
    void setRotationEpsilon(double epsilon) { this->rotation_epsilon_ = epsilon; }

    /// 设置位移的迭代步长阈值
    void setTranslationEpsilon(double epsilon) { this->translation_epsilon_ = epsilon; }

    /// 设置配准初值
    void setInitPose(Eigen::Matrix4d &init_T_target_source) { T_target_source_ = init_T_target_source; }

private:

    /// SVD求解的参数
    bool log_flag_ = false;
    int max_iterations_ = 10;
    double nearest_dist_ = 10.0;
    double rotation_epsilon_ = 1e-6;
    double translation_epsilon_ = 1e-6;
    Eigen::Matrix4d T_target_source_ = Eigen::Matrix4d::Identity();
    double score_ = 0.0;


    /// Target的点云信息
    typename pcl::PointCloud<PointT>::Ptr target_cloud_ptr_ = boost::make_shared<pcl::PointCloud<PointT>>();
    Eigen::Vector3d target_center_;
    pcl::KdTreeFLANN<PointT> target_kd_tree_;

    /// Source的点云信息
    typename pcl::PointCloud<PointT>::Ptr source_cloud_ptr_ = boost::make_shared<pcl::PointCloud<PointT>>();
    Eigen::Vector3d source_center_;

};

template<typename PointT>
class ICP_P2Point {
public:
    ICP_P2Point() = default;

    ~ICP_P2Point() = default;


    /// 设置Target点云
    void setTargetCloud(const typename pcl::PointCloud<PointT>::Ptr &target_cloud_ptr){

        target_cloud_ptr_ = target_cloud_ptr;

        target_center_ = std::accumulate(target_cloud_ptr_->points.begin(),
                                         target_cloud_ptr_->points.end(),
                                         Eigen::Vector3d::Zero().eval(),
                                         [](const Eigen::Vector3d &c, const PointT &pt) -> Eigen::Vector3d {
                                             return c + Eigen::Vector3d(pt.x, pt.y, pt.z);
                                         });
        target_center_ /= static_cast<double>(target_cloud_ptr_->points.size());

        target_kd_tree_.setInputCloud(target_cloud_ptr_);

        if(log_flag_){
            std::cout << "target_cloud_ptr_ size: " << target_cloud_ptr_->points.size() << std::endl;
            std::cout << "target_center_: " << target_center_.transpose() << std::endl;
        }

    }

    /// 设置Source点云
    void setSourceCloud(const typename pcl::PointCloud<PointT>::Ptr &source_cloud_ptr) {

        source_cloud_ptr_ = source_cloud_ptr;

        source_center_ = std::accumulate(source_cloud_ptr_->points.begin(),
                                         source_cloud_ptr_->points.end(),
                                         Eigen::Vector3d::Zero().eval(),
                                         [](const Eigen::Vector3d &c, const PointT &pt) -> Eigen::Vector3d {
                                             return c + Eigen::Vector3d(pt.x, pt.y, pt.z);
                                         });
        source_center_ /= static_cast<double>(source_cloud_ptr_->points.size());

        if(log_flag_){
            std::cout << "source_cloud_ptr_ size: " << source_cloud_ptr_->points.size() << std::endl;
            std::cout << "source_center_: " << source_center_.transpose() << std::endl;
        }

    }

    /**
     *
     */
    void Aligned(){

        typename pcl::PointCloud<PointT>::Ptr transformed_cloud(new pcl::PointCloud<PointT>);

        for (int iter = 0; iter < max_iterations_; ++iter) {

            // STEP1: 点云变换
            pcl::transformPointCloud(*source_cloud_ptr_, *transformed_cloud, T_target_source_);

            // STEP2: 进行最近邻查询，组织数据
            std::vector<Eigen::Vector3d> source_vec;
            std::vector<Eigen::Vector3d> error_vec;
            source_vec.reserve(source_cloud_ptr_->points.size());  // 预分配内存
            error_vec.reserve(source_cloud_ptr_->points.size());   // 预分配内存

            for (size_t index = 0; index < source_cloud_ptr_->points.size(); ++index) {

                // 计算最近邻
                int candidate = 1;
                std::vector<int> idx;
                std::vector<float> dist;
                target_kd_tree_.nearestKSearch(transformed_cloud->points[index], candidate, idx, dist);

                if (dist.front() > nearest_dist_) {
                    continue;
                }

                source_vec.emplace_back(source_cloud_ptr_->points[index].x,
                                        source_cloud_ptr_->points[index].y,
                                        source_cloud_ptr_->points[index].z);

                // q - ( Rp + t )
                error_vec.emplace_back(target_cloud_ptr_->points[idx[0]].x - transformed_cloud->points[index].x,
                                       target_cloud_ptr_->points[idx[0]].y - transformed_cloud->points[index].y,
                                       target_cloud_ptr_->points[idx[0]].z - transformed_cloud->points[index].z);

            }

            TicToc manager;
            // 制作点点残差
            // 耗时分析
            //  一共120000点， manager1: 0.00327ms manager2: 0.017378ms 每一个点耗时0.02ms 一共120000个点
            //  120000 * 0.02 = 2400000ms = 2.4s

            //  svd方法:
            //      计算前的数据组织 M_target_3xn M_source_3xn，直接做矩阵计算 很快

            // 精度分析：
            //      svd6自由度分开计算，平移量都是直接借助质心相差计算的，精度不及整体梯度下降好
            //      点点迭代的方式，每一次迭代都要计算雅克比矩阵，一起优化

            Eigen::MatrixXd H = Eigen::Matrix<double, 6, 6>::Zero();
            Eigen::Matrix<double, 6, 1> b = Eigen::Matrix<double, 6, 1>::Zero();
            for (int index = 0; index < source_vec.size(); index++) {

                TicToc manager1;
                // 先计算雅阁比矩阵
                //  fx = q - ( Rp + t )
                // [ d(fx) / d(R) d(fx) / d(t)]
                Eigen::Matrix<double, 3, 6> J;
                J.block<3, 3>(0, 0) = T_target_source_.block<3, 3>(0, 0) * Skew(source_vec[index]);
                J.block<3, 3>(0, 3) = -1.0 * Eigen::Matrix3d::Identity();
//                std::cout << "manager1 cost: " << manager1.toc() << std::endl;

                TicToc manager2;
                // 计算H矩阵
                H += J.transpose() * J;

                // 计算b矩阵
                b += -1.0 * J.transpose() * error_vec[index];
//                std::cout << "manager2 cost: " << manager2.toc() << std::endl;
            }
            std::cout << "manager cost: " << manager.toc() << std::endl;


            // 求解 H x = b
            TicToc solver;
            Eigen::Matrix<double, 6, 1> x = H.ldlt().solve(b);
            std::cout << "solver cost: " << solver.toc() << std::endl;

            if (x.block<3, 1>(3, 0).norm() > last_epsilon_) {
                break;
            }

            // 计算平移量的delta量
            double cur_epsilon = x.block<3, 1>(3, 0).norm();

            // 更新pose
            T_target_source_.block<3, 3>(0, 0) = T_target_source_.block<3, 3>(0, 0) * ManifoldMath::Exp<double>(x.block<3, 1>(0, 0));
            T_target_source_.block<3, 1>(0, 3) = T_target_source_.block<3, 1>(0, 3) + x.block<3, 1>(3, 0);

            if(log_flag_){
                std::cout << "iter: " << iter << std::endl;
                std::cout << "cur_epsilon: " << cur_epsilon << std::endl;
            }

            // 计算位移量迭代的epsilon
            if (cur_epsilon < epsilon_) {
                break;
            }

            last_epsilon_ = cur_epsilon;
        }
    }

    /// 设置是否需要log
    void setLogFlag(bool log_flag) { this->log_flag_ = log_flag; }

    /// 设置最大迭代次数
    void setMaxIterations(int max_iterations) { this->max_iterations_ = max_iterations; }

    /// 设置迭代精度阈值
    void setEpsilon(double epsilon) { this->epsilon_ = epsilon; }

    /// 设置最近邻查询的距离阈值
    void setNearestDist(double nearest_dist) { this->nearest_dist_ = nearest_dist; }

    /// 设置配准初值
    void setInitPose(Eigen::Matrix4d &init_T_target_source) { T_target_source_ = init_T_target_source; }

    /// 获取配准得分
    double getAlignedScore() { return score_; }

    /// 获取配准外参
    Eigen::Matrix4d getAlignedExternal() { return T_target_source_; }

    /// 获取 transformed_source_cloud
    typename pcl::PointCloud<PointT>::Ptr getAlignedCloud(){
        typename pcl::PointCloud<PointT>::Ptr transformed_source(new pcl::PointCloud<PointT>);
        pcl::transformPointCloud(*source_cloud_ptr_, *transformed_source, T_target_source_);
        return transformed_source;
    }



private:
    /// P2Point求解的参数
    bool log_flag_ = false;
    int max_iterations_ = 10;
    double epsilon_ = 1e-3;
    double nearest_dist_ = 10.0;

    double last_epsilon_ = std::numeric_limits<double>::max();

    /// 基础变换矩阵
    Eigen::Matrix4d T_target_source_ = Eigen::Matrix4d::Identity();

    /// Target的点云信息
    typename pcl::PointCloud<PointT>::Ptr target_cloud_ptr_ = boost::make_shared<pcl::PointCloud<PointT>>();
    Eigen::Vector3d target_center_;
    pcl::KdTreeFLANN<PointT> target_kd_tree_;

    /// Source的点云信息
    typename pcl::PointCloud<PointT>::Ptr source_cloud_ptr_ = boost::make_shared<pcl::PointCloud<PointT>>();
    Eigen::Vector3d source_center_;

    /// Aligned的Score以及结果
    double score_;


};


template<typename PointT>
class ICP_P2Line {
public:
    ICP_P2Line()=default;

    ~ICP_P2Line()=default;

    /// 设置Target点云
    void setTargetCloud(const typename pcl::PointCloud<PointT>::Ptr &target_cloud_ptr){

        target_cloud_ptr_ = target_cloud_ptr;

        target_center_ = std::accumulate(target_cloud_ptr_->points.begin(),
                                         target_cloud_ptr_->points.end(),
                                         Eigen::Vector3d::Zero().eval(),
                                         [](const Eigen::Vector3d &c, const PointT &pt) -> Eigen::Vector3d {
                                             return c + Eigen::Vector3d(pt.x, pt.y, pt.z);
                                         });
        target_center_ /= static_cast<double>(target_cloud_ptr_->points.size());

        target_kd_tree_.setInputCloud(target_cloud_ptr_);

        if(log_flag_){
            std::cout << "target_cloud_ptr_ size: " << target_cloud_ptr_->points.size() << std::endl;
            std::cout << "target_center_: " << target_center_.transpose() << std::endl;
        }

    }

    /// 设置Source点云
    void setSourceCloud(const typename pcl::PointCloud<PointT>::Ptr &source_cloud_ptr) {

        source_cloud_ptr_ = source_cloud_ptr;

        source_center_ = std::accumulate(source_cloud_ptr_->points.begin(),
                                         source_cloud_ptr_->points.end(),
                                         Eigen::Vector3d::Zero().eval(),
                                         [](const Eigen::Vector3d &c, const PointT &pt) -> Eigen::Vector3d {
                                             return c + Eigen::Vector3d(pt.x, pt.y, pt.z);
                                         });
        source_center_ /= static_cast<double>(source_cloud_ptr_->points.size());

        if(log_flag_){
            std::cout << "source_cloud_ptr_ size: " << source_cloud_ptr_->points.size() << std::endl;
            std::cout << "source_center_: " << source_center_.transpose() << std::endl;
        }

    }

    /// 进行配准
    void Aligned(){

        typename pcl::PointCloud<PointT>::Ptr transformed_cloud(new pcl::PointCloud<PointT>);

        for (int iter = 0; iter < max_iterations_; ++iter) {

            // 迭代耗时统计
            TicToc solver;

            // STEP1: 点云变换
            pcl::transformPointCloud(*source_cloud_ptr_, *transformed_cloud, T_target_source_);

            // STEP2: 进行最近邻查询，组织数据
            Eigen::MatrixXd H = Eigen::Matrix<double, 6, 6>::Zero();
            Eigen::Matrix<double, 6, 1> b = Eigen::Matrix<double, 6, 1>::Zero();
            double loss_mean = 0.0;
            int effective_cnts = 0;
            for (size_t index = 0; index < source_cloud_ptr_->points.size(); ++index) {

                //  计算最近邻
                int candidate = 5;
                std::vector<int> idx;
                std::vector<float> dist;
                target_kd_tree_.nearestKSearch(transformed_cloud->points[index], candidate, idx, dist);

                //  最近邻距离check
                if (dist.size() < 5 || dist.back() > nearest_dist_) {
                    continue;
                }

                //  方向向量计算估计
                std::vector<Eigen::Vector3d> kdtree_points;
                for (const auto &tmp: idx) {
                    kdtree_points.emplace_back(target_cloud_ptr_->points[tmp].x, target_cloud_ptr_->points[tmp].y, target_cloud_ptr_->points[tmp].z);
                }

                Eigen::Vector3d direction_tmp;
                if(!GeometryMath::estimate_line(kdtree_points, p2line_dist_, direction_tmp)){
                    continue;
                }

                //  垂直投影矩阵估计
                Eigen::Matrix3d vertical_projection_matrix = Eigen::Matrix3d::Identity() - (direction_tmp * direction_tmp.transpose()) / (direction_tmp.transpose() * direction_tmp);

                //  存储source数据
                Eigen::Vector3d source_point(source_cloud_ptr_->points[index].x, source_cloud_ptr_->points[index].y, source_cloud_ptr_->points[index].z);

                //  计算误差
                Eigen::Vector3d error = vertical_projection_matrix * Eigen::Vector3d(target_cloud_ptr_->points[idx[0]].x - transformed_cloud->points[index].x,
                                                                                     target_cloud_ptr_->points[idx[0]].y - transformed_cloud->points[index].y,
                                                                                     target_cloud_ptr_->points[idx[0]].z - transformed_cloud->points[index].z);
                //
                Eigen::Matrix<double, 3, 6> J;
                J.block<3, 3>(0, 0) = vertical_projection_matrix *  T_target_source_.block<3, 3>(0, 0) * Skew(source_point);
                J.block<3, 3>(0, 3) = vertical_projection_matrix * (-1.0) * Eigen::Matrix3d::Identity();

                // 计算H矩阵
                H += J.transpose() * J;

                // 计算b矩阵
                b += -1.0 * J.transpose() * error;

                // 计算总loss
                effective_cnts++;
                loss_mean += error.norm();
            }
            loss_mean /= effective_cnts;

            // 求解 H x = b
            Eigen::Matrix<double, 6, 1> delta_X = H.ldlt().solve(b);

            //  梯度下降异常检测
            if (loss_mean > last_loss_mean_)
                break;
            last_loss_mean_ = loss_mean;    //  更新总loss的均值

            // 更新pose
            T_target_source_.block<3, 3>(0, 0) = T_target_source_.block<3, 3>(0, 0) * ManifoldMath::Exp<double>(delta_X.block<3, 1>(0, 0));
            T_target_source_.block<3, 1>(0, 3) = T_target_source_.block<3, 1>(0, 3) + delta_X.block<3, 1>(3, 0);

            if(log_flag_)
                std::cout << std::fixed << std::setprecision(5) <<
                          "iter: " << iter << " | " <<
                          "loss_mean: " << loss_mean << " | " <<
                          "delta_X(r t): " << delta_X.block<3, 1>(0, 0).norm() << " " << delta_X.block<3, 1>(3, 0).norm() << " | " <<
                          "cost time (ms) : " << solver.toc() << std::endl;

            // 判断是否可以提前结束迭代
            if (delta_X.block<3, 1>(0, 0).norm() < rotation_epsilon_ && delta_X.block<3, 1>(3, 0).norm() < translation_epsilon_) {
                break;
            }
        }
    }

    ///***************************************** 参数配置 *****************************************///
    ///     设置是否需要log
    void setLogFlag(bool log_flag) { this->log_flag_ = log_flag; }

    ///     设置最大迭代次数
    void setMaxIterations(int max_iterations) { this->max_iterations_ = max_iterations; }

    ///     设置终止迭代步长的阈值
    void setRotationEpsilon(double epsilon) { this->rotation_epsilon_ = epsilon; }

    void setTranslationEpsilon(double epsilon) { this->translation_epsilon_ = epsilon; }

    ///     设置最近邻查询的距离阈值
    void setNearestDist(double nearest_dist) { this->nearest_dist_ = nearest_dist; }

    ///     设置配准初值
    void setInitPose(Eigen::Matrix4d &init_T_target_source) { T_target_source_ = init_T_target_source; }

    /// 获取结果
    ///     获取配准得分
    double getAlignedScore() { return score_; }

    ///     获取配准外参
    Eigen::Matrix4d getAlignedExternal() { return T_target_source_; }

    ///     获取 transformed_source_cloud
    typename pcl::PointCloud<PointT>::Ptr getAlignedCloud(){
        typename pcl::PointCloud<PointT>::Ptr transformed_source(new pcl::PointCloud<PointT>);
        pcl::transformPointCloud(*source_cloud_ptr_, *transformed_source, T_target_source_);
        return transformed_source;
    }

private:

    /// P2LINE求解的参数
    bool log_flag_ = false;
    int max_iterations_ = 10;
    double rotation_epsilon_ = 1e-3, translation_epsilon_ = 1e-3;
    double last_loss_mean_ = std::numeric_limits<double>::max();
    double nearest_dist_ = 10.0;
    double p2line_dist_ = 1.0;

    /// 基础变换矩阵
    Eigen::Matrix4d T_target_source_ = Eigen::Matrix4d::Identity();

    /// Target的点云信息
    typename pcl::PointCloud<PointT>::Ptr target_cloud_ptr_ = boost::make_shared<pcl::PointCloud<PointT>>();
    Eigen::Vector3d target_center_;
    pcl::KdTreeFLANN<PointT> target_kd_tree_;

    /// Source的点云信息
    typename pcl::PointCloud<PointT>::Ptr source_cloud_ptr_ = boost::make_shared<pcl::PointCloud<PointT>>();
    Eigen::Vector3d source_center_;

    /// Aligned的Score以及结果
    double score_;

};

template<typename PointT>
class ICP_P2Plane {
public:
    ICP_P2Plane() = default;

    ~ICP_P2Plane() = default;

    /// 设置Target点云
    void setTargetCloud(const typename pcl::PointCloud<PointT>::Ptr &target_cloud_ptr){

        target_cloud_ptr_ = target_cloud_ptr;

        target_center_ = std::accumulate(target_cloud_ptr_->points.begin(),
                                         target_cloud_ptr_->points.end(),
                                         Eigen::Vector3d::Zero().eval(),
                                         [](const Eigen::Vector3d &c, const PointT &pt) -> Eigen::Vector3d {
                                             return c + Eigen::Vector3d(pt.x, pt.y, pt.z);
                                         });
        target_center_ /= static_cast<double>(target_cloud_ptr_->points.size());

        target_kd_tree_.setInputCloud(target_cloud_ptr_);

        if(log_flag_){
            std::cout << "target_cloud_ptr_ size: " << target_cloud_ptr_->points.size() << std::endl;
            std::cout << "target_center_: " << target_center_.transpose() << std::endl;
        }

    }

    /// 设置Source点云
    void setSourceCloud(const typename pcl::PointCloud<PointT>::Ptr &source_cloud_ptr) {

        source_cloud_ptr_ = source_cloud_ptr;

        source_center_ = std::accumulate(source_cloud_ptr_->points.begin(),
                                         source_cloud_ptr_->points.end(),
                                         Eigen::Vector3d::Zero().eval(),
                                         [](const Eigen::Vector3d &c, const PointT &pt) -> Eigen::Vector3d {
                                             return c + Eigen::Vector3d(pt.x, pt.y, pt.z);
                                         });
        source_center_ /= static_cast<double>(source_cloud_ptr_->points.size());

        if(log_flag_){
            std::cout << "source_cloud_ptr_ size: " << source_cloud_ptr_->points.size() << std::endl;
            std::cout << "source_center_: " << source_center_.transpose() << std::endl;
        }

    }

    /// 进行配准
    void Aligned(){

        typename pcl::PointCloud<PointT>::Ptr transformed_cloud(new pcl::PointCloud<PointT>);

        for (int iter = 0; iter < max_iterations_; ++iter) {

            TicToc solver;

            // STEP1: 点云变换
            pcl::transformPointCloud(*source_cloud_ptr_, *transformed_cloud, T_target_source_);

            // STEP2: 进行最近邻查询，组织数据
            std::vector<Eigen::Vector3d> source_vec;            //  原始点云数据
            std::vector<Eigen::Vector4d> plane_vec;             //  单位平面方程的参数
            std::vector<Eigen::Matrix3d> normal_project_vec;    //  法向量的投影矩阵
            std::vector<Eigen::Vector3d> error_vec;             //  投影的误差向量

            source_vec.reserve(source_cloud_ptr_->points.size());           // 原始点云
            normal_project_vec.reserve(source_cloud_ptr_->points.size());   // 方向向量的垂直投影矩阵
            error_vec.reserve(source_cloud_ptr_->points.size());            // 误差向量

            // 对每个点计算最近邻关系，估计多维度信息（线、面），组织高斯牛顿数据
            Eigen::MatrixXd H = Eigen::Matrix<double, 6, 6>::Zero();
            Eigen::Matrix<double, 6, 1> b = Eigen::Matrix<double, 6, 1>::Zero();
            double loss_mean = 0.0;
            int effective_cnts = 0;
            if (tbb_param_flag_) {
                tbb::parallel_for(
                        tbb::blocked_range<size_t>(0, source_cloud_ptr_->points.size()),
                        [&](const tbb::blocked_range<size_t> &r) {
                            for (size_t index = r.begin(); index < r.end(); ++index) {
                                //  计算最近邻
                                int candidate = 5;
                                std::vector<int> idx;
                                std::vector<float> dist;
                                target_kd_tree_.nearestKSearch(transformed_cloud->points[index], candidate, idx, dist);

                                //  最近邻距离check
                                if (dist.size() < 5 || dist.back() > nearest_dist_) {
                                    continue;
                                }

                                //  方向向量计算估计
                                std::vector<Eigen::Vector3d> target_kdtree_points;
                                for (const auto &tmp: idx) {
                                    target_kdtree_points.emplace_back(target_cloud_ptr_->points[tmp].x,
                                                                      target_cloud_ptr_->points[tmp].y,
                                                                      target_cloud_ptr_->points[tmp].z);
                                }

                                Eigen::Vector4d plane_param;
                                if (!GeometryMath::estimate_plane(target_kdtree_points, p2plane_dist_, plane_param)) {
                                    continue;
                                }

                                //  平面法向量的投影矩阵
                                Eigen::Vector3d normal_direction = plane_param.block<3, 1>(0, 0);
                                Eigen::Matrix3d normal_projection = normal_direction * normal_direction.transpose() /
                                                                    (normal_direction.transpose() * normal_direction);

                                //  source的point
                                Eigen::Vector3d source_point(source_cloud_ptr_->points[index].x,
                                                             source_cloud_ptr_->points[index].y,
                                                             source_cloud_ptr_->points[index].z);

                                //  存储误差err
                                Eigen::Vector3d error = normal_projection * Eigen::Vector3d(
                                        target_cloud_ptr_->points[idx[0]].x - transformed_cloud->points[index].x,
                                        target_cloud_ptr_->points[idx[0]].y - transformed_cloud->points[index].y,
                                        target_cloud_ptr_->points[idx[0]].z - transformed_cloud->points[index].z);

                                //  计算雅阁比矩阵
                                Eigen::Matrix<double, 3, 6> J;
                                J.block<3, 3>(0, 0) =
                                        normal_projection * T_target_source_.block<3, 3>(0, 0) * Skew(source_point);
                                J.block<3, 3>(0, 3) = normal_projection * (-1.0) * Eigen::Matrix3d::Identity();

                                // 计算H矩阵
                                H += J.transpose() * J;

                                // 计算b矩阵
                                b += -1.0 * J.transpose() * error;

                                loss_mean += error.norm();
                                effective_cnts++;

                            }
                        }
                );
            }
            else {
                for (size_t index = 0; index < source_cloud_ptr_->points.size(); ++index) {

                    //  计算最近邻
                    int candidate = 5;
                    std::vector<int> idx;
                    std::vector<float> dist;
                    target_kd_tree_.nearestKSearch(transformed_cloud->points[index], candidate, idx, dist);

                    //  最近邻距离check
                    if (dist.size() < 5 || dist.back() > nearest_dist_) {
                        continue;
                    }

                    //  方向向量计算估计
                    std::vector<Eigen::Vector3d> target_kdtree_points;
                    for (const auto &tmp: idx) {
                        target_kdtree_points.emplace_back(target_cloud_ptr_->points[tmp].x,
                                                          target_cloud_ptr_->points[tmp].y,
                                                          target_cloud_ptr_->points[tmp].z);
                    }

                    Eigen::Vector4d plane_param;
                    if (!GeometryMath::estimate_plane(target_kdtree_points, p2plane_dist_, plane_param)) {
                        continue;
                    }

                    //  平面法向量的投影矩阵
                    Eigen::Vector3d normal_direction = plane_param.block<3, 1>(0, 0);
                    Eigen::Matrix3d normal_projection = normal_direction * normal_direction.transpose() /
                                                        (normal_direction.transpose() * normal_direction);

                    //  source的point
                    Eigen::Vector3d source_point(source_cloud_ptr_->points[index].x, source_cloud_ptr_->points[index].y,
                                                 source_cloud_ptr_->points[index].z);

                    //  存储误差err
                    Eigen::Vector3d error = normal_projection * Eigen::Vector3d(
                            target_cloud_ptr_->points[idx[0]].x - transformed_cloud->points[index].x,
                            target_cloud_ptr_->points[idx[0]].y - transformed_cloud->points[index].y,
                            target_cloud_ptr_->points[idx[0]].z - transformed_cloud->points[index].z);

                    //  计算雅阁比矩阵
                    Eigen::Matrix<double, 3, 6> J;
                    J.block<3, 3>(0, 0) = normal_projection * T_target_source_.block<3, 3>(0, 0) * Skew(source_point);
                    J.block<3, 3>(0, 3) = normal_projection * (-1.0) * Eigen::Matrix3d::Identity();

                    // 计算H矩阵
                    H += J.transpose() * J;

                    // 计算b矩阵
                    b += -1.0 * J.transpose() * error;

                    loss_mean += error.norm();
                    effective_cnts++;
                }
            }
            loss_mean /= effective_cnts;


            // 求解 H x = b
            Eigen::Matrix<double, 6, 1> delta_X = H.ldlt().solve(b);

            // 梯度下降方向错误检查
            if (loss_mean > last_loss_mean_) {
                break;
            }
            last_loss_mean_ = loss_mean;

            // 更新pose
            T_target_source_.block<3, 3>(0, 0) = T_target_source_.block<3, 3>(0, 0) * ManifoldMath::Exp<double>(delta_X.block<3, 1>(0, 0));
            T_target_source_.block<3, 1>(0, 3) = T_target_source_.block<3, 1>(0, 3) + delta_X.block<3, 1>(3, 0);

            if(log_flag_)
                std::cout << std::fixed << std::setprecision(5) <<
                          "iter: " << iter << " | " <<
                          "loss_mean: " << loss_mean << " | " <<
                          "delta_X(r t): " << delta_X.block<3, 1>(0, 0).norm() << " " << delta_X.block<3, 1>(3, 0).norm() << " | " <<
                          "cost time (ms) : " << solver.toc() << std::endl;

            // 提前结束迭代（delta步长小于给定阈值）
            if (delta_X.block<3, 1>(0, 0).norm() < rotation_epsilon_ && delta_X.block<3, 1>(3, 0).norm() < translation_epsilon_)
                break;

        }
    }

    /// 结果获取
    ///     获取配准得分
    double getAlignedScore() { return score_; }

    ///     获取配准外参
    Eigen::Matrix4d getAlignedExternal() { return T_target_source_; }

    ///     获取 transformed_source_cloud
    typename pcl::PointCloud<PointT>::Ptr getAlignedCloud(){
        typename pcl::PointCloud<PointT>::Ptr transformed_source(new pcl::PointCloud<PointT>);
        pcl::transformPointCloud(*source_cloud_ptr_, *transformed_source, T_target_source_);
        return transformed_source;
    }

    /// ***************************************** 参数管理 ***************************************** ///
    ///     设置是否需要log
    void setLogFlag(bool log_flag) { this->log_flag_ = log_flag; }

    ///     设置最大迭代次数
    void setMaxIterations(int max_iterations) { this->max_iterations_ = max_iterations; }

    ///     设置迭代精度阈值
    void setTranslationEpsilon(double epsilon) { this->translation_epsilon_ = epsilon; }

    void setRotationEpsilon(double epsilon) { this->rotation_epsilon_ = epsilon; }

    ///     设置最近邻查询的距离阈值
    void setNearestDist(double nearest_dist) { this->nearest_dist_ = nearest_dist; }

    ///     设置P2Plane距离阈值
    void setP2PlaneDist(double p2plane_dist) { this->p2plane_dist_ = p2plane_dist; }

    ///     设置配准初值
    void setInitPose(Eigen::Matrix4d &init_T_target_source) { T_target_source_ = init_T_target_source; }

    ///     是否开启并行计算
    void setTbbParam(bool tbb_param) { tbb_param_flag_ = tbb_param; }

private:

    /// P2Plane求解的参数
    bool log_flag_ = false;
    int max_iterations_ = 10;
    double rotation_epsilon_ = 1e-3;
    double translation_epsilon_ = 1e-3;
    double nearest_dist_ = 10.0;
    double p2plane_dist_ = 1.0;
    Eigen::Matrix4d T_target_source_ = Eigen::Matrix4d::Identity();    /// 基础变换矩阵

    double last_loss_mean_ = std::numeric_limits<double>::max();

    /// Target的点云信息
    typename pcl::PointCloud<PointT>::Ptr target_cloud_ptr_ = boost::make_shared<pcl::PointCloud<PointT>>();
    Eigen::Vector3d target_center_;
    pcl::KdTreeFLANN<PointT> target_kd_tree_;

    /// Source的点云信息
    typename pcl::PointCloud<PointT>::Ptr source_cloud_ptr_ = boost::make_shared<pcl::PointCloud<PointT>>();
    Eigen::Vector3d source_center_;

    /// Aligned的Score以及结果
    double score_;

    /// 是否进行并行计算
    bool tbb_param_flag_ = false;

};


template<typename PointT>
class NDT{
public:

private:

};

#endif //EIGEN_ICP_ICP_MOTHOD_H
