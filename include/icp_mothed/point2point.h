//
// Created by westwell on 25-1-15.
//

#ifndef EIGEN_ICP_POINT2POINT_H
#define EIGEN_ICP_POINT2POINT_H

#include "registration_base.h"

namespace Registration {
class Point2PointRegistration : public RegistrationBase {
 public:
  Point2PointRegistration() {
    registration_mode_ = RegistrationMode::Point2Point;

    iterations_ = 10;
    epsilon_ = 1e-6;
    nearest_dist_ = 5.0;
    use_tbb_flag_ = false;

    init_T_ = Eigen::Matrix4d::Identity();
    source_cloud_ptr_ = boost::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
    target_cloud_ptr_ = boost::make_shared<pcl::PointCloud<pcl::PointXYZI>>();

    convergence_flag_ = true;
    final_T_ = Eigen::Matrix4d::Identity();
  }

  ~Point2PointRegistration() override = default;

  // 参数配置
  void setIterations(int iterations) override { iterations_ = iterations; }
  void setEpsilon(double epsilon) override { epsilon_ = epsilon; }
  void setNearestDist(double nearest_dist) override { nearest_dist_ = nearest_dist; }
  void setTBBFlag(bool use_tbb_flag) override { use_tbb_flag_ = use_tbb_flag; }
  void logParameter() override {
    std::cout << "  Registration Mode: " << getRegistrationMode(registration_mode_) << std::endl;
    std::cout << "    iterations: " << iterations_ << std::endl;
    std::cout << "    epsilon: " << epsilon_ << std::endl;
    std::cout << "    nearest_dist: " << nearest_dist_ << std::endl;
    std::cout << "    use_tbb_flag: " << (use_tbb_flag_ ? "true" : "false") << std::endl;
    std::cout << "    init_T_target_source: " << std::endl << init_T_.matrix() << std::endl;
  }

  // 配置输入参数
  void setSourceCloud(const pcl::PointCloud<pcl::PointXYZI>::Ptr &source_cloud_ptr) override {
    source_cloud_ptr_ = source_cloud_ptr;
  }
  void setTargetCloud(const pcl::PointCloud<pcl::PointXYZI>::Ptr &target_cloud_ptr) override {
    target_cloud_ptr_ = target_cloud_ptr;
  }
  void setInitT(const Eigen::Matrix4d &init_T) override { init_T_ = init_T; }

  // 优化求解
  bool Handle() override {

    // 参数打印
    if (use_log_flag_)
      logParameter();

    // 断言检测
    assert(!target_cloud_ptr_->points.empty());
    assert(!source_cloud_ptr_->points.empty());

    // 构建目标点云的Kdtree
    pcl::KdTreeFLANN<pcl::PointXYZI> target_KDtree;
    target_KDtree.setInputCloud(target_cloud_ptr_);

    // 初始化每一轮外参变换后的transformed_cloud
    pcl::PointCloud<pcl::PointXYZI>::Ptr transformed_cloud_ptr(new pcl::PointCloud<pcl::PointXYZI>());

    // 记录残差更新
    double last_mean_residual = 1e10;

    for (int iter = 0; iter < iterations_; ++iter) {
      double res_mean = 0;
      int effect_cnt = 0;

      // 存储  JT * J  以及 -1 * J * err
      std::pair<Eigen::Matrix<double, 6, 6>, Eigen::Matrix<double, 6, 1>> H_and_err =
          std::make_pair(Eigen::Matrix<double, 6, 6>::Zero(), Eigen::Matrix<double, 6, 1>::Zero());

      // 对原始点云进行变换
      pcl::transformPointCloud(*source_cloud_ptr_, *transformed_cloud_ptr, final_T_);

      // 计算最近邻，计算雅阁比
      for (int index = 0; index < source_cloud_ptr_->points.size(); ++index) {

        // 最近邻查询
        pcl::PointXYZI &transformed_point = transformed_cloud_ptr->points[index];
        std::vector<int> index_vec;
        std::vector<float> dist_vec;
        int num_found = target_KDtree.nearestKSearch(transformed_point, 1, index_vec, dist_vec);

        // 判断点点关联是否可以有效
        auto checkEffective = [&index_vec, &dist_vec, this]() -> bool {
          return (!index_vec.empty() && dist_vec.front() < nearest_dist_);
        };

        if (checkEffective()) {

          // 记录最近邻的target点 p
          pcl::PointXYZI &knn_target_point = target_cloud_ptr_->points[index_vec[0]];
          Eigen::Vector3d target_point_eigen(knn_target_point.x, knn_target_point.y, knn_target_point.z);
          // 记录变换后的激光点位置
          Eigen::Vector3d transformed_point_eigen(transformed_point.x, transformed_point.y, transformed_point.z);
          // 构建误差 err
          Eigen::Vector3d err = target_point_eigen - transformed_point_eigen;

          // 原始激光点位置
          pcl::PointXYZI &origin_point = source_cloud_ptr_->points[index];
          Eigen::Vector3d origin_point_eigen(origin_point.x, origin_point.y, origin_point.z);

          // 构建雅阁比 J
          Eigen::Matrix<double, 3, 6> J;
          J.block<3, 3>(0, 0) = final_T_.block<3, 3>(0, 0) * manifold_math::skew_sym_mat(origin_point_eigen);
          J.block<3, 3>(0, 3) = -Eigen::Matrix3d::Identity();

          // 更新海森矩阵
          H_and_err.first += J.transpose() * J;
          H_and_err.second += -1.0 * J.transpose() * err;

          // 残差的均值递归求解
          effect_cnt++;
          res_mean = res_mean + (err.norm() - res_mean) / effect_cnt;

        }

      }

      // 统计本次的总体点到点之间的平均残差
      if (res_mean < last_mean_residual) {
        std::cout << "[point2point] iter: " << iter << ", res: " << res_mean << std::endl;

        // 计算delta_x
        Eigen::Matrix<double, 6, 6> H = H_and_err.first;
        Eigen::Matrix<double, 6, 1> err = H_and_err.second;
        Eigen::Matrix<double, 6, 1> dx = H.inverse() * err;

        // 更新 final_T
        final_T_.block<3, 3>(0, 0) = final_T_.block<3, 3>(0, 0) * manifold_math::Exp<double>(dx.head<3>());
        final_T_.block<3, 1>(0, 3) += dx.tail<3>();

        last_mean_residual = res_mean;

      } else {
        std::cout << "[point2point] iter: " << iter << ", res: " << res_mean << std::endl;
        std::cout << "  [Error] registration gradient descent anomaly, ready to exit..." << std::endl;
        break;
      }
    }

    return convergence_flag_;
  }

  // 获取结果
  void getInitTransform(Eigen::Matrix4d &init_T) override { init_T = init_T_; }
  void getRegistrationTransform(Eigen::Matrix4d &option_transform) override { option_transform = final_T_; }
  void getTransformedOriginCloud(const pcl::PointCloud<pcl::PointXYZI>::Ptr &transformed_cloud) override {
    pcl::transformPointCloud(*source_cloud_ptr_, *transformed_cloud, final_T_);
  };
};
}

#endif //EIGEN_ICP_POINT2POINT_H
