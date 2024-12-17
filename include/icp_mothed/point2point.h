//
// Created by westwell on 25-1-15.
//

#ifndef EIGEN_ICP_POINT2POINT_H
#define EIGEN_ICP_POINT2POINT_H

#include "registration_base.h"

/**
 * @brief 点到点的icp
 */

namespace Registration {
class Point2PointRegistration : public RegistrationBase {
 public:
  Point2PointRegistration() {
    registration_mode_ = RegistrationMode::Point2Point;

    iterations_ = 10;
    epsilon_ = 1e-6;
    init_T_ = Eigen::Matrix4d::Identity();
    nearest_dist_ = 5.0;
    use_tbb_flag_ = false;

    convergence_flag_ = true;
    final_T_ = Eigen::Matrix4d::Identity();

    source_cloud_ptr_ = boost::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
    target_cloud_ptr_ = boost::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
  }

  ~Point2PointRegistration() override = default;

  // 配置参数
  void setIterations(int iterations) override { iterations_ = iterations; }
  void setEpsilon(int epsilon) override { epsilon_ = epsilon; }
  void setNearestDist(double nearest_dist) override { nearest_dist_ = nearest_dist; }
  void setTBBFlag(bool use_tbb_flag) override { use_tbb_flag_ = use_tbb_flag; }

  // 配置输入参数
  void setSourceCloud(
      const pcl::PointCloud<pcl::PointXYZI>::Ptr &source_cloud_ptr) override { source_cloud_ptr_ = source_cloud_ptr; }

  void setTargetCloud(
      const pcl::PointCloud<pcl::PointXYZI>::Ptr &target_cloud_ptr) override { target_cloud_ptr_ = target_cloud_ptr; }

  void setInitT(const Eigen::Matrix4d &init_T) override { init_T_ = init_T; }

  // 打印参数
  void logParameter() override {
    std::cout << " SVD Aligned Parameters: " << std::endl;
    std::cout << "  iterations: " << iterations_ << std::endl;
    std::cout << std::fixed << std::setprecision(9) << "  epsilon: " << epsilon_ << std::endl;
    std::cout << "  nearest_dist: " << nearest_dist_ << std::endl;
    std::cout << "  use_tbb_flag: " << (use_tbb_flag_ ? "true" : "false") << std::endl;
  }

  // 点到点的icp
  bool Handle() override {
    // 断言检测
    assert(!target_cloud_ptr_->points.empty() && !source_cloud_ptr_->points.empty());

    // 构建目标点云的Kdtree
    pcl::KdTreeFLANN<pcl::PointXYZI> target_KDtree;
    target_KDtree.setInputCloud(target_cloud_ptr_);

    // 构建每一次迭代origin点云变换后的点云数据
    pcl::PointCloud<pcl::PointXYZI>::Ptr transformed_cloud_ptr(new pcl::PointCloud<pcl::PointXYZI>());

    // 构建雅阁比矩阵
    size_t point_size = source_cloud_ptr_->points.size();
    std::vector<bool> effect_pts(point_size, false);
    std::vector<Eigen::Matrix<double, 3, 6>> jacobians(point_size);
    std::vector<Eigen::Vector3d> errors(point_size);

    // 记录残差更新
    double last_mean_residual = 1e10;

    for (int iter = 0; iter < iterations_; ++iter) {

      // 对原始点云进行变换
      pcl::transformPointCloud(*source_cloud_ptr_, *transformed_cloud_ptr, final_T_);

      // 计算最近邻，并计算雅阁比
      for (int index = 0; index < source_cloud_ptr_->points.size(); ++index) {

        pcl::PointXYZI &origin_point = source_cloud_ptr_->points[index];
        pcl::PointXYZI &transformed_point = transformed_cloud_ptr->points[index];

        std::vector<int> index_vec;
        std::vector<float> dist_vec;
        int num_found = target_KDtree.nearestKSearch(transformed_point, 1, index_vec, dist_vec);

        // 有效判断 并记录有效点信息计算单个雅阁比
        if (!index_vec.empty() && dist_vec.front() < nearest_dist_) {

          effect_pts[index] = true;

          // 记录最近邻的target点
          pcl::PointXYZI &knn_target_point = target_cloud_ptr_->points[index_vec[0]];
          Eigen::Vector3d target_point_eigen(knn_target_point.x,
                                             knn_target_point.y,
                                             knn_target_point.z);

          // 记录变换后的激光点位置
          Eigen::Vector3d transformed_point_eigen(transformed_point.x,
                                                  transformed_point.y,
                                                  transformed_point.z);

          // 原始激光点位置
          Eigen::Vector3d origin_point_eigen(origin_point.x,
                                             origin_point.y,
                                             origin_point.z);

          // 记录雅阁比信息
          Eigen::Vector3d err = target_point_eigen - transformed_point_eigen;
          Eigen::Matrix<double, 3, 6> J;
          J.block<3, 3>(0, 0) = final_T_.block<3, 3>(0, 0) * manifold_math::skew_sym_mat(origin_point_eigen);
          J.block<3, 3>(0, 3) = -Eigen::Matrix3d::Identity();

          jacobians[index] = J;
          errors[index] = err;
        }

      }

      // 雷达Hessian和error 计算dx
      double total_res = 0;
      int effective_num = 0;
      std::pair<Eigen::Matrix<double, 6, 6>, Eigen::Matrix<double, 6, 1>> H_and_err =
          std::make_pair(Eigen::Matrix<double, 6, 6>::Zero(), Eigen::Matrix<double, 6, 1>::Zero());
      for (size_t idx = 0; idx < source_cloud_ptr_->points.size(); idx++) {
        if (effect_pts[idx]) {
          total_res += errors[idx].dot(errors[idx]);
          effective_num++;

          H_and_err.first += jacobians[idx].transpose() * jacobians[idx];
          H_and_err.second += -1.0 * jacobians[idx].transpose() * errors[idx];
        }
      }

      // 统计本次的总体点到点之间的平均残差
      double mean_res = total_res / effective_num;
      if (mean_res < last_mean_residual) {
        std::cout << "[point2point] iter: " << iter << ", res: " << mean_res << std::endl;

        // 计算delta_x
        Eigen::Matrix<double, 6, 6> H = H_and_err.first;
        Eigen::Matrix<double, 6, 1> err = H_and_err.second;
        Eigen::Matrix<double, 6, 1> dx = H.inverse() * err;

        // 更新 final_T
        final_T_.block<3, 3>(0, 0) = final_T_.block<3, 3>(0, 0) * manifold_math::Exp<double>(dx.head<3>());
        final_T_.block<3, 1>(0, 3) += dx.tail<3>();

        last_mean_residual = mean_res;

      } else {
        std::cout << "[point2point] iter: " << iter << ", res: " << mean_res << std::endl;
        std::cout << "  [Error] registration gradient descent anomaly, ready to exit..." << std::endl;
        break;
      }
    }
  }

  // 获取最终的外参
  void getRegistrationTransform(Eigen::Matrix4d &option_transform) override { option_transform = final_T_; }

  // 获取origin变换后的点云
  void getTransformedOriginCloud(const pcl::PointCloud<pcl::PointXYZI>::Ptr &transformed_cloud) override {
    pcl::transformPointCloud(*source_cloud_ptr_, *transformed_cloud, final_T_);
  };

};
}

#endif //EIGEN_ICP_POINT2POINT_H
