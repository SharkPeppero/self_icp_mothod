//
// Created by westwell on 25-1-15.
//

#ifndef EIGEN_ICP_SVD_ALIGNED_H
#define EIGEN_ICP_SVD_ALIGNED_H

#include "registration_base.h"

/**
 * 基于SVD实现点云配准
 */

namespace Registration {

class SVDAligned : public RegistrationBase {
 public:
  SVDAligned() {
    // 当前ICP的模式
    registration_mode_ = RegistrationMode::SVD_ALIGNED;

    // SVD Registration的配准参数
    iterations_ = 10;
    epsilon_ = 1e-6;
    nearest_dist_ = 5.0;
    use_tbb_flag_ = false;
    use_log_flag_ = false;

    // SVD最终的结果
    convergence_flag_ = true;
    final_T_ = Eigen::Matrix4d::Identity();

    // 输入的点云数据
    init_T_ = Eigen::Matrix4d::Identity();
    source_cloud_ptr_ = boost::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
    target_cloud_ptr_ = boost::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
  }

  ~SVDAligned() override = default;

  // 参数配置接口
  void setIterations(int iterations) override { iterations_ = iterations; }
  void setEpsilon(double epsilon) override { epsilon_ = epsilon; }
  void setNearestDist(double nearest_dist) override { nearest_dist_ = nearest_dist; }
  void setTBBFlag(bool use_tbb_flag) override { use_tbb_flag_ = use_tbb_flag; }
  void setLogFlag(bool use_log_flag) override { use_log_flag_ = use_log_flag; }

  // 输入数据接口
  void setSourceCloud(const pcl::PointCloud<pcl::PointXYZI>::Ptr &source_cloud_ptr) override { source_cloud_ptr_ = source_cloud_ptr; }
  void setTargetCloud(const pcl::PointCloud<pcl::PointXYZI>::Ptr &target_cloud_ptr) override { target_cloud_ptr_ = target_cloud_ptr; }
  void setInitT(const Eigen::Matrix4d &init_T) override { init_T_ = init_T; }

  // 打印参数
  void logParameter() override {
    std::cout << "  Registration Mode: " << getRegistrationMode(registration_mode_) << std::endl;
    std::cout << "  iterations: " << iterations_ << std::endl;
    std::cout << "  epsilon: " << epsilon_ << std::endl;
    std::cout << "  nearest_dist: " << nearest_dist_ << std::endl;
    std::cout << "  use_tbb_flag: " << (use_tbb_flag_ ? "true" : "false") << std::endl;
    std::cout << "  init_T_target_source: " << std::endl << init_T_.matrix() << std::endl;
  }

  // 输入的target以及source是已经配对好的
  bool correspondHandle() {

    // 断言检测
    assert(target_cloud_ptr_->points.empty());
    assert(source_cloud_ptr_->points.empty());
    assert(target_cloud_ptr_->points.size() == source_cloud_ptr_->points.size());

    auto start = std::chrono::steady_clock::now();

    // 组织SVD数据
    std::vector<Eigen::Vector3d> origin_correspond; // 原始点云的激光点数据
    std::vector<Eigen::Vector3d> target_correspond; // 目标点点云激光点数据
    for (size_t i = 0; i < source_cloud_ptr_->points.size(); ++i) {
      origin_correspond.emplace_back(source_cloud_ptr_->points[i].x,
                                     source_cloud_ptr_->points[i].y,
                                     source_cloud_ptr_->points[i].z);
      target_correspond.emplace_back(target_cloud_ptr_->points[i].x,
                                     target_cloud_ptr_->points[i].y,
                                     target_cloud_ptr_->points[i].z);
    }

    // Step3: 组织SVD求解前的数据形式
    //      P(origin_cloud数据)     Q(target_cloud数据)
    //         x0 x1 x2 ... xn        x0 x1 x2 ... xn
    //         y0 y1 y2 ... yn        y0 y1 y2 ... yn
    //         z0 z1 z2 ... zn        z0 z1 z2 ... zn
    Eigen::MatrixXd P(3, origin_correspond.size());
    Eigen::MatrixXd Q(3, target_correspond.size());
    for (size_t i = 0; i < origin_correspond.size(); ++i) {
      P.col(i) = origin_correspond[i];
      Q.col(i) = target_correspond[i];
    }

    // Step4：进行SVD求解变换矩阵
    //  去除质心
    Eigen::Vector3d p_mean = P.rowwise().mean();
    Eigen::Vector3d q_mean = Q.rowwise().mean();
    Eigen::MatrixXd one_matrix(1, P.cols());
    one_matrix.setOnes();
    auto p_means = p_mean * one_matrix;
    auto q_means = q_mean * one_matrix;
    P = P - p_means;
    Q = Q - q_means;

    // 计算协方差
    Eigen::Matrix3d W = Q * P.transpose(); // 去质心后的目标点云(3xn) 去质心后的原始点云
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(W, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();

    // 进行旋转矩阵求解，这里考虑到vUT的行列式小于0的问题，
    // A=UΣVT
    // SVD分解的过程，有时会产生 det(V*UT) = -1，代表一个反射
    // https://www.liuxiao.org/2019/08/%e4%bd%bf%e7%94%a8-svd-%e6%96%b9%e6%b3%95%e6%b1%82%e8%a7%a3-icp-%e9%97%ae%e9%a2%98/
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();       // 设为对角矩阵初始值
    R(2, 2) = (V * U.transpose()).determinant();           // 设置调整矩阵元素
    R = V * R * U.transpose();                             // 这样可以保证将反射矩阵变换成旋转矩阵

    // 进行t矩阵计算
    Eigen::Vector3d t = q_mean - R * p_mean;

    final_T_.block<3, 3>(0, 0) = R;
    final_T_.block<3, 1>(0, 3) = t;

    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Operation Cost " << duration.count() << " ms." << std::endl;

    return convergence_flag_;
  }

  // 没有初始配对点信息，利用kdtree进行最近邻查询的配准
  bool Handle() override {

    // 参数打印
    if (use_log_flag_)
      logParameter();

    // 断言
    assert(!source_cloud_ptr_->points.empty());
    assert(!target_cloud_ptr_->points.empty());

    // 配置target点云的kdtree
    pcl::KdTreeFLANN<pcl::PointXYZI> target_KDtree;
    target_KDtree.setInputCloud(target_cloud_ptr_);

    final_T_ = init_T_;
    double last_mean_res = 1e10;

    // 迭代 SVD Registration
    auto total_start = std::chrono::steady_clock::now();
    for (int iter = 0; iter < iterations_; ++iter) {
      auto iteration_start = std::chrono::steady_clock::now();

      auto manager_data_start = std::chrono::steady_clock::now();
      // Step1: source点云进行欧式变换
      pcl::PointCloud<pcl::PointXYZI>::Ptr transformed_cloud_ptr(new pcl::PointCloud<pcl::PointXYZI>());
      pcl::transformPointCloud(*source_cloud_ptr_, *transformed_cloud_ptr, final_T_);

      // Step2：进行最近邻查询，实现点云配对
      std::vector<Eigen::Vector3d> origin_correspond; // 原始点云的激光点数据
      std::vector<Eigen::Vector3d> target_correspond; // 目标点点云激光点数据
      int effective_cnt = 0;
      double latest_mean_res = 0.0;
      for (size_t i = 0; i < transformed_cloud_ptr->points.size(); ++i) {
        std::vector<int> idx;
        std::vector<float> dist;
        target_KDtree.nearestKSearch(transformed_cloud_ptr->points[i], 1, idx, dist);
        if (dist.front() <= nearest_dist_) {
          origin_correspond.emplace_back(source_cloud_ptr_->points[i].x, source_cloud_ptr_->points[i].y, source_cloud_ptr_->points[i].z);
          target_correspond.emplace_back(target_cloud_ptr_->points[idx[0]].x, target_cloud_ptr_->points[idx[0]].y, target_cloud_ptr_->points[idx[0]].z);

          Eigen::Vector3d err(target_cloud_ptr_->points[idx[0]].x - transformed_cloud_ptr->points[i].x,
                              target_cloud_ptr_->points[idx[0]].y - transformed_cloud_ptr->points[i].y,
                              target_cloud_ptr_->points[idx[0]].z - transformed_cloud_ptr->points[i].z);
          effective_cnt++;
          latest_mean_res = latest_mean_res + (err.norm() - latest_mean_res) / effective_cnt;
        }
      }
      auto manager_data_end = std::chrono::steady_clock::now();
      auto manager_data_duration = std::chrono::duration_cast<std::chrono::milliseconds>(manager_data_end - manager_data_start);


      auto solve_start = std::chrono::steady_clock::now();
      // Step3: 组织SVD求解前的数据形式
      //        P(origin_cloud数据)     Q(target_cloud数据)
      //          x0 x1 x2 ... xn        x0 x1 x2 ... xn
      //          y0 y1 y2 ... yn        y0 y1 y2 ... yn
      //          z0 z1 z2 ... zn        z0 z1 z2 ... zn
      Eigen::MatrixXd P(3, origin_correspond.size());
      Eigen::MatrixXd Q(3, target_correspond.size());
      for (size_t i = 0; i < origin_correspond.size(); ++i) {
        P.col(i) = origin_correspond[i];
        Q.col(i) = target_correspond[i];
      }

      // Step4：进行SVD求解变换矩阵
      //  去除质心
      Eigen::Vector3d p_mean = P.rowwise().mean();
      Eigen::Vector3d q_mean = Q.rowwise().mean();
      Eigen::MatrixXd one_matrix(1, P.cols());
      one_matrix.setOnes();
      auto p_means = p_mean * one_matrix;
      auto q_means = q_mean * one_matrix;
      P = P - p_means;
      Q = Q - q_means;

      // 计算协方差
      Eigen::Matrix3d W = Q * P.transpose(); // 去质心后的目标点云(3xn) 去质心后的原始点云
      Eigen::JacobiSVD<Eigen::Matrix3d> svd(W, Eigen::ComputeFullU | Eigen::ComputeFullV);
      Eigen::Matrix3d U = svd.matrixU();
      Eigen::Matrix3d V = svd.matrixV();

      // 进行旋转矩阵求解，这里考虑到vUT的行列式小于0的问题，
      // A=UΣVT
      // SVD分解的过程，有时会产生 det(V*UT) = -1，代表一个反射
      // https://www.liuxiao.org/2019/08/%e4%bd%bf%e7%94%a8-svd-%e6%96%b9%e6%b3%95%e6%b1%82%e8%a7%a3-icp-%e9%97%ae%e9%a2%98/
      Eigen::Matrix3d R = Eigen::Matrix3d::Identity();       // 设为对角矩阵初始值
      R(2, 2) = (V * U.transpose()).determinant();           // 设置调整矩阵元素
      R = V * R * U.transpose();                             // 这样可以保证将反射矩阵变换成旋转矩阵

      // 进行t矩阵计算
      Eigen::Vector3d t = q_mean - R * p_mean;

      auto solve_end = std::chrono::steady_clock::now();
      auto solve_duration = std::chrono::duration_cast<std::chrono::milliseconds>(solve_end - solve_start);

      // 结果的check
      // 计算 epsilon
      double latest_epsilon = (final_T_.block<3, 1>(0, 3) - t).norm();
      if(latest_epsilon > epsilon_ && latest_mean_res < last_mean_res){
        // 更新外参
        final_T_.block<3, 3>(0, 0) = R;
        final_T_.block<3, 1>(0, 3) = t;

        last_mean_res = latest_mean_res;
      } else {
        // 迭代终止判断
        //  位姿变换足够小
        //  总体的残差不再下降
        if(latest_epsilon < epsilon_){
          std::cout << "iteration epsilon is enough small, break." << std::endl;
        }
        if(latest_mean_res > last_mean_res){
          std::cout << "mean res error, break." << std::endl;
        }
        break;
      }

      auto iteration_end = std::chrono::steady_clock::now();
      auto iteration_duration = std::chrono::duration_cast<std::chrono::milliseconds>(iteration_end - iteration_start);

      if (use_log_flag_)
        std::cout << std::fixed << std::setprecision(9)
                  << "Registration Mode: " << getRegistrationMode(registration_mode_)
                  << " | iteration: " << iter
                  << " | epsilon: " << latest_epsilon
                  << " | mean_res: " << latest_mean_res
                  << " | manager data cost time(ms): " << manager_data_duration.count()
                  << " | solve data cost time(ms): " << solve_duration.count()
                  << " | iteration cost time(ms): " << iteration_duration.count()
                  << std::endl;

    }
    auto total_end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start);
    if(use_log_flag_)
      std::cout << std::fixed << std::setprecision(9) << "Registration Mode: " << getRegistrationMode(registration_mode_) << " | Total Cost Time: " << duration.count() << " ms." << std::endl;

    // todo: 添加点云配准的收敛判断
    return convergence_flag_;
  }

  // 获取最终的外参
  void getRegistrationTransform(Eigen::Matrix4d &option_transform) override { option_transform = final_T_; }

  // 获取origin变换后的点云
  void getTransformedOriginCloud(const pcl::PointCloud<pcl::PointXYZI>::Ptr &transformed_cloud) override {
    pcl::transformPointCloud(*source_cloud_ptr_, *transformed_cloud, final_T_);
  };
};

}

#endif //EIGEN_ICP_SVD_ALIGNED_H
