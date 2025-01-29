//
// Created by westwell on 25-1-21.
//

#ifndef EIGEN_ICP_INCLUDE_ICP_MOTHED_NICP_REGISTRATION_H_
#define EIGEN_ICP_INCLUDE_ICP_MOTHED_NICP_REGISTRATION_H_

#include "icp_mothed/registration_base.h"

namespace Registration {

/**
 * @brief
 *  核心算法逻辑：
 *  !!! 关于最小二乘多个error误差函数如何编写残差函数，
 *          点对之间的距离残差，
 *          点对之间的曲率残差，
 *          点对之间的法向量从残差
 *
 *      数据关联的原则： 点对的距离、点对的方向量
 *      基础属性计算函数设计:
 *       1、增量计算当前对机器附近点的协方差矩阵
 *       2、SVD计算法向量 （最小特征值对应的方向向量）
 *          计算并提取点云曲率（ Lamada = lamada1  / (lamada1 + lamada2 + lamada3 )）
 *
 *       3、数据关联原则：
 *          点距离超过阈值
 *          曲率接近
 *          方向向量接近
 *       4、目标函数如何计算
 *
 */

class NICPRegistration : public RegistrationBase {
 public:
  NICPRegistration() {
    registration_mode_ = RegistrationMode::NICP;

    iterations_ = 10;
    epsilon_ = 1e-6;
    nearest_dist_ = 5.0;
    use_tbb_flag_ = false;
    use_log_flag_ = false;

    init_T_ = Eigen::Matrix4d::Identity();
    source_cloud_ptr_ = boost::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
    target_cloud_ptr_ = boost::make_shared<pcl::PointCloud<pcl::PointXYZI>>();

    convergence_flag_ = true;
    final_T_ = Eigen::Matrix4d::Identity();

    nearest_pts_num_ = 10;
    preprocessed_target_pc_ptr_ = boost::make_shared<pcl::PointCloud<pcl::PointXYZINormal>>();
    preprocessed_source_pc_ptr_ = boost::make_shared<pcl::PointCloud<pcl::PointXYZINormal>>();
  }

  ~NICPRegistration() override = default;

  // 配置参数
  void setIterations(int iterations) override { iterations_ = iterations; }
  void setEpsilon(double epsilon) override { epsilon_ = epsilon; }
  void setNearestDist(double nearest_dist) override { nearest_dist_ = nearest_dist; }
  void setTBBFlag(bool use_tbb_flag) override { use_tbb_flag_ = use_tbb_flag; }
  void setLogFlag(bool use_log_flag) override { use_log_flag_ = use_log_flag; }
  void logParameter() override {
    std::cout << "  Registration Mode: " << getRegistrationMode(registration_mode_) << std::endl;
    std::cout << "    iterations: " << iterations_ << std::endl;
    std::cout << "    epsilon: " << epsilon_ << std::endl;
    std::cout << "    nearest_dist: " << nearest_dist_ << std::endl;
    std::cout << "    use_tbb_flag: " << (use_tbb_flag_ ? "true" : "false") << std::endl;
    std::cout << "    init_T_target_source: " << std::endl << init_T_.matrix() << std::endl;
  }

  // 处理函数
  bool Handle() override {

    assert(target_cloud_ptr_ != nullptr);
    assert(source_cloud_ptr_ != nullptr);
    assert(!target_cloud_ptr_->points.empty());
    assert(!source_cloud_ptr_->points.empty());

    preprocessed_target_pc_ptr_ = processPointCloud(target_cloud_ptr_);
    preprocessed_source_pc_ptr_ = processPointCloud(source_cloud_ptr_);

//    pcl::io::savePCDFile("");


  }

  // 配置输入参数
  void setSourceCloud(const pcl::PointCloud<pcl::PointXYZI>::Ptr &source_cloud_ptr) override {
    source_cloud_ptr_ = source_cloud_ptr;
  }
  void setTargetCloud(const pcl::PointCloud<pcl::PointXYZI>::Ptr &target_cloud_ptr) override {
    target_cloud_ptr_ = target_cloud_ptr;
  }
  void setInitT(const Eigen::Matrix4d &init_T) override { init_T_ = init_T; }

  // 获取结果
  void getInitTransform(Eigen::Matrix4d &init_T) override { init_T = init_T_; }
  void getRegistrationTransform(Eigen::Matrix4d &option_transform) override { option_transform = final_T_; }
  void getTransformedOriginCloud(const pcl::PointCloud<pcl::PointXYZI>::Ptr &transformed_cloud) override {
    pcl::transformPointCloud(*source_cloud_ptr_, *transformed_cloud, final_T_);
  };

  // target点云预处理
  //  计算点云法向量以及曲率
  pcl::PointCloud<pcl::PointXYZINormal>::Ptr processPointCloud(const pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud) {
    // 创建TargetCloud KdTree对象
    pcl::KdTreeFLANN<pcl::PointXYZI> kdtree;
    kdtree.setInputCloud(cloud);

    pcl::PointCloud<pcl::PointXYZINormal>::Ptr handled_point_cloud_ptr(new pcl::PointCloud<pcl::PointXYZINormal>());

    // 遍历点云中的每一个点
    for (size_t i = 0; i < cloud->points.size(); ++i) {
      pcl::PointXYZI searchPoint = cloud->points[i];

      // 进行最近邻搜索，搜索半径为10cm，最多返回10个点
      std::vector<int> pointIdxNKNSearch(nearest_pts_num_);
      std::vector<float> pointNKNSquaredDistance(nearest_pts_num_);
      int found = kdtree.radiusSearch(searchPoint, 0.10, pointIdxNKNSearch, pointNKNSquaredDistance, nearest_pts_num_);
      if (found < nearest_pts_num_) {
        continue;
      }

      // 计算均值以及协方差矩阵
      Eigen::Matrix3f covariance_matrix;
      Eigen::Vector4f centroid;
      pcl::compute3DCentroid(*cloud, pointIdxNKNSearch, centroid);
      pcl::computeCovarianceMatrixNormalized(*cloud, pointIdxNKNSearch, centroid, covariance_matrix);

      // 对协方差矩阵进行SVD分解
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance_matrix);
      Eigen::Vector3f eigenvalues = eigen_solver.eigenvalues();
      Eigen::Matrix3f eigenvectors = eigen_solver.eigenvectors();

      // 特征值从小到大分别为 l1, l2, l3
      float l1 = eigenvalues(0);
      float l2 = eigenvalues(1);
      float l3 = eigenvalues(2);

      // 计算曲率
      float curvature = l1 / (l1 + l2 + l3);

      // 计算法向量
      Eigen::Vector3f normal_vec = eigenvectors.col(0);

      pcl::PointXYZINormal pcl_tmp;
      pcl_tmp.x = searchPoint.x;
      pcl_tmp.y = searchPoint.y;
      pcl_tmp.z = searchPoint.z;
      pcl_tmp.intensity = searchPoint.intensity;
      pcl_tmp.curvature = curvature;
      pcl_tmp.normal_x = normal_vec.x();
      pcl_tmp.normal_x = normal_vec.y();
      pcl_tmp.normal_x = normal_vec.z();

      handled_point_cloud_ptr->points.push_back(pcl_tmp);
    }

    return handled_point_cloud_ptr;
  }

  int nearest_pts_num_;
  pcl::PointCloud<pcl::PointXYZINormal>::Ptr preprocessed_target_pc_ptr_ = nullptr;
  pcl::PointCloud<pcl::PointXYZINormal>::Ptr preprocessed_source_pc_ptr_ = nullptr;

};

}

#endif //EIGEN_ICP_INCLUDE_ICP_MOTHED_NICP_REGISTRATION_H_
