//
// Created by westwell on 25-1-17.
//

#ifndef EIGEN_ICP_NDT_ALIGNED_H
#define EIGEN_ICP_NDT_ALIGNED_H

#include "registration_base.h"
#include "unordered_map"
/**
 * 基于NDT实现点云配准
 *  1.将目标点云进行体素管理，计算每个体素内的局部统计信息——均值和方差
 *  2.配准时，先计算每一个点应该在哪一个体素内，建立该点与该体素的残差
 *  3.利用高斯牛顿对位姿进行迭代优化 JT * J * deltaX = -1 * JT * err
 * 最核心的问题，err_function如何表达
 *  source激光点变换后落在栅格内的残差可以表示为:
 *      err = Rq + t - q
 *  联系上栅格内的方差信息，得到加权的最小二乘问题
 *      argmin_R_t =
 */

// ============================= 体素管理 ======================
// 哈希映射参数
#define HASH_P 116101
#define MAX_N 10000000019
#define SMALL_EPS 1e-10

// 体素位置的三维索引(整形)
class VOXEL_LOC {
 public:
  int64_t x, y, z;
  explicit VOXEL_LOC(int64_t vx = 0, int64_t vy = 0, int64_t vz = 0) : x(vx), y(vy), z(vz) {}
  bool operator==(const VOXEL_LOC &other) const { return (x == other.x && y == other.y && z == other.z); }
};

namespace std {
template<>
struct hash<VOXEL_LOC> {
  size_t operator()(const VOXEL_LOC &s) const {
    using std::size_t;
    using std::hash;
    // return (((hash<int64_t>()(s.z)*HASH_P)%MAX_N + hash<int64_t>()(s.y))*HASH_P)%MAX_N + hash<int64_t>()(s.x);
    long long index_x, index_y, index_z;
    double cub_len = 0.125;
    index_x = int(round(floor((s.x) / cub_len + SMALL_EPS)));
    index_y = int(round(floor((s.y) / cub_len + SMALL_EPS)));
    index_z = int(round(floor((s.z) / cub_len + SMALL_EPS)));
    return (((((index_z * HASH_P) % MAX_N + index_y) * HASH_P) % MAX_N) + index_x) % MAX_N;
  }
};
}

// 体素内部的信息
//  体素只接受push 以及 reset
//    push会递归计算体素内部的均值 协方差 信息矩阵
//    reset会重置当前体素的均值 协方差 信息矩阵
struct VoxelData {
  VoxelData() = default;

  // 增量更新均值和协方差
  //  U_n+1 = Un + (X_n+1 - U_n) / (n + 1)
  //  补充 gpt
  void push(const Eigen::Vector3d &input_point) {
    effective_cnt_++;

    // 增量更新均值和协方差
    if (effective_cnt_ == 1) {
      // 初始化均值和协方差矩阵
      mean_ = input_point;
      covariance_matrix_ = Eigen::Matrix3d::Zero();
    } else {
      Eigen::Vector3d diff = input_point - mean_;  // 当前点与均值的差
      // 增量更新协方差矩阵 维纳公式（Welford's method）推广
      covariance_matrix_ += (effective_cnt_ - 1.0) / effective_cnt_ * (diff * diff.transpose());
      // 更新均值
      mean_ = mean_ + diff / effective_cnt_;  // 更新均值
    }

    // 增量更新均值和方差
/*    if (effective_cnt_ == 1) {
      covariance_matrix_ = Eigen::Matrix3d::Zero();  // 初始化协方差矩阵为零
      mean_ = input_point;  // 第一个点直接作为均值
    } else {
      Eigen::Vector3d diff = input_point - mean_;  // 当前点与均值的差
      double factor = 1.0 / effective_cnt_;  // 归一化因子

      // 更新协方差矩阵的对角线元素
      covariance_matrix_.diagonal() =
          effective_cnt_ / (effective_cnt_ - 1.0) * covariance_matrix_.diagonal() +
              (effective_cnt_ - 1.0) * (diff.array().square()).matrix();

      // 更新均值
      mean_ += factor * diff;
    }*/
  }

  // 计算信息矩阵作为加权的依据
  Eigen::Matrix3d getInformation(){
    return  (covariance_matrix_ + Eigen::Matrix3d::Identity() * 1e-3).inverse();
  }

  // 重置栅格内部的局部信息
  void reset() {
    effective_cnt_ = 0;
    mean_ = Eigen::Vector3d::Zero();
    covariance_matrix_ = Eigen::Matrix3d::Zero();
  }

  size_t effective_cnt_ = 0;
  Eigen::Vector3d mean_ = Eigen::Vector3d::Zero();                // 体素内均值
  Eigen::Matrix3d covariance_matrix_ = Eigen::Matrix3d::Zero();   // 体素内协方差矩阵
};

// 最近邻体素查找的原则
enum NearbyType {
  CENTER = 0,
  NEARBY6,
};

namespace Registration {

class NDTAligned : public RegistrationBase {
 public:
  NDTAligned() {
    // 当前配准的模式
    registration_mode_ = RegistrationMode::NDT_ALIGNED;

    // NDT Registration的配准参数
    iterations_ = 10;       // 迭代次数
    epsilon_ = 1e-6;        // 迭代步长
    nearest_dist_ = 5.0;    // 最近邻查询(用不上)
    use_tbb_flag_ = false;
    use_log_flag_ = false;

    // SVD最终的结果
    convergence_flag_ = true;
    final_T_ = Eigen::Matrix4d::Identity();

    // 输入的点云数据
    init_T_ = Eigen::Matrix4d::Identity();
    source_cloud_ptr_ = boost::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
    target_cloud_ptr_ = boost::make_shared<pcl::PointCloud<pcl::PointXYZI>>();

    // 初始化NDT的参数
    voxel_size_ = 0.01;
    min_pts_in_voxel_ = 10;
    nearby_type_ = NearbyType::CENTER;

  }

  ~NDTAligned() override = default;

// 参数配置接口
  void setIterations(int iterations) override { iterations_ = iterations; }
  void setEpsilon(double epsilon) override { epsilon_ = epsilon; }
  void setNearestDist(double nearest_dist) override { nearest_dist_ = nearest_dist; }
  void setTBBFlag(bool use_tbb_flag) override { use_tbb_flag_ = use_tbb_flag; }
  void setLogFlag(bool use_log_flag) override { use_log_flag_ = use_log_flag; }

// 输入数据接口
  void setSourceCloud(const pcl::PointCloud<pcl::PointXYZI>::Ptr &source_cloud_ptr) override {
    source_cloud_ptr_ = source_cloud_ptr;
  }
  void setTargetCloud(const pcl::PointCloud<pcl::PointXYZI>::Ptr &target_cloud_ptr) override {
    target_cloud_ptr_ = target_cloud_ptr;
  }
  void setInitT(const Eigen::Matrix4d &init_T) override {
    std::cout << __LINE__ << std::endl;
    init_T_ = init_T;
    std::cout << init_T.matrix() << std::endl;
    std::cout << init_T_.matrix() << std::endl;
  }

  // 进行target点云的Voxel构建
  //  构建target点云体素栅格，
  //    增量更新均值以及协方差
  //    剔除栅格内激光点小于指定苏木
  void buildVoxels() {

    // 对target点云进行体素划分并增量计算均值以及协方差矩阵
    for (auto & pt : target_cloud_ptr_->points) {
      // 计算每一个激光点对应的voxel索引
      Eigen::Vector3d pt_origin(pt.x, pt.y, pt.z);
      float loc_xyz[3];
      for (int j = 0; j < 3; j++) {
        loc_xyz[j] = static_cast<float>(pt_origin[j]) / voxel_size_;
        if (loc_xyz[j] < 0)
          loc_xyz[j] -= 1.0;
      }
      VOXEL_LOC position((int64_t) loc_xyz[0], (int64_t) loc_xyz[1], (int64_t) loc_xyz[2]);

      // 对体素进行增量更新
      auto iter = feature_map.find(position);
      if (iter != feature_map.end()) {
        feature_map[position].push(pt_origin);
      } else {
        VoxelData voxel_data;
        voxel_data.push(pt_origin);
        feature_map[position] = voxel_data;
      }
    }
    std::cout << "  初始Voxel分割数目: " << feature_map.size() << std::endl;

    // 剔除小于指定大小的体素
    for (auto iter = feature_map.begin(); iter != feature_map.end(); iter++) {
      if (iter->second.effective_cnt_ < min_pts_in_voxel_) {
        feature_map.erase(iter->first);
      }
    }
    std::cout << "  剔除后Voxel分割数目: " << feature_map.size() << std::endl;

  }

  // 初始化Correspondence的方式
  void buildNearbyType() {
    if (nearby_type_ == NearbyType::CENTER) {
      nearby_search_range_.emplace_back(0, 0, 0);
    } else if (nearby_type_ == NearbyType::NEARBY6) {
      nearby_search_range_.emplace_back(0, 0, 0);
      nearby_search_range_.emplace_back(1, 0, 0);
      nearby_search_range_.emplace_back(-1, 0, 0);
      nearby_search_range_.emplace_back(0, 1, 0);
      nearby_search_range_.emplace_back(0, -1, 0);
      nearby_search_range_.emplace_back(0, 0, 1);
      nearby_search_range_.emplace_back(0, 0, -1);
    } else {
      std::cerr << "输入了无效的NearbyType." << std::endl;
    }
  }

  // 打印参数
  void logParameter() override {
    std::cout << "  Registration Mode: " << getRegistrationMode(registration_mode_) << std::endl;
    std::cout << "  iterations: " << iterations_ << std::endl;
    std::cout << "  epsilon: " << epsilon_ << std::endl;
    std::cout << "  nearest_dist: " << nearest_dist_ << std::endl;
    std::cout << "  use_tbb_flag: " << (use_tbb_flag_ ? "true" : "false") << std::endl;
    std::cout << "  init_T_target_source: " << std::endl << init_T_.matrix() << std::endl;
  }

  // 没有初始配对点信息，利用kdtree进行最近邻查询的配准
  bool Handle() override {
    // 断言
    assert(target_cloud_ptr_ != nullptr);
    assert(source_cloud_ptr_ != nullptr);
    assert(!target_cloud_ptr_->points.empty());
    assert(!source_cloud_ptr_->points.empty());

    std::cout << "init_T: " << std::endl;
    std::cout << init_T_.matrix() << std::endl;

    final_T_ = init_T_;
    std::cout << "final_T: " << std::endl;
    std::cout << final_T_.matrix() << std::endl;

    // 拆解target点云的voxel
    buildVoxels();
    std::cout << "--完成Target点云体素分解,voxel size: " << feature_map.size() << std::endl;
    pcl::PointCloud<pcl::PointXYZI>::Ptr target_voxel_cloud(new pcl::PointCloud<pcl::PointXYZI>());
    for (const auto &voxel_tmp : feature_map) {
      pcl::PointXYZI pt_tmp;
      pt_tmp.x = voxel_tmp.second.mean_.x();
      pt_tmp.y = voxel_tmp.second.mean_.y();
      pt_tmp.z = voxel_tmp.second.mean_.z();
      pt_tmp.intensity = 255.0;
      target_voxel_cloud->push_back(pt_tmp);
    }
    pcl::io::savePCDFileBinary("./target_voxel.pcd", *target_voxel_cloud);

    // 构建体素关联的手段
    buildNearbyType();
    std::cout << "--完成NearbyType分解,nearby range size: " << nearby_search_range_.size() << std::endl;

    // 迭代残差
    double last_mean_residual = 1e10;

    // 迭代优化NDT
    for (int iteration = 0; iteration < iterations_; iteration++) {

      std::cout << "当前是第" << iteration << "轮迭代." << std::endl;

      // 本轮迭代的高斯牛顿 H delta_x = b
      Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
      Eigen::Matrix<double, 6, 1> b = Eigen::Matrix<double, 6, 1>::Zero();
      int effective_cnt = 0;
      double latest_residual_mean = 0.0;

      // 对原始点云进行变换
      // 初始化每一轮外参变换后的transformed_cloud
      pcl::PointCloud<pcl::PointXYZI>::Ptr transformed_cloud_ptr(new pcl::PointCloud<pcl::PointXYZI>());
      pcl::transformPointCloud(*source_cloud_ptr_, *transformed_cloud_ptr, final_T_);

      // 构建残差表达：构建加权的最小二乘
      //  1、体素内均值的误差作为基础
      //  2、体素内的协方差矩阵作为加权
      pcl::PointCloud<pcl::PointXYZI>::Ptr debug_cloud(new pcl::PointCloud<pcl::PointXYZI>());
      for (size_t i = 0; i < transformed_cloud_ptr->points.size(); ++i) {

        // 计算每一个激光点对应的voxel索引
        auto& origin_point = source_cloud_ptr_->points[i];
        Eigen::Vector3d origin_point_eigen(origin_point.x, origin_point.y, origin_point.z);

        auto& transformed_point = transformed_cloud_ptr->points[i];
        Eigen::Vector3d transformed_point_eigen(transformed_point.x, transformed_point.y, transformed_point.z);

        // 进行体素Correspondence并组织高斯牛顿
        float loc_xyz[3];
        for (int j = 0; j < 3; j++) {
          loc_xyz[j] = static_cast<float>(transformed_point_eigen[j]) / voxel_size_;
          if (loc_xyz[j] < 0)
            loc_xyz[j] -= 1.0;
        }
        VOXEL_LOC position((int64_t) loc_xyz[0], (int64_t) loc_xyz[1], (int64_t) loc_xyz[2]);

        for (const auto &nearby_grid : nearby_search_range_) {
          // 计算当前调整后的体素索引
          VOXEL_LOC position_key(position.x + nearby_grid.x(),
                                 position.y + nearby_grid.y(),
                                 position.z + nearby_grid.z());

          // 尝试获取对应体素信息
          auto iter = feature_map.find(position_key);
          if (iter != feature_map.end()) {

            //
            pcl::PointXYZI point_tmp;
            point_tmp.x = transformed_point_eigen.x();
            point_tmp.y = transformed_point_eigen.y();
            point_tmp.z = transformed_point_eigen.z();
            point_tmp.intensity = iteration;
            debug_cloud->push_back(point_tmp);

            // 构建error
            Eigen::Vector3d err = transformed_point_eigen - iter->second.mean_;

            // 构建加权error
            double res = err.transpose() * iter->second.getInformation() * err;

            effective_cnt++;
            latest_residual_mean = latest_residual_mean + (res - latest_residual_mean) / effective_cnt;

            // 构建雅阁比矩阵
            Eigen::Matrix<double, 3, 6> J = Eigen::Matrix<double, 3, 6>::Zero();
            J.block<3, 3>(0, 0) =
                -1.0 * final_T_.block<3, 3>(0, 0).matrix() * manifold_math::skew_sym_mat(origin_point_eigen);
            J.block<3,3>(0,3) = Eigen::Matrix3d::Identity();

            // 构建海森矩阵部分进行累加
            H += J.transpose() * iter->second.getInformation() * J;
            b += -1.0 * J.transpose() * iter->second.getInformation() * err;
          }
        }

      }

      std::string pcd_path = std::string("./") + std::to_string(iteration) + ".pcd";
      pcl::io::savePCDFileBinary(pcd_path, *debug_cloud);

      // 进行本轮DNT的求解
      Eigen::Matrix<double, 6, 1> delta_x = H.inverse() * b;

      // 整体的residual残差check
      double latest_epsilon = delta_x.tail<3>().norm();
      std::cout << "本轮迭代的residual: " << latest_residual_mean << std::endl;
      std::cout << "本轮迭代的epsilon: " << latest_epsilon << std::endl;
      if (latest_residual_mean < last_mean_residual || latest_epsilon > epsilon_) {
        final_T_.block<3, 3>(0, 0) *= manifold_math::Exp<double>(delta_x.head<3>());
        final_T_.block<3, 1>(0, 3) += delta_x.tail<3>();
        last_mean_residual = latest_residual_mean;
      } else {
        if(latest_epsilon < epsilon_){
          std::cout << "iteration epsilon is enough small, break." << std::endl;
        }
        if(latest_residual_mean > last_mean_residual){
          std::cout << "mean res error, break." << std::endl;
        }
        break;
      }


    }

    return convergence_flag_;
  }

  // 获取最终的外参
  void getRegistrationTransform(Eigen::Matrix4d &option_transform) override { option_transform = final_T_; }

  // 获取origin变换后的点云
  void getTransformedOriginCloud(const pcl::PointCloud<pcl::PointXYZI>::Ptr &transformed_cloud) override {
    pcl::transformPointCloud(*source_cloud_ptr_, *transformed_cloud, final_T_);
  };

  // build voxel的相关参数
  std::unordered_map<VOXEL_LOC, VoxelData> feature_map;
  float voxel_size_;
  size_t min_pts_in_voxel_;

  // 构建残差的Correspondence方式
  NearbyType nearby_type_;
  std::vector<Eigen::Vector3i> nearby_search_range_;

};

}

#endif //EIGEN_ICP_NDT_ALIGNED_H
