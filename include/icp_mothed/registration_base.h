//
// Created by westwell on 25-1-15.
//

#ifndef EIGEN_ICP_MATCH_BASE_H
#define EIGEN_ICP_MATCH_BASE_H

#include "iostream"
#include "chrono"

#include "Eigen/Core"
#include "Eigen/Geometry"
#include "Eigen/Dense"

#include "pcl/point_cloud.h"
#include "pcl/point_types.h"
#include <pcl/common/transforms.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/io/pcd_io.h>

// Tools
namespace tools {
class Tictoc {
 private:
  std::chrono::steady_clock::time_point start_time;
  bool is_running;

 public:

  Tictoc() : is_running(false) {}

  void tic() {
    start_time = std::chrono::steady_clock::now();
    is_running = true;
  }

  // 停止计时器并返回经过的时间
  double toc() {
    if (is_running) {
      auto end_time = std::chrono::steady_clock::now();
      std::chrono::duration<double> elapsed = end_time - start_time;
      is_running = false;
      return elapsed.count();  // 返回经过的秒数
    } else {
      std::cerr << "Tictoc: Timer was not started!" << std::endl;
      return 0.0;
    }
  }

  // 返回当前经过的时间，不停止计时器
  double elapsed() {
    if (is_running) {
      auto current_time = std::chrono::steady_clock::now();
      std::chrono::duration<double> elapsed = current_time - start_time;
      return elapsed.count();
    } else {
      std::cerr << "Tictoc: Timer was not started!" << std::endl;
      return 0.0;
    }
  }
};
}

// 流行空间运算
namespace manifold_math {
// 计算反对称矩阵
template<typename T>
Eigen::Matrix<T, 3, 3> skew_sym_mat(const Eigen::Matrix<T, 3, 1> &v) {
  Eigen::Matrix<T, 3, 3> skew_sym_mat;
  skew_sym_mat << 0.0, -v[2], v[1], v[2], 0.0, -v[0], -v[1], v[0], 0.0;
  return skew_sym_mat;
}

// 李代数 ——> 李群
template<typename T>
Eigen::Matrix<T, 3, 3> Exp(const Eigen::Matrix<T, 3, 1> &ang) {
  T ang_norm = ang.norm();
  Eigen::Matrix<T, 3, 3> Eye3 = Eigen::Matrix<T, 3, 3>::Identity();
  if (ang_norm > 0.0000001) {
    Eigen::Matrix<T, 3, 1> r_axis = ang / ang_norm;
    Eigen::Matrix<T, 3, 3> K = skew_sym_mat(r_axis);
    /// Roderigous Tranformation
    return Eye3 + std::sin(ang_norm) * K + (1.0 - std::cos(ang_norm)) * K * K;
  } else {
    return Eye3;
  }
}

// 李代数 --->> 李群
template<typename T, typename Ts>
Eigen::Matrix<T, 3, 3> Exp(const Eigen::Matrix<T, 3, 1> &ang_vel, const Ts &dt) {
  T ang_vel_norm = ang_vel.norm();
  Eigen::Matrix<T, 3, 3> Eye3 = Eigen::Matrix<T, 3, 3>::Identity();

  if (ang_vel_norm > 0.0000001) {
    Eigen::Matrix<T, 3, 1> r_axis = ang_vel / ang_vel_norm;
    Eigen::Matrix<T, 3, 3> K;

    K << SKEW_SYM_MATRX(r_axis);

    T r_ang = ang_vel_norm * dt;

    /// Roderigous Tranformation
    return Eye3 + std::sin(r_ang) * K + (1.0 - std::cos(r_ang)) * K * K;
  } else {
    return Eye3;
  }
}

// 李代数到李群
template<typename T>
Eigen::Matrix<T, 3, 3> Exp(const T &v1, const T &v2, const T &v3) {
  T &&norm = sqrt(v1 * v1 + v2 * v2 + v3 * v3);
  Eigen::Matrix<T, 3, 3> Eye3 = Eigen::Matrix<T, 3, 3>::Identity();
  if (norm > 0.00001) {
    T r_ang[3] = {v1 / norm, v2 / norm, v3 / norm};
    Eigen::Matrix<T, 3, 3> K;
    K << SKEW_SYM_MATRX(r_ang);

    /// Roderigous Tranformation
    return Eye3 + std::sin(norm) * K + (1.0 - std::cos(norm)) * K * K;
  } else {
    return Eye3;
  }
}

/* Logrithm of a Rotation Matrix */
template<typename T>
Eigen::Matrix<T, 3, 1> Log(const Eigen::Matrix<T, 3, 3> &R) {
  T theta = (R.trace() > 3.0 - 1e-6) ? 0.0 : std::acos(0.5 * (R.trace() - 1));
  Eigen::Matrix<T, 3, 1> K(R(2, 1) - R(1, 2), R(0, 2) - R(2, 0), R(1, 0) - R(0, 1));
  return (std::abs(theta) < 0.001) ? (0.5 * K) : (0.5 * theta / std::sin(theta) * K);
}

template<typename T>
Eigen::Matrix<T, 3, 1> RotMtoEuler(const Eigen::Matrix<T, 3, 3> &rot) {
  T sy = sqrt(rot(0, 0) * rot(0, 0) + rot(1, 0) * rot(1, 0));
  bool singular = sy < 1e-6;
  T x, y, z;
  if (!singular) {
    x = atan2(rot(2, 1), rot(2, 2));
    y = atan2(-rot(2, 0), sy);
    z = atan2(rot(1, 0), rot(0, 0));
  } else {
    x = atan2(-rot(1, 2), rot(1, 1));
    y = atan2(-rot(2, 0), sy);
    z = 0;
  }
  Eigen::Matrix<T, 3, 1> ang(x, y, z);
  return ang;
}
}

// 常见的几何估计
namespace GeometryMath {

/**
 * @brief 计算最近邻点的直线主方向
 * @param points    输入的最近邻点
 * @param thresh    最近邻点到直线的距离阈值
 * @param out       PCA计算的直线主方向
 * @return
 */
bool estimate_line(std::vector<Eigen::Vector3d> &points,
                   const double &thresh,
                   Eigen::Vector3d &mean,
                   Eigen::Vector3d &out) {
  //  激光个数进行判断
  if (points.size() <= 2) {
    return false;
  }

  //  计算点云簇的均值
  for (const auto &point : points) {
    mean = mean + point;
  }
  mean = mean / points.size();

  //  计算点云去中心的协方差
  Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
  for (auto &point : points) {
    Eigen::Vector3d diff = point - mean;
    cov += diff * diff.transpose();
  }

  // 进行特征值分解
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigen_solver(cov);
  Eigen::Vector3d eigenvalues = eigen_solver.eigenvalues();
  Eigen::Matrix3d eigenvectors = eigen_solver.eigenvectors();

  // 计算主方向
  int principal_index;
  eigenvalues.maxCoeff(&principal_index);
  out = eigenvectors.col(principal_index);

  // 对拟合直线的点距离进行计算，判断是否满足阈值
  for (const auto &point : points) {
    double dist = point.dot(out);
    if (dist > thresh) {
      return false;
    }
  }

  return true;
}

/**
 * @brief 计算最近邻点的平面主方向
 * @param points    输入的最近邻点
 * @param thresh    check 最近邻的所有点到平面的距离是否满足阈值要求
 * @param out       输出的归一化后的平面方程
 * @return
 */
// fastlio 中进行平面参数估计的方式，前提是场景有大面积平面特征，都是角点，估计不准平面（雕塑场景）
bool estimate_plane(std::vector<Eigen::Vector3d> &points,
                    const double &thresh,
                    Eigen::Vector3d &mean,
                    Eigen::Vector4d &out) {
  //  激光个数进行判断
  if (points.size() <= 2) {
    return false;
  }

  //  计算点云簇的均值
  for (const auto &point : points) {
    mean = mean + point;
  }
  mean = mean / points.size();

  /// 求解平面的法向量
  // Ax + By + Cz + D = 0
  // A/D x + B/D y + C/D z = -1

  Eigen::MatrixXd A(points.size(), 3);
  A.setZero();
  for (int i = 0; i < points.size(); i++) {
    A.row(i) = points[i].transpose();
  }

  Eigen::MatrixXd b(points.size(), 1);
  b.setOnes();
  b *= -1.0;

  Eigen::Vector3d normvec = A.colPivHouseholderQr().solve(b);


  // 确定最终的平面方程为: 法向量的归一化形式
  // Ax + By + Cz + D = 0
  // A / sqrt(A2 + B2 + C2) x + B / sqrt(A2 + B2 + C2) y + C / sqrt(A2 + B2 + C2) z + D / sqrt(A2 + B2 + C2) = 0
  double norm = normvec.norm();
  out[0] = normvec(0) / norm;
  out[1] = normvec(1) / norm;
  out[2] = normvec(2) / norm;
  out[3] = 1.0 / norm;
  for (auto &point : points) {
    if (std::fabs(out(0) * point.x() + out(1) * point.y() + out(2) * point.z() + out(3)) > thresh) {
      return false;
    }
  }
  return true;
}

/**
 * Todo: SVD进行平面特征求解
 */

}

// 点云配准
namespace Registration {

// 配准方法的种类
enum RegistrationMode {
  SVD_ALIGNED = 0,
  Point2Point,
  Point2Line,
  Point2Plane,
  NDT_ALIGNED,
  NICP,
  IMLS_ICP,
  GICP,
};

std::string getRegistrationMode(const RegistrationMode &registrationMode) {
  std::string mode_name;
  switch (registrationMode) {
    case RegistrationMode::SVD_ALIGNED:mode_name = std::string("SVD_ALIGNED");
      break;
    case RegistrationMode::Point2Point:mode_name = std::string("Point2Point");
      break;
    case RegistrationMode::Point2Line:mode_name = std::string("Point2Line");
      break;
    case RegistrationMode::Point2Plane:mode_name = std::string("Point2Plane");
      break;
    case RegistrationMode::NDT_ALIGNED:mode_name = std::string("NDT_ALIGNED");
      break;
    case RegistrationMode::NICP:mode_name = std::string("NICP");
      break;
    case RegistrationMode::IMLS_ICP:mode_name = std::string("IMLS_ICP");
      break;
    default:mode_name = std::string("Unknown Registration Mode");
      break;
  }
  return mode_name;
}

class RegistrationBase {
 public:
  RegistrationBase() {}

  virtual ~RegistrationBase() = default; // 虚析构函数，确保派生类对象能正确析构

  // 参数配置
  virtual void setIterations(int iterations) { std::cout << "去子类进行实现..." << std::endl; }       // 优化的迭代次数
  virtual void setEpsilon(double epsilon) { std::cout << "去子类进行实现..." << std::endl; }          // 迭代优化终止的epsilon
  virtual void setNearestDist(double nearest_dist) { std::cout << "去子类进行实现..." << std::endl; } // 最近邻查询的距离
  virtual void setTBBFlag(bool use_tbb_flag) { std::cout << "去子类进行实现..." << std::endl; }       // 是否使用TBB加速 todo
  virtual void setLogFlag(bool use_log_flag) { std::cout << "去子类进行实现..." << std::endl; }       // 是否参数打印
  virtual void logParameter() { std::cout << "去子类进行实现..." << std::endl; }

  // 配置输入的参数
  virtual void setSourceCloud(const pcl::PointCloud<pcl::PointXYZI>::Ptr &source_cloud_ptr) {
    std::cout << "去子类进行实现..." << std::endl;
  }
  virtual void setTargetCloud(const pcl::PointCloud<pcl::PointXYZI>::Ptr &target_cloud_ptr) {
    std::cout << "去子类进行实现..." << std::endl;
  }
  virtual void setInitT(const Eigen::Matrix4d &init_T) { std::cout << "去子类进行实现..." << std::endl; }

  // 进行优化处理
  virtual bool Handle() = 0;

  // 获取结果
  virtual void getInitTransform(Eigen::Matrix4d &init_T) { std::cout << "去子类进行实现..." << std::endl; }
  virtual void getRegistrationTransform(Eigen::Matrix4d &option_transform) {
    std::cout << "去子类进行实现..." << std::endl;
  }
  virtual void getTransformedOriginCloud(const pcl::PointCloud<pcl::PointXYZI>::Ptr &transformed_cloud) {
    std::cout << "去子类进行实现..." << std::endl;
  };

  RegistrationMode registration_mode_; // 配准模式

  int iterations_{};      // 迭代次数
  double epsilon_{};      // 位移量的epsilon 位姿变换足够小停止迭代
  double nearest_dist_{}; // 最近邻查询距离
  bool use_tbb_flag_{};   // 是否使用TBB加速
  bool use_log_flag_{};   // 是否需要log

  Eigen::Matrix4d init_T_; // 初始外参
  pcl::PointCloud<pcl::PointXYZI>::Ptr target_cloud_ptr_ = nullptr;
  pcl::PointCloud<pcl::PointXYZI>::Ptr source_cloud_ptr_ = nullptr;

  bool convergence_flag_{}; // 是否收敛
  Eigen::Matrix4d final_T_; // 最终的外参
};
}

#endif //EIGEN_ICP_MATCH_BASE_H
