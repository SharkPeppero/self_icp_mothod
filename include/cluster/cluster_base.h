//
// Created by xu on 25-1-29.
//

#ifndef EIGEN_ICP_INCLUDE_CLUSTER_CLUSTER_H_
#define EIGEN_ICP_INCLUDE_CLUSTER_CLUSTER_H_

#include "string"
#include "vector"

#include "pcl/point_types.h"
#include "pcl/point_cloud.h"

typedef pcl::PointXYZI PointType;
typedef pcl::PointCloud<PointType> CloudType;

namespace clustering {

/// 配准方法的种类
enum ClusterMode {
  DBSCAN_Mode = 0,
};

/// 获取聚类方法
std::string getClusterMode(const ClusterMode &cluster_mode) {
  std::string mode_name;
  switch (cluster_mode) {
    case ClusterMode::DBSCAN_Mode:mode_name = std::string("DBSCAN Mode");
      break;
    default:mode_name = std::string("Unknown Registration Mode");
      break;
  }
  return mode_name;
}

class ClusterBase {
 public:
  ClusterBase() = default;
  virtual ~ClusterBase() = default;
  virtual void setInput(const CloudType::Ptr &input) {
    std::cout << "去子类进行实现..." << std::endl;
  }
  virtual std::vector<CloudType::Ptr> getClusterPC() {
    std::cout << "去子类进行实现..." << std::endl;
    return {};
  };
  virtual void Handle() {
    std::cout << "去子类进行实现..." << std::endl;
  }

  ClusterMode cluster_mode_;
  CloudType::Ptr input_ptr_ = nullptr;
  std::map<int, CloudType::Ptr> cluster_res_;
};
}

#endif //EIGEN_ICP_INCLUDE_CLUSTER_CLUSTER_H_
