//
// Created by xu on 25-1-29.
//

#ifndef EIGEN_ICP_INCLUDE_CLUSTER_DBSCAN_H_
#define EIGEN_ICP_INCLUDE_CLUSTER_DBSCAN_H_

/**
 * @brief DBSCAN本质是深度优先搜索
 *  radius以及min_pts控制密度
 *  深度优先搜索实现点云簇的统计
 */
#include "cluster_base.h"

namespace clustering {
class DBSCAN : public ClusterBase {
 public:
  DBSCAN() {
    cluster_mode_ = DBSCAN_Mode;

    epsilon_ = 5.0;
    min_pts_ = 10;
  }

  ~DBSCAN() override = default;

  void setInput(const CloudType::Ptr &input) override {
    input_ptr_ = input;
  }

  void setEpsilon(double epsilon) { epsilon_ = epsilon; }
  void setMinPts(int min_pts) { min_pts_ = min_pts; }

  void Handle() override {
    /// 断言
    assert(input_ptr_ != nullptr);
    assert(!input_ptr_->points.empty());

    /// 构建点云的 KD-Tree
    pcl::KdTreeFLANN<PointType> kdtree;
    kdtree.setInputCloud(input_ptr_);

    /// 初始化
    std::vector<bool> visited(input_ptr_->size(), false);  // 标记点是否被访问过
    std::vector<int> labels(input_ptr_->size(), -1);       // 存储每个点的簇标签（-1 表示噪声点）
    int cluster_id = 0;                                    // 簇的 ID

    /// 遍历点云
    for (size_t point_index = 0; point_index < input_ptr_->size(); ++point_index) {

      // 跳过已访问的点
      if (visited[point_index]) continue;

      // 标记当前点为已访问
      visited[point_index] = true;

      /// 邻域查询
      std::vector<int> neighbor_indices;
      std::vector<float> neighbor_distances;
      if (kdtree.radiusSearch(input_ptr_->points[point_index], epsilon_, neighbor_indices, neighbor_distances) >= min_pts_) {
        /// 给当前激光点赋予ID信息
        labels[point_index] = cluster_id;

        /// 扩充当前激光点的点云簇
        expandCluster(kdtree,           // 目标点云的KDtree
                      neighbor_indices, // 圆域搜索结果
                      labels,           // 激光点的聚类ID标签
                      visited,          // 是否访问标签
                      cluster_id        // 当前深度优先搜素的点云簇id
        );

        // 增加簇 ID
        cluster_id++;

      } else {
        labels[point_index] = -1;
      }
    }

    /// 组织成聚类的结果
    for (size_t i = 0; i < labels.size(); i++) {
      int label_key = labels[i];
      PointType point_value = input_ptr_->points[i];
      auto iter = cluster_res_.find(label_key);
      if (iter == cluster_res_.end()) {
        CloudType::Ptr clustering_cloud(new CloudType());
        clustering_cloud->push_back(point_value);
        cluster_res_[label_key] = clustering_cloud;
      } else {
        cluster_res_[label_key]->push_back(point_value);
      }
    }
  }

  /// 扩展簇
  /**
   * @brief 输入的点云最好进行降采样，不然这里深度优先搜索会造成 栈内存溢出！！！！
   * @param kdtree
   * @param neighbor_indices
   * @param labels
   * @param visited
   * @param cluster_id
   */
  void expandCluster(const pcl::KdTreeFLANN<PointType> &kdtree,
                     const std::vector<int> &neighbor_indices,
                     std::vector<int> &labels,
                     std::vector<bool> &visited,
                     int cluster_id) {
    for (size_t i = 0; i < neighbor_indices.size(); ++i) {
      int neighbor_index = neighbor_indices[i];

      if (!visited[neighbor_index]) {
        visited[neighbor_index] = true;
        std::vector<int> new_neighbor_indices;
        std::vector<float> new_neighbor_distances;
        if (kdtree.radiusSearch(input_ptr_->points[neighbor_index], epsilon_, new_neighbor_indices, new_neighbor_distances) >= min_pts_) {
          expandCluster(kdtree, new_neighbor_indices, labels, visited, cluster_id);
        }
      }

      if (labels[neighbor_index] == -1) {
        labels[neighbor_index] = cluster_id;
      }
    }
  }

  // kdtree圆域搜索的范围以及最小的搜索数目
  double epsilon_{};
  int min_pts_{};
};
}

#endif //EIGEN_ICP_INCLUDE_CLUSTER_DBSCAN_H_
