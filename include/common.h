//
// Created by xu on 24-10-12.
//

#ifndef EIGEN_ICP_COMMON_H
#define EIGEN_ICP_COMMON_H

#include "chrono"
#include "pcl/point_types.h"
#include "pcl/point_cloud.h"

template<typename PointT>
Eigen::Vector3d point2vec(const PointT& point){
    return Eigen::Vector3d(point.x, point.y, point.z);
}

class TicToc {
public:
    TicToc() { tic(); }

    ~TicToc() = default;

    void tic() { start_time_ = std::chrono::steady_clock::now(); }

    /// util: ms
    double toc() {
        end_time_ = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::milli>(end_time_ - start_time_).count();
    }

private:
    std::chrono::steady_clock::time_point start_time_, end_time_;
};

#endif //EIGEN_ICP_COMMON_H
