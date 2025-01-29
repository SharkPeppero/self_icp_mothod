/**
 * @brief 利用Eigen进行实现
 * SVD配准
 * P2point P2Line P2Plane 的ICP
 * GICP
 * NICP
 * IMSLICP
 */

#include "thread"

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/voxel_grid.h>

#include "icp_mothed/registration_base.h"
#include "icp_mothed/svd_aligned.h"
#include "icp_mothed/point2point.h"
#include "icp_mothed/point2line.h"
#include "icp_mothed/point2plane.h"
#include "icp_mothed/ndt_aligned.h"
#include "icp_mothed/nicp_registration.h"

#include "cluster/dbscan.h"

void displayCloud(const pcl::PointCloud<pcl::PointXYZI>::Ptr &target_cloud_ptr,
                  const pcl::PointCloud<pcl::PointXYZI>::Ptr &transformed_source_cloud_ptr) {
  // 创建可视化窗口
  pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Point Cloud Viewer"));
  viewer->setBackgroundColor(0, 0, 0);

  // 将 target_cloud_ptr 设置成红色
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> target_color(target_cloud_ptr, 255, 0, 0);
  viewer->addPointCloud<pcl::PointXYZI>(target_cloud_ptr, target_color, "target cloud");

  // 将 transformed_source_cloud_ptr 设置成绿色
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> source_color(transformed_source_cloud_ptr, 0, 255,
                                                                                0);
  viewer->addPointCloud<pcl::PointXYZI>(transformed_source_cloud_ptr, source_color, "transformed source cloud");

  // 设置点云大小
  viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "target cloud");
  viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1,
                                           "transformed source cloud");

  // 显示坐标轴
  viewer->addCoordinateSystem(1.0);
  viewer->initCameraParameters();

  // 显示循环
  while (!viewer->wasStopped()) {
    viewer->spinOnce(100);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

}

int main(int argc, char *argv[]) {

  if (argc != 4) {
    std::cerr << "请输入source点云以及target点云的路径参考 以及 模式." << std::endl;
    return EXIT_FAILURE;
  }

  std::string target_cloud_path = argv[1];
  std::string source_cloud_path = argv[2];
  std::string mode_index = argv[3];
  std::cout << "target_cloud_path: " << target_cloud_path << std::endl;
  std::cout << "source_cloud_path: " << source_cloud_path << std::endl;
  std::cout << "mode index: " << mode_index << std::endl;

  // 读取指定路径的点云数据 target source
  pcl::PointCloud<pcl::PointXYZI>::Ptr target_cloud_ptr(new pcl::PointCloud<pcl::PointXYZI>());
  pcl::PointCloud<pcl::PointXYZI>::Ptr source_cloud_ptr(new pcl::PointCloud<pcl::PointXYZI>());

  if (pcl::io::loadPCDFile<pcl::PointXYZI>(target_cloud_path, *target_cloud_ptr) == -1 ||
      pcl::io::loadPCDFile<pcl::PointXYZI>(source_cloud_path, *source_cloud_ptr) == -1) {
    PCL_ERROR("Couldn't read the point cloud files \n");
    return -1;
  }
  printf("source cloud size: %ld\n", source_cloud_ptr->points.size());
  printf("target cloud size: %ld\n", target_cloud_ptr->points.size());

  std::string debug_path = DEBUG_PATH;
  printf("debug reference path: %s\n", debug_path.c_str());

  if (mode_index == "0") {
    // ============================ SVD配准 =================================
    std::shared_ptr<Registration::RegistrationBase>
        registration_base_ptr = std::make_shared<Registration::SVDAligned>();
    registration_base_ptr->setLogFlag(true);
    registration_base_ptr->setIterations(50);
    registration_base_ptr->setEpsilon(1e-6);
    registration_base_ptr->setInitT(Eigen::Matrix4d::Identity());
    registration_base_ptr->setSourceCloud(source_cloud_ptr);
    registration_base_ptr->setTargetCloud(target_cloud_ptr);
    std::shared_ptr<Registration::SVDAligned>
        svd_registration = std::dynamic_pointer_cast<Registration::SVDAligned>(registration_base_ptr);
    svd_registration->Handle();

    pcl::PointCloud<pcl::PointXYZI>::Ptr transformed_cloud_ptr(new pcl::PointCloud<pcl::PointXYZI>());
    registration_base_ptr->getTransformedOriginCloud(transformed_cloud_ptr);
    displayCloud(target_cloud_ptr, transformed_cloud_ptr);
  } else if (mode_index == "1") {
    // ============================ point2point 配准 =======================
    std::shared_ptr<Registration::RegistrationBase>
        registration_base_ptr = std::make_shared<Registration::Point2PointRegistration>();
    registration_base_ptr->setIterations(50);
    registration_base_ptr->setEpsilon(1e-6);
    registration_base_ptr->logParameter();
    registration_base_ptr->setInitT(Eigen::Matrix4d::Identity());
    registration_base_ptr->setSourceCloud(source_cloud_ptr);
    registration_base_ptr->setTargetCloud(target_cloud_ptr);
    registration_base_ptr->Handle();
    pcl::PointCloud<pcl::PointXYZI>::Ptr transformed_cloud_ptr(new pcl::PointCloud<pcl::PointXYZI>());
    registration_base_ptr->getTransformedOriginCloud(transformed_cloud_ptr);
    displayCloud(target_cloud_ptr, transformed_cloud_ptr);

  } else if (mode_index == "2") {
    // ============================ point2line 配准 ==========================
    std::shared_ptr<Registration::RegistrationBase>
        registration_base_ptr = std::make_shared<Registration::Point2LineRegistration>();
    registration_base_ptr->setIterations(50);
    registration_base_ptr->setEpsilon(1e-6);
    registration_base_ptr->logParameter();
    registration_base_ptr->setInitT(Eigen::Matrix4d::Identity());
    registration_base_ptr->setSourceCloud(source_cloud_ptr);
    registration_base_ptr->setTargetCloud(target_cloud_ptr);
    registration_base_ptr->Handle();
    pcl::PointCloud<pcl::PointXYZI>::Ptr transformed_cloud_ptr(new pcl::PointCloud<pcl::PointXYZI>());
    registration_base_ptr->getTransformedOriginCloud(transformed_cloud_ptr);
    displayCloud(target_cloud_ptr, transformed_cloud_ptr);

  } else if (mode_index == "3") {
    // ============================ point2Plane 配准 ==========================
    std::shared_ptr<Registration::RegistrationBase>
        registration_base_ptr = std::make_shared<Registration::Point2PlaneRegistration>();
    registration_base_ptr->setIterations(50);
    registration_base_ptr->setEpsilon(1e-6);
    registration_base_ptr->logParameter();
    registration_base_ptr->setInitT(Eigen::Matrix4d::Identity());
    registration_base_ptr->setSourceCloud(source_cloud_ptr);
    registration_base_ptr->setTargetCloud(target_cloud_ptr);
    registration_base_ptr->Handle();
    pcl::PointCloud<pcl::PointXYZI>::Ptr transformed_cloud_ptr(new pcl::PointCloud<pcl::PointXYZI>());
    registration_base_ptr->getTransformedOriginCloud(transformed_cloud_ptr);
    displayCloud(target_cloud_ptr, transformed_cloud_ptr);
    std::cout << registration_base_ptr->final_T_.matrix() << std::endl;
  } else if (mode_index == "4") {
    // ============================ NDT 配准 ==========================
    std::shared_ptr<Registration::RegistrationBase>
        registration_base_ptr = std::make_shared<Registration::NDTAligned>();
    // 父类属性配置
    registration_base_ptr->setLogFlag(true);
    registration_base_ptr->setIterations(50);
    registration_base_ptr->setEpsilon(1e-6);
    registration_base_ptr->setInitT(Eigen::Matrix4d::Identity());
    registration_base_ptr->setSourceCloud(source_cloud_ptr);
    registration_base_ptr->setTargetCloud(target_cloud_ptr);
    auto ndt_registration_ptr = std::dynamic_pointer_cast<Registration::NDTAligned>(registration_base_ptr);
    // 子类属性配置
    ndt_registration_ptr->setMinPtsInVoxel(30);
    ndt_registration_ptr->setNearbyType(NearbyType::CENTER);
    ndt_registration_ptr->setVoxelSize(0.05);

    registration_base_ptr->Handle();
    pcl::PointCloud<pcl::PointXYZI>::Ptr transformed_cloud_ptr(new pcl::PointCloud<pcl::PointXYZI>());
    registration_base_ptr->getTransformedOriginCloud(transformed_cloud_ptr);
    displayCloud(target_cloud_ptr, transformed_cloud_ptr);

  } else if (mode_index == "5") {
    // ============================ NICP 配准 ==========================
    std::shared_ptr<Registration::RegistrationBase>
        registration_base_ptr = std::make_shared<Registration::NICPRegistration>();
    // 父类属性配置
    registration_base_ptr->setLogFlag(true);
    registration_base_ptr->setIterations(50);
    registration_base_ptr->setEpsilon(1e-6);
    registration_base_ptr->setInitT(Eigen::Matrix4d::Identity());
    registration_base_ptr->setSourceCloud(source_cloud_ptr);
    registration_base_ptr->setTargetCloud(target_cloud_ptr);

    registration_base_ptr->Handle();
    pcl::PointCloud<pcl::PointXYZI>::Ptr transformed_cloud_ptr(new pcl::PointCloud<pcl::PointXYZI>());
    registration_base_ptr->getTransformedOriginCloud(transformed_cloud_ptr);
    displayCloud(target_cloud_ptr, transformed_cloud_ptr);

  } else if (mode_index == "6") {
    // ============================ NICP 配准 ==========================
  } else if (mode_index == "7") {
    // ============================  配准 ==========================
  } else if (mode_index == "10") {
    // ============================ 执行DBSCAN ========================
    // 体素下采样
    pcl::VoxelGrid<PointType> voxel_grid;
    voxel_grid.setInputCloud(target_cloud_ptr);
    voxel_grid.setLeafSize(0.2, 0.2, 0.2);  // 设置体素大小
    voxel_grid.filter(*target_cloud_ptr);  // 执行下采样

    std::shared_ptr<clustering::ClusterBase> cluster_base_ptr =
        std::make_shared<clustering::DBSCAN>();
    cluster_base_ptr->setInput(target_cloud_ptr);
    std::shared_ptr<clustering::DBSCAN> dbscan_ptr =
        std::dynamic_pointer_cast<clustering::DBSCAN>(cluster_base_ptr);
    dbscan_ptr->setEpsilon(1.0);
    dbscan_ptr->setMinPts(30);

    cluster_base_ptr->Handle();

    std::cout << "cluster size: " << cluster_base_ptr->cluster_res_.size() << std::endl;
    if (!cluster_base_ptr->cluster_res_.empty()) {
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());

      // 为每个聚类分配不同的颜色
      int cluster_idx = 0;
      for (const auto &cluster_pair : cluster_base_ptr->cluster_res_) {
        CloudType::Ptr cluster = cluster_pair.second;
        for (size_t i = 0; i < cluster->points.size(); ++i) {
          pcl::PointXYZRGB point;
          point.x = cluster->points[i].x;
          point.y = cluster->points[i].y;
          point.z = cluster->points[i].z;

          // 给每个聚类分配不同的颜色
          point.r = static_cast<uint8_t>((cluster_idx * 50) % 256);  // 红色分量
          point.g = static_cast<uint8_t>((cluster_idx * 100) % 256); // 绿色分量
          point.b = static_cast<uint8_t>((cluster_idx * 150) % 256); // 蓝色分量

          colored_cloud->points.push_back(point);
        }
        cluster_idx++;
      }

      colored_cloud->height = 1;
      colored_cloud->width = colored_cloud->points.size();

      std::string debug_pcd_path = debug_path + "clustering.pcd";
      pcl::io::savePCDFile(debug_pcd_path, *colored_cloud);
    }

  } else {

  }

  return EXIT_SUCCESS;
}