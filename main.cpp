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

  } else if (mode_index == "6") {
    // ============================ NICP 配准 ==========================
  } else if (mode_index == "7") {
    // ============================  配准 ==========================
  } else {

  }

  return EXIT_SUCCESS;
}