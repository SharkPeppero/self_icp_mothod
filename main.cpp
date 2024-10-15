//
// Created by xu on 24-9-9.
//

/**
 * @brief 利用Eigen进行实现 点云的点点 点线 点面配准
 */

#include "thread"

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/registration/icp.h>
#include <pcl/filters/voxel_grid.h>

#include "icp_mothod.h"


void displayCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& target_cloud_ptr,
                  const pcl::PointCloud<pcl::PointXYZ>::Ptr& transformed_source_cloud_ptr) {
    // 创建可视化窗口
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Point Cloud Viewer"));
    viewer->setBackgroundColor(0, 0, 0);

    // 将 target_cloud_ptr 设置成红色
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> target_color(target_cloud_ptr, 255, 0, 0);
    viewer->addPointCloud<pcl::PointXYZ>(target_cloud_ptr, target_color, "target cloud");

    // 将 transformed_source_cloud_ptr 设置成绿色
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> source_color(transformed_source_cloud_ptr, 0, 255, 0);
    viewer->addPointCloud<pcl::PointXYZ>(transformed_source_cloud_ptr, source_color, "transformed source cloud");

    // 设置点云大小
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "target cloud");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "transformed source cloud");

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

    // 读取指定路径的点云数据 target source
    pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud_ptr(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud_ptr(new pcl::PointCloud<pcl::PointXYZ>());

    if (pcl::io::loadPCDFile<pcl::PointXYZ>("/home/xu/ZME/icp_eigen/PCDdata/target.pcd", *target_cloud_ptr) == -1 ||
        pcl::io::loadPCDFile<pcl::PointXYZ>("/home/xu/ZME/icp_eigen/PCDdata/source.pcd", *source_cloud_ptr) == -1) {
        PCL_ERROR("Couldn't read the point cloud files \n");
        return -1;
    }
    printf("source cloud size: %ld\n", source_cloud_ptr->points.size());
    printf("target cloud size: %ld\n", target_cloud_ptr->points.size());

/*    {
        //////////////////////////////  进行SVD的ICP   //////////////////////////////
        // iter: 44
        // cost time 8737.24 ms.
//        0.999999        0.000984988   0.00100448      1.30576
//        -0.00098566     0.999999  0.000669145    0.0245338
//        -0.00100382     -0.000670134     0.999999   0.00481544
//        0            0            0            1

        ICP_SVD<pcl::PointXYZ> icpSvd;
        icpSvd.setLogFlag(true);
        icpSvd.setMaxIterations(100);
        icpSvd.setEpsilon(1e-3);
        icpSvd.setNearestDist(10.0);
        /// !!!!!初值给的差，可能收敛不回来（不知道位置的情况下如何给出好初值）!!!!!
        /// 初值给的好提升速度
        Eigen::Matrix4d init_T_target_source = Eigen::Matrix4d::Identity();
        icpSvd.setInitPose(init_T_target_source);
        icpSvd.setSourceCloud(source_cloud_ptr);
        icpSvd.setTargetCloud(target_cloud_ptr);

        TicToc svd_timer;
        icpSvd.Aligned();
        std::cout << "svd result: " << icpSvd.getAlignedExternal().matrix() << std::endl;
        std::cout << "svd time: " << svd_timer.toc() << " ms." << std::endl;
        displayCloud(target_cloud_ptr, icpSvd.getAlignedCloud());

    }*/

/*    {
        //////////////////////////////  进行P2Point的ICP  //////////////////////////////
        // 44
        // 120568 ms.
//        0.999978 -0.00621523 -0.00209931     1.33506
//        0.00621148    0.999979 -0.00178819   0.0134711
//        0.00211038  0.00177511    0.999996   0.0102646
//        0           0           0           1
        ICP_P2Point<pcl::PointXYZ> icpP2Point;
        icpP2Point.setLogFlag(true);
        icpP2Point.setMaxIterations(100);
        icpP2Point.setEpsilon(1e-3);
        icpP2Point.setNearestDist(10.0);
        /// !!!!!初值给的差，可能收敛不回来（不知道位置的情况下如何给出好初值）!!!!!
        /// 初值给的好提升速度
        Eigen::Matrix4d init_T_target_source = Eigen::Matrix4d::Identity();
        icpP2Point.setInitPose(init_T_target_source);
        icpP2Point.setSourceCloud(source_cloud_ptr);
        icpP2Point.setTargetCloud(target_cloud_ptr);

        TicToc svd_timer;
        icpP2Point.Aligned();
        std::cout << "svd result: " << icpP2Point.getAlignedExternal().matrix() << std::endl;
        std::cout << "svd time: " << svd_timer.toc() << " ms." << std::endl;
        displayCloud(target_cloud_ptr, icpP2Point.getAlignedCloud());
    }*/

    {
        ICP_P2Line<pcl::PointXYZ> icpP2Line;
        icpP2Line.setLogFlag(true);
        icpP2Line.setMaxIterations(100);
        icpP2Line.setNearestDist(10.0);
        /// !!!!!初值给的差，可能收敛不回来（不知道位置的情况下如何给出好初值）!!!!!
        /// 初值给的好提升速度
        Eigen::Matrix4d init_T_target_source = Eigen::Matrix4d::Identity();
        icpP2Line.setInitPose(init_T_target_source);
        icpP2Line.setSourceCloud(source_cloud_ptr);
        icpP2Line.setTargetCloud(target_cloud_ptr);

        TicToc svd_timer;
        icpP2Line.Aligned();
        std::cout << "svd result: " << icpP2Line.getAlignedExternal().matrix() << std::endl;
        std::cout << "svd time: " << svd_timer.toc() << " ms." << std::endl;
        displayCloud(target_cloud_ptr, icpP2Line.getAlignedCloud());
    }

/*    {
        ICP_P2Plane<pcl::PointXYZ> icpP2Plane;
        icpP2Plane.setLogFlag(true);
        icpP2Plane.setMaxIterations(100);
        icpP2Plane.setNearestDist(10.0);
        /// !!!!!初值给的差，可能收敛不回来（不知道位置的情况下如何给出好初值）!!!!!
        /// 初值给的好提升速度
        Eigen::Matrix4d init_T_target_source = Eigen::Matrix4d::Identity();
        init_T_target_source.block<3, 1>(0, 3) = Eigen::Vector3d(0.0, 0.0, 0.0);
        icpP2Plane.setInitPose(init_T_target_source);
        icpP2Plane.setSourceCloud(source_cloud_ptr);
        icpP2Plane.setTargetCloud(target_cloud_ptr);

        TicToc svd_timer;
        icpP2Plane.Aligned();
        std::cout << "svd result: " << icpP2Plane.getAlignedExternal().matrix() << std::endl;
        std::cout << "svd time: " << svd_timer.toc() << " ms." << std::endl;
        displayCloud(target_cloud_ptr, icpP2Plane.getAlignedCloud());
    }*/

    return 0;

}