# slef_icp_mothod

## icp原理特点总结

点线 点面关键点在于投影矩阵：
https://www.cnblogs.com/bigmonkey/p/9897047.html

点点以及ndt的差别在于，是否有加权信息矩阵参与梯度下降，加权最小二乘可以加速梯度下降的速度。



## 精度测试总结：

 各种配准算法依赖场景的先验特点，大概总结：
    point2point可以理解成低速的ndt 

​     p2line，场景中的直线特征可能不太存在，但是在棱角分明的场景比如全是墙面ok

​     p2plane，在规则的多面场景提供面约束比较充足的场景，ok

​     对于给定的雕塑数据集，point2point和ndt的效果比较好，没有明显line以及平面，如果曲面可以理解成平面的话，就需要给定平面估计的参数，参数给不好，也会适得其反



​    想要有一个方法，可以不挑场景，挖掘场景的基础信息

todo: 接下来尝试代码书写以及效果比较

NICP

GICP

IMSLICP



## SVD配准以及高斯牛顿实现点云配准
特点：
  1.手动实现雅阁比的实现，深入理解算法优化逻辑，
  2.相比pcl源码缺少预处理，对于角点以及平面点占多或场景小但是场景结构丰富快速匹配的场景需要了解残差构建方式，搭配合适的残差构建起到优化的 **快与准**
  3.总结了目前常用的icp方法并扩充了为开源个人手推的配准算法

author： Jay.xu
Westwell Lab
