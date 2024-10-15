//
// Created by xu on 24-10-12.
//

#ifndef EIGEN_ICP_MATH_H
#define EIGEN_ICP_MATH_H

#include "Eigen/Core"
#include "Eigen/Dense"
#include "Eigen/Geometry"

#define SKEW_SYM_MATRX(v) 0.0,-v[2],v[1],v[2],0.0,-v[0],-v[1],v[0],0.0

/// 反对称矩阵
inline Eigen::Matrix3d Skew(const Eigen::Vector3d &v) {
    Eigen::Matrix3d res;
    res << 0, -v(2), v(1),
            v(2), 0, -v(0),
            -v(1), v(0), 0;
    return res;
}

/// 计算质心以及对角协方差


/// 计算质心以及完整的协方差

/// 流行空间计算
namespace ManifoldMath {
    template<typename T>
    Eigen::Matrix<T, 3, 3> skew_sym_mat(const Eigen::Matrix<T, 3, 1> &v) {
        Eigen::Matrix<T, 3, 3> skew_sym_mat;
        skew_sym_mat << 0.0, -v[2], v[1], v[2], 0.0, -v[0], -v[1], v[0], 0.0;
        return skew_sym_mat;
    }

    template<typename T>
    Eigen::Matrix<T, 3, 3> Exp(const Eigen::Matrix<T, 3, 1> &ang) {
        T ang_norm = ang.norm();
        Eigen::Matrix<T, 3, 3> Eye3 = Eigen::Matrix<T, 3, 3>::Identity();
        if (ang_norm > 0.0000001) {
            Eigen::Matrix<T, 3, 1> r_axis = ang / ang_norm;
            Eigen::Matrix<T, 3, 3> K;
            K << SKEW_SYM_MATRX(r_axis);
            /// Roderigous Tranformation
            return Eye3 + std::sin(ang_norm) * K + (1.0 - std::cos(ang_norm)) * K * K;
        } else {
            return Eye3;
        }
    }

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


namespace GeometryMath {

    /**
     * @brief 计算最近邻点的直线主方向
     * @param points    输入的最近邻点
     * @param thresh    最近邻点到直线的距离阈值
     * @param out       PCA计算的直线主方向
     * @return
     */
    bool estimate_line(std::vector<Eigen::Vector3d> &points, const double &thresh, Eigen::Vector3d &out) {
        //  激光个数进行判断
        if(points.size() <= 2){
            return false;
        }

        //  计算点云簇的均值
        Eigen::Vector3d mean = Eigen::Vector3d::Zero();
        for (int index = 0; index < points.size(); index++) {
            mean = mean + points[index];
        }
        mean = mean / points.size();

        //  计算点云去中心的协方差
        Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
        for (int index = 0; index < points.size(); index++) {
            Eigen::Vector3d diff = points[index] - mean;
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
        for (const auto &point: points) {
            double dist = point.dot(out);
            if(dist > thresh){
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
    bool estimate_plane(std::vector<Eigen::Vector3d> &points, const double &thresh, Eigen::Vector4d &out) {

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
        for (auto & point : points) {
            if (std::fabs(out(0) * point.x() + out(1) * point.y() + out(2) * point.z() + out(3)) > thresh) {
                return false;
            }
        }
        return true;
    }
}




#endif //EIGEN_ICP_MATH_H


