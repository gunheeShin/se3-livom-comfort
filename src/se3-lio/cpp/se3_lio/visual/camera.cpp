#include "visual/camera.h"

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

namespace se3_lio {

PinholeCamera::PinholeCamera(int width, int height, double fx, double fy, double cx, double cy,
                             const std::vector<double> &dist_coeffs, bool fisheye)
    : width_(width), height_(height), fx_(fx), fy_(fy), cx_(cx), cy_(cy) {
    for (double d : dist_coeffs) {
        if (d != 0.0) {
            has_dist_ = true;
            break;
        }
    }
    if (has_dist_) {
        cv::Mat K = (cv::Mat_<double>(3, 3) << fx_, 0, cx_, 0, fy_, cy_, 0, 0, 1);
        cv::Mat D(dist_coeffs);
        if (fisheye)
            cv::fisheye::initUndistortRectifyMap(K, D, cv::Mat(), K, cv::Size(width_, height_),
                                                 CV_32FC1, undist_map1_, undist_map2_);
        else
            cv::initUndistortRectifyMap(K, D, cv::Mat(), K, cv::Size(width_, height_), CV_32FC1,
                                        undist_map1_, undist_map2_);
    }
}

cv::Mat PinholeCamera::undistort(const cv::Mat &img) const {
    if (!has_dist_) {
        return img;
    }
    cv::Mat out;
    cv::remap(img, out, undist_map1_, undist_map2_, cv::INTER_LINEAR);
    return out;
}

Eigen::Matrix<double, 2, 3> PinholeCamera::projectJacobian(const Eigen::Vector3d &p_c) const {
    const double z_inv = 1.0 / p_c.z();
    const double z_inv2 = z_inv * z_inv;
    Eigen::Matrix<double, 2, 3> J;
    J << fx_ * z_inv, 0.0, -fx_ * p_c.x() * z_inv2,  //
        0.0, fy_ * z_inv, -fy_ * p_c.y() * z_inv2;
    return J;
}

}  // namespace se3_lio
