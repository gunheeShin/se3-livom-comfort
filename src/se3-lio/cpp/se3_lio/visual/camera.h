#ifndef VISUAL_CAMERA_H
#define VISUAL_CAMERA_H

#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include <vector>

namespace se3_lio {

// Pinhole camera. Distorted inputs (radtan, or equidistant when fisheye) are
// rectified once per frame with undistort() onto the same K;
// project()/projectJacobian() assume an ideal pinhole model.
class PinholeCamera {
public:
    PinholeCamera(int width, int height, double fx, double fy, double cx, double cy,
                  const std::vector<double> &dist_coeffs = {}, bool fisheye = false);

    cv::Mat undistort(const cv::Mat &img) const;

    inline Eigen::Vector2d project(const Eigen::Vector3d &p_c) const {
        return {fx_ * p_c.x() / p_c.z() + cx_, fy_ * p_c.y() / p_c.z() + cy_};
    }

    inline Eigen::Vector3d unproject(const Eigen::Vector2d &px) const {
        return {(px.x() - cx_) / fx_, (px.y() - cy_) / fy_, 1.0};
    }

    Eigen::Matrix<double, 2, 3> projectJacobian(const Eigen::Vector3d &p_c) const;

    inline bool isInFrame(const Eigen::Vector2d &px, int border) const {
        return px.x() >= border && px.x() < width_ - border && px.y() >= border &&
               px.y() < height_ - border;
    }

    int width() const { return width_; }
    int height() const { return height_; }

private:
    int width_, height_;
    double fx_, fy_, cx_, cy_;
    bool has_dist_ = false;
    cv::Mat undist_map1_, undist_map2_;
};

}  // namespace se3_lio

#endif  // VISUAL_CAMERA_H
