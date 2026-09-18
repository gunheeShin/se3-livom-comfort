// Standalone checks for the visual module:
//  1. photometric Jacobian vs finite differences
//  2. plane-induced affine warp vs directly rendered patch
//  3. full iterated update recovers a perturbed pose on a synthetic scene
#include <cmath>
#include <iostream>
#include <memory>

#include "core/lie.h"
#include "visual/visual_update.h"

using namespace se3_lio;

namespace {

// Smooth multi-frequency texture on the z = plane_z world plane.
double tex(double x, double y) {
    return 128.0 + 50.0 * std::sin(1.3 * x) * std::cos(0.9 * y) + 30.0 * std::sin(2.1 * y) +
           20.0 * std::cos(1.7 * x);
}

cv::Mat renderPlaneImage(const PinholeCamera &cam, const Sophus::SE3d &T_c_w, double plane_z) {
    const Sophus::SE3d T_w_c = T_c_w.inverse();
    const Eigen::Matrix3d R_w_c = T_w_c.rotationMatrix();
    const Eigen::Vector3d o = T_w_c.translation();
    cv::Mat img(cam.height(), cam.width(), CV_32FC1);
    for (int v = 0; v < cam.height(); v++) {
        for (int u = 0; u < cam.width(); u++) {
            const Eigen::Vector3d dir = R_w_c * cam.unproject({(double)u, (double)v});
            const double t = (plane_z - o.z()) / dir.z();
            const Eigen::Vector3d p = o + t * dir;
            img.at<float>(v, u) = (float)tex(p.x(), p.y());
        }
    }
    return img;
}

Sophus::SE3d makeTci() {
    const Eigen::Vector3d axis(0.0, 1.0, 0.0);
    return Sophus::SE3d(Sophus::SO3d::exp(axis * (5.0 * M_PI / 180.0)),
                        Eigen::Vector3d(0.1, -0.05, 0.02));
}

State makeState(const Sophus::SE3d &T_w_i) {
    State s;
    s.pose = T_w_i.matrix();
    s.grav = Eigen::Vector3d(0.0, 0.0, -9.81);
    s.covariance = CovMatrix::Identity() * 1e-4;
    s.covariance.block<6, 6>(0, 0) = Eigen::Matrix<double, 6, 6>::Identity() * 1e-2;
    return s;
}

bool testJacobian() {
    PinholeCamera cam(640, 480, 250.0, 250.0, 320.0, 240.0);

    // Globally linear image: bilinear sampling is exact, so only the pose
    // chain is under test.
    cv::Mat img(480, 640, CV_32FC1);
    for (int v = 0; v < 480; v++) {
        for (int u = 0; u < 640; u++) {
            img.at<float>(v, u) = 100.0f + 0.2f * u + 0.15f * v;
        }
    }

    const Sophus::SE3d T_c_i = makeTci();
    Eigen::Matrix<double, 6, 1> xi;
    xi << 0.05, -0.03, 0.02, 0.01, -0.02, 0.015;
    const Sophus::SE3d T_w_i = Sophus::SE3d::exp(xi);

    const Eigen::Vector3d p_c0(0.2, 0.1, 4.0);
    const Eigen::Vector3d p_w = (T_c_i * T_w_i.inverse()).inverse() * p_c0;

    auto residual = [&](const Eigen::Matrix<double, 6, 1> &delta) {
        const Sophus::SE3d T = T_w_i * Sophus::SE3d::exp(delta);
        return sampleBilinear(img, cam.project(T_c_i * (T.inverse() * p_w)));
    };

    // Analytic row.
    const Eigen::Vector3d p_i = T_w_i.inverse() * p_w;
    const Eigen::Vector3d p_c = T_c_i * p_i;
    double val;
    Eigen::Vector2d grad;
    if (!sampleWithGradient(img, cam.project(p_c), 1, val, grad)) {
        std::cout << "  sample out of bounds\n";
        return false;
    }
    const Eigen::Matrix<double, 1, 6> J_an =
        grad.transpose() * cam.projectJacobian(p_c) * pointJacobian(T_c_i.rotationMatrix(), p_i);

    // Finite differences.
    const double eps = 1e-5;
    double max_rel_err = 0.0;
    for (int k = 0; k < 6; k++) {
        Eigen::Matrix<double, 6, 1> delta = Eigen::Matrix<double, 6, 1>::Zero();
        delta(k) = eps;
        const double J_fd = (residual(delta) - residual(-delta)) / (2.0 * eps);
        const double rel = std::fabs(J_fd - J_an(k)) / std::max(1.0, std::fabs(J_an(k)));
        max_rel_err = std::max(max_rel_err, rel);
        std::cout << "  dim " << k << ": analytic " << J_an(k) << ", fd " << J_fd << ", rel "
                  << rel << "\n";
    }
    return max_rel_err < 1e-3;
}

bool testWarp() {
    PinholeCamera cam(640, 480, 250.0, 250.0, 320.0, 240.0);
    const double plane_z = 5.0;
    const Sophus::SE3d T_c_i = makeTci();

    const Sophus::SE3d T_c_w_ref = T_c_i;  // T_w_i = identity
    Eigen::Matrix<double, 6, 1> xi;
    xi << 0.15, -0.10, 0.05, 0.03, -0.02, 0.04;
    const Sophus::SE3d T_c_w_cur = T_c_i * Sophus::SE3d::exp(xi).inverse();

    const cv::Mat img_ref = renderPlaneImage(cam, T_c_w_ref, plane_z);
    const cv::Mat img_cur = renderPlaneImage(cam, T_c_w_cur, plane_z);

    // Plane point seen at the reference image center.
    const Eigen::Vector2d ref_px(320.0, 240.0);
    const Sophus::SE3d T_w_ref = T_c_w_ref.inverse();
    const Eigen::Vector3d dir_w = T_w_ref.rotationMatrix() * cam.unproject(ref_px);
    const double t = (plane_z - T_w_ref.translation().z()) / dir_w.z();
    const Eigen::Vector3d p_w = T_w_ref.translation() + t * dir_w;

    const Eigen::Vector3d p_ref = T_c_w_ref * p_w;
    const Eigen::Vector3d n_ref = T_c_w_ref.so3() * Eigen::Vector3d(0.0, 0.0, -1.0);

    const Sophus::SE3d T_cur_ref = T_c_w_cur * T_c_w_ref.inverse();
    const Eigen::Matrix2d A_cur_ref = computeWarpAffine(cam, T_cur_ref, ref_px, p_ref, n_ref);
    const Eigen::Matrix2d A_ref_cur = A_cur_ref.inverse();
    const Eigen::Vector2d cur_px = cam.project(T_c_w_cur * p_w);

    constexpr int kPatch = 8, kHalf = 4;
    double mean_abs_err = 0.0;
    for (int y = 0; y < kPatch; y++) {
        for (int x = 0; x < kPatch; x++) {
            const Eigen::Vector2d offset(x - kHalf, y - kHalf);
            const double warped = sampleBilinear(img_ref, ref_px + A_ref_cur * offset);
            const double direct = sampleBilinear(img_cur, cur_px + offset);
            mean_abs_err += std::fabs(warped - direct);
        }
    }
    mean_abs_err /= kPatch * kPatch;
    std::cout << "  warp mean abs err: " << mean_abs_err << " (intensity)\n";
    return mean_abs_err < 2.0;
}

bool testConvergence() {
    PinholeCamera cam(640, 480, 250.0, 250.0, 320.0, 240.0);
    const double plane_z = 5.0;
    const Sophus::SE3d T_c_i = makeTci();

    VisualUpdateConfig cfg;
    VisualUpdate vu(cfg, cam, T_c_i);

    // Reference frame at T_w_i = identity: seed the map from a pixel grid.
    const Sophus::SE3d T_c_w_ref = T_c_i;
    const cv::Mat img_ref = renderPlaneImage(cam, T_c_w_ref, plane_z);
    const Sophus::SE3d T_w_ref = T_c_w_ref.inverse();

    std::vector<Eigen::Vector3d> scan_world;
    for (int v = 60; v <= 420; v += 40) {
        for (int u = 60; u <= 580; u += 40) {
            const Eigen::Vector2d px((double)u, (double)v);
            const Eigen::Vector3d dir_w = T_w_ref.rotationMatrix() * cam.unproject(px);
            const double t = (plane_z - T_w_ref.translation().z()) / dir_w.z();
            const Eigen::Vector3d p_w = T_w_ref.translation() + t * dir_w;

            auto pt = std::make_shared<VisualPoint>();
            pt->pos = p_w;
            pt->normal = Eigen::Vector3d(0.0, 0.0, -1.0);
            auto ftr = std::make_shared<Feature>();
            ftr->img = img_ref;
            ftr->px = px;
            ftr->T_c_w = T_c_w_ref;
            pt->obs.push_back(ftr);
            vu.map().insert(pt);
            scan_world.push_back(p_w);
        }
    }

    // Current frame: true pose + a perturbed prior.
    Eigen::Matrix<double, 6, 1> xi_true;
    xi_true << 0.03, -0.02, 0.01, 0.008, -0.006, 0.01;
    const Sophus::SE3d T_w_i_true = Sophus::SE3d::exp(xi_true);
    const cv::Mat img_cur = renderPlaneImage(cam, T_c_i * T_w_i_true.inverse(), plane_z);

    Eigen::Matrix<double, 6, 1> delta_err;
    delta_err << 0.04, -0.03, 0.02, 0.010, 0.008, -0.006;
    const Sophus::SE3d T_w_i_prior = T_w_i_true * Sophus::SE3d::exp(delta_err);

    const State prior = makeState(T_w_i_prior);
    const State posterior = vu.update(prior, img_cur, scan_world);

    auto poseError = [&](const State &s) {
        return (toSE3(s.pose).inverse() * T_w_i_true).log().norm();
    };
    const double err_prior = poseError(prior);
    const double err_post = poseError(posterior);
    std::cout << "  tracked " << vu.numTracked() << " points, pose error " << err_prior << " -> "
              << err_post << "\n";
    return vu.numTracked() >= 50 && err_post < 0.5 * err_prior;
}

}  // namespace

int main() {
    struct {
        const char *name;
        bool (*fn)();
    } tests[] = {
        {"jacobian_finite_diff", testJacobian},
        {"warp_round_trip", testWarp},
        {"update_convergence", testConvergence},
    };

    int failed = 0;
    for (const auto &t : tests) {
        std::cout << "[ RUN  ] " << t.name << "\n";
        const bool ok = t.fn();
        std::cout << (ok ? "[ PASS ] " : "[ FAIL ] ") << t.name << "\n";
        failed += ok ? 0 : 1;
    }
    return failed;
}
