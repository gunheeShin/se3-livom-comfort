#include "lie.h"

namespace se3_lio {

namespace {
constexpr double kGravNorm = 9.81;
constexpr double kS2Tol = 1e-11;
}  // namespace

Eigen::Matrix<double, 3, 2> S2_basis(const Eigen::Vector3d &g) {
    Eigen::Matrix<double, 3, 2> B;
    if (g(0) + kGravNorm > kS2Tol) {
        B << -g(1), -g(2),
            kGravNorm - g(1) * g(1) / (kGravNorm + g(0)), -g(2) * g(1) / (kGravNorm + g(0)),
            -g(2) * g(1) / (kGravNorm + g(0)), kGravNorm - g(2) * g(2) / (kGravNorm + g(0));
        B /= kGravNorm;
    } else {
        B.setZero();
        B(1, 1) = -1;
        B(2, 0) = 1;
    }
    return B;
}

Eigen::Matrix<double, 3, 2> S2_Mx(const Eigen::Vector3d &g, const Eigen::Vector2d &delta) {
    Eigen::Matrix<double, 3, 2> B = S2_basis(g);
    if (delta.norm() < kS2Tol) {
        return -skew_sym(g) * B;
    }
    Eigen::Vector3d Bu = B * delta;
    return -Sophus::SO3d::exp(Bu).matrix() * skew_sym(g) *
           Sophus::SO3d::leftJacobian(Bu).transpose() * B;
}

Eigen::Matrix<double, 2, 3> S2_Nx_yy(const Eigen::Vector3d &g) {
    return S2_basis(g).transpose() * skew_sym(g) / (kGravNorm * kGravNorm);
}

State boxplus(const State &state, const ErrVector &delta) {
    State new_state = state;
    new_state.stamp = state.stamp;
    new_state.pose = state.pose * Sophus::SE3d::exp(delta.block<6, 1>(POSE, 0)).matrix();
    new_state.vel = Sophus::SO3d::exp(delta.block<3, 1>(ROT, 0)).inverse() *
                    (state.vel + delta.block<3, 1>(VEL, 0));
    new_state.bg = state.bg + delta.block<3, 1>(B_G, 0);
    new_state.ba = state.ba + delta.block<3, 1>(B_A, 0);
    new_state.grav =
        Sophus::SO3d::exp(S2_basis(state.grav) * delta.block<2, 1>(GRAV, 0)).matrix() *
        state.grav;

    return new_state;
}

ErrVector boxminus(const State &tgt_state, const State &src_state) {
    ErrVector dx = ErrVector::Zero();

    dx.block<6, 1>(POSE, 0) = Sophus::SE3d(src_state.pose.inverse() * tgt_state.pose).log();
    dx.block<3, 1>(VEL, 0) =
        Sophus::SO3d::exp(dx.block<3, 1>(ROT, 0)) * tgt_state.vel - src_state.vel;
    dx.block<3, 1>(B_G, 0) = tgt_state.bg - src_state.bg;
    dx.block<3, 1>(B_A, 0) = tgt_state.ba - src_state.ba;

    Eigen::Vector3d cross = src_state.grav.cross(tgt_state.grav);
    double v_sin = cross.norm();
    double v_cos = src_state.grav.dot(tgt_state.grav);
    double theta = std::atan2(v_sin, v_cos);
    if (v_sin < kS2Tol) {
        dx.block<2, 1>(GRAV, 0) =
            (std::fabs(theta) > kS2Tol) ? Eigen::Vector2d(M_PI, 0) : Eigen::Vector2d::Zero();
    } else {
        dx.block<2, 1>(GRAV, 0) = theta / v_sin * S2_basis(src_state.grav).transpose() * cross;
    }

    return dx;
}

std::tuple<ErrVector, CovMatrix> resetErrorState(const State &src_state, const State &tgt_state) {
    ErrVector dx_prior = boxminus(tgt_state, src_state);
    CovMatrix J_prior = CovMatrix::Identity();

    J_prior.block<6, 6>(POSE, POSE) =
        Sophus::SE3d::leftJacobian(-dx_prior.block<6, 1>(POSE, 0)).matrix().inverse();
    J_prior.block<3, 3>(VEL, VEL) = Sophus::SO3d::exp(dx_prior.block<3, 1>(ROT, 0)).matrix();
    // S2 tangent transport src -> tgt is Nx_yy(tgt) * Mx(src, dx); J_prior maps
    // the opposite way (P is transported as J^-1 P J^-T), hence the inverse.
    J_prior.block<2, 2>(GRAV, GRAV) =
        (S2_Nx_yy(tgt_state.grav) * S2_Mx(src_state.grav, dx_prior.block<2, 1>(GRAV, 0)))
            .inverse();

    return std::make_tuple(dx_prior, J_prior);
}

}  // namespace se3_lio
