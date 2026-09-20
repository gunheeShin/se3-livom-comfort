#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <memory>

#include "common/data_type.h"
#include "common/utils.h"
#include "hba/online_hba.h"
#include "pipeline/SE3_LIO.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace {

using Arr = py::array_t<double, py::array::c_style | py::array::forcecast>;
using Gray = py::array_t<uint8_t, py::array::c_style | py::array::forcecast>;

// (H, W) uint8 -> CV_8UC1 copy: the visual map keeps reference frames alive past this call,
// so the Mat must own its pixels. Empty array -> empty Mat (no visual update).
cv::Mat GrayFromArray(const Gray &gray) {
    if (gray.size() == 0) return cv::Mat();
    if (gray.ndim() != 2) throw std::invalid_argument("gray must have shape (H, W) uint8");
    return cv::Mat(static_cast<int>(gray.shape(0)), static_cast<int>(gray.shape(1)), CV_8UC1,
                   const_cast<uint8_t *>(gray.data()))
        .clone();
}

std::vector<cv::Mat> GraysFromList(const std::vector<Gray> &grays) {
    std::vector<cv::Mat> out;
    for (const auto &g : grays) out.push_back(GrayFromArray(g));
    return out;
}

std::vector<se3_lio::IMU> ImuFromArray(const Arr &imu) {
    if (imu.ndim() != 2 || imu.shape(1) != 7)
        throw std::invalid_argument("imu must have shape (M, 7): [t, ax, ay, az, gx, gy, gz]");
    auto u = imu.unchecked<2>();
    std::vector<se3_lio::IMU> out;
    out.reserve(u.shape(0));
    for (py::ssize_t i = 0; i < u.shape(0); ++i) {
        se3_lio::IMU s;
        s.header.timestamp = u(i, 0);
        s.linear_acceleration = Eigen::Vector3d(u(i, 1), u(i, 2), u(i, 3));
        s.angular_velocity = Eigen::Vector3d(u(i, 4), u(i, 5), u(i, 6));
        out.push_back(s);
    }
    return out;
}

se3_lio::LiDAR LidarFromArrays(const Arr &points, const Arr &point_times, double stamp) {
    if (points.ndim() != 2 || points.shape(1) != 3)
        throw std::invalid_argument("points must have shape (N, 3)");
    if (point_times.ndim() != 1 || point_times.shape(0) != points.shape(0))
        throw std::invalid_argument("point_times must have shape (N,) matching points");
    se3_lio::LiDAR lidar;
    lidar.header.timestamp = stamp;
    auto pts = points.unchecked<2>();
    auto times = point_times.unchecked<1>();
    lidar.points.reserve(pts.shape(0));
    for (py::ssize_t i = 0; i < pts.shape(0); ++i) {
        CustomPointType p;
        p.x = static_cast<float>(pts(i, 0));
        p.y = static_cast<float>(pts(i, 1));
        p.z = static_cast<float>(pts(i, 2));
        p.intensity = 0.0f;
        p.timestamp = times(i);
        lidar.points.push_back(p);
    }
    return lidar;
}

const se3_lio::PointCloudType &DeskewedCloud(const se3_lio::Measurement &m) {
    return m.raw_lidar.points.empty() ? m.lidar.points : m.raw_lidar.points;
}

py::array_t<double> CloudToArray(const se3_lio::PointCloudType &pts, bool with_meta = false) {
    py::ssize_t n = static_cast<py::ssize_t>(pts.size());
    py::array_t<double> out({n, static_cast<py::ssize_t>(with_meta ? 5 : 3)});
    auto c = out.mutable_unchecked<2>();
    for (py::ssize_t i = 0; i < n; ++i) {
        c(i, 0) = pts[i].x;
        c(i, 1) = pts[i].y;
        c(i, 2) = pts[i].z;
        if (with_meta) {
            c(i, 3) = pts[i].intensity;
            c(i, 4) = pts[i].timestamp;
        }
    }
    return out;
}

// Mirrors the per-frame data path of the ROS2 node (lio_node.cpp::process):
// build a synced Measurement, apply the LiDAR extrinsic, sort points by
// relative timestamp, then run a single estimatePose step.
class SE3LIOWrapper {
public:
    SE3LIOWrapper(const se3_lio::pipeline::SE3_LIO_Config &config,
                  const Eigen::Matrix4d &lidar_extrinsic)
        : pipeline_(config), extrinsic_(lidar_extrinsic) {}

    // Returns (state, cloud): cloud is the deskewed scan in the body frame (the
    // undistorted points estimatePose produced), matching the C++ node's
    // /local/cloud_registered_body.
    py::tuple RegisterFrame(
        const py::array_t<double, py::array::c_style | py::array::forcecast> &points,
        const py::array_t<double, py::array::c_style | py::array::forcecast> &point_times,
        const py::array_t<double, py::array::c_style | py::array::forcecast> &imu,
        double frame_stamp,
        const py::array_t<double, py::array::c_style | py::array::forcecast> &lidar_idx,
        double end_time, const std::vector<Gray> &grays) {
        if (points.ndim() != 2 || points.shape(1) != 3)
            throw std::invalid_argument("points must have shape (N, 3)");
        if (point_times.ndim() != 1 || point_times.shape(0) != points.shape(0))
            throw std::invalid_argument("point_times must have shape (N,) matching points");
        if (imu.ndim() != 2 || imu.shape(1) != 7)
            throw std::invalid_argument("imu must have shape (M, 7): [t, ax, ay, az, gx, gy, gz]");
        bool has_idx = lidar_idx.size() > 0;
        if (has_idx && (lidar_idx.ndim() != 1 || lidar_idx.shape(0) != points.shape(0)))
            throw std::invalid_argument("lidar_idx must have shape (N,) matching points");

        auto meas = std::make_shared<se3_lio::Measurement>();
        meas->is_synced = true;
        meas->end_time = end_time;

        auto imu_u = imu.unchecked<2>();
        meas->imu.reserve(imu_u.shape(0));
        for (py::ssize_t i = 0; i < imu_u.shape(0); ++i) {
            se3_lio::IMU sample;
            sample.header.timestamp = imu_u(i, 0);
            sample.linear_acceleration =
                Eigen::Vector3d(imu_u(i, 1), imu_u(i, 2), imu_u(i, 3));
            sample.angular_velocity = Eigen::Vector3d(imu_u(i, 4), imu_u(i, 5), imu_u(i, 6));
            meas->imu.push_back(sample);
        }

        meas->lidar.header.timestamp = frame_stamp;
        auto pts = points.unchecked<2>();
        auto times = point_times.unchecked<1>();
        auto idx = lidar_idx.unchecked<1>();
        meas->lidar.points.reserve(pts.shape(0));
        for (py::ssize_t i = 0; i < pts.shape(0); ++i) {
            CustomPointType point;
            point.x = static_cast<float>(pts(i, 0));
            point.y = static_cast<float>(pts(i, 1));
            point.z = static_cast<float>(pts(i, 2));
            point.intensity = has_idx ? static_cast<float>(idx(i)) : 0.0f;
            point.timestamp = times(i);
            meas->lidar.points.push_back(point);
        }

        meas->lidar.points = se3_lio::transformPointCloud(meas->lidar.points, extrinsic_);
        std::sort(meas->lidar.points.begin(), meas->lidar.points.end(),
                  [](const CustomPointType &a, const CustomPointType &b) {
                      return a.timestamp < b.timestamp;
                  });

        se3_lio::MeasurementPtr meas_ptr = meas;
        pipeline_.estimatePose(meas_ptr, GraysFromList(grays));

        // undistortCloud() overwrote meas->lidar.points in place with the
        // deskewed (body-frame) points; return them next to the state.
        return py::make_tuple(pipeline_.getState(), CloudToArray(DeskewedCloud(*meas)));
    }

    // Multi-LiDAR step: one scan per sensor (sensor frame, own start stamp);
    // the core applies config.lidar_extrinsics and merges (SE3_LIO::mergeLiDARs).
    py::tuple RegisterMulti(const std::vector<Arr> &points_list, const std::vector<Arr> &times_list,
                            const std::vector<double> &stamps, const Arr &imu, double end_time,
                            const std::vector<Gray> &grays) {
        if (points_list.size() != times_list.size() || points_list.size() != stamps.size())
            throw std::invalid_argument("points_list, times_list, stamps must have the same length");
        auto meas = std::make_shared<se3_lio::Measurement>();
        meas->is_synced = true;
        meas->end_time = end_time;
        meas->imu = ImuFromArray(imu);
        meas->lidars = BuildLidars(points_list, times_list, stamps);
        se3_lio::MeasurementPtr meas_ptr = meas;
        pipeline_.estimatePose(meas_ptr, GraysFromList(grays));
        return py::make_tuple(pipeline_.getState(), CloudToArray(DeskewedCloud(*meas)));
    }

    // Merge only (no estimation): returns (N, 5) [x y z lidar_idx t_offset] and the merged stamp.
    py::tuple MergeLidars(const std::vector<Arr> &points_list, const std::vector<Arr> &times_list,
                          const std::vector<double> &stamps) {
        se3_lio::Measurement meas;
        meas.lidars = BuildLidars(points_list, times_list, stamps);
        pipeline_.mergeLiDARs(meas);
        return py::make_tuple(CloudToArray(meas.lidar.points, true), meas.lidar.header.timestamp);
    }

private:
    std::vector<se3_lio::LiDAR> BuildLidars(const std::vector<Arr> &points_list,
                                            const std::vector<Arr> &times_list,
                                            const std::vector<double> &stamps) {
        std::vector<se3_lio::LiDAR> lidars;
        for (size_t i = 0; i < points_list.size(); ++i)
            lidars.push_back(LidarFromArrays(points_list[i], times_list[i], stamps[i]));
        return lidars;
    }

public:
    int NumTracked() const { return pipeline_.numTrackedPoints(); }

private:
    se3_lio::pipeline::SE3_LIO pipeline_;
    Eigen::Matrix4d extrinsic_;
};

class OnlineHBAWrapper {
public:
    explicit OnlineHBAWrapper(const se3_lio::hba::Params &params) : hba_(params) {}

    void Push(const Eigen::Vector4d &q_xyzw, const Eigen::Vector3d &p,
              const py::array_t<float, py::array::c_style | py::array::forcecast> &xyz, double stamp, const Eigen::Vector3d &acc,
              const Eigen::Vector3d &ba, const Eigen::Vector3d &grav) {
        if (xyz.ndim() != 2 || xyz.shape(1) != 3) throw std::invalid_argument("xyz must have shape (N, 3)");
        hba_.push(q_xyzw, p, xyz.data(), static_cast<int>(xyz.shape(0)), stamp, acc, ba, grav);
    }

    py::array_t<double> Finish() {
        std::vector<se3_lio::hba::Pose> poses;
        {
            py::gil_scoped_release release;
            poses = hba_.finish();
        }
        py::array_t<double> out({static_cast<py::ssize_t>(poses.size()), py::ssize_t(4), py::ssize_t(4)});
        auto o = out.mutable_unchecked<3>();
        for (size_t i = 0; i < poses.size(); i++) {
            Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
            T.block<3, 3>(0, 0) = poses[i].R;
            T.block<3, 1>(0, 3) = poses[i].p;
            for (int r = 0; r < 4; r++)
                for (int c = 0; c < 4; c++) o(i, r, c) = T(r, c);
        }
        return out;
    }

    std::vector<std::tuple<int, int, int, int, double, double, int>> Stats() const {
        std::vector<std::tuple<int, int, int, int, double, double, int>> out;
        for (const auto &s : hba_.stats()) out.emplace_back(s.round, s.start_scan, s.kfs, s.iters, s.ba_ms, s.pgo_ms, s.done_scan);
        return out;
    }

    double PendingMs() const { return hba_.pending_ms(); }

private:
    se3_lio::hba::OnlineHBA hba_;
};

}  // namespace

PYBIND11_MODULE(se3_lio_pybind, m) {
    using se3_lio::State;
    using Config = se3_lio::pipeline::SE3_LIO_Config;
    using VisualConfig = se3_lio::VisualUpdateConfig;
    using CameraConfig = se3_lio::pipeline::CameraConfig;

    py::class_<CameraConfig>(m, "_CameraConfig")
        .def(py::init<>())
        .def_readwrite("width", &CameraConfig::width)
        .def_readwrite("height", &CameraConfig::height)
        .def_readwrite("fx", &CameraConfig::fx)
        .def_readwrite("fy", &CameraConfig::fy)
        .def_readwrite("cx", &CameraConfig::cx)
        .def_readwrite("cy", &CameraConfig::cy)
        .def_readwrite("dist", &CameraConfig::dist)
        .def_readwrite("fisheye", &CameraConfig::fisheye)
        .def_readwrite("T_cam_imu", &CameraConfig::T_cam_imu);

    py::class_<VisualConfig>(m, "_VisualConfig")
        .def(py::init<>())
        .def_readwrite("patch_size", &VisualConfig::patch_size)
        .def_readwrite("pyramid_levels", &VisualConfig::pyramid_levels)
        .def_readwrite("grid_size", &VisualConfig::grid_size)
        .def_readwrite("max_iter", &VisualConfig::max_iter)
        .def_readwrite("img_point_cov", &VisualConfig::img_point_cov)
        .def_readwrite("outlier_threshold", &VisualConfig::outlier_threshold)
        .def_readwrite("depth_continuous_thresh", &VisualConfig::depth_continuous_thresh)
        .def_readwrite("voxel_size", &VisualConfig::voxel_size)
        .def_readwrite("max_obs", &VisualConfig::max_obs)
        .def_readwrite("max_view_angle_deg", &VisualConfig::max_view_angle_deg)
        .def_readwrite("max_gyro_deg", &VisualConfig::max_gyro_deg)
        .def_readwrite("keep_cov", &VisualConfig::keep_cov)
        .def_readwrite("probe", &VisualConfig::probe)
        .def_readwrite("pose_only", &VisualConfig::pose_only)
        .def_readwrite("verbose", &VisualConfig::verbose);

    py::class_<Config>(m, "_SE3LIOConfig")
        .def(py::init<>())
        .def_readwrite("acc_noise", &Config::acc_noise)
        .def_readwrite("gyr_noise", &Config::gyr_noise)
        .def_readwrite("bg_noise", &Config::bg_noise)
        .def_readwrite("ba_noise", &Config::ba_noise)
        .def_readwrite("lidar_range_noise", &Config::lidar_range_noise)
        .def_readwrite("lidar_angle_noise", &Config::lidar_angle_noise)
        .def_readwrite("lidar_range_noises", &Config::lidar_range_noises)
        .def_readwrite("lidar_angle_noises", &Config::lidar_angle_noises)
        .def_readwrite("lidar_extrinsics", &Config::lidar_extrinsics)
        .def_readwrite("lidar_min_ranges", &Config::lidar_min_ranges)
        .def_readwrite("downsample_resolution", &Config::downsample_resolution)
        .def_readwrite("downsample_centroid", &Config::downsample_centroid)
        .def_readwrite("downsample_target_inliers", &Config::downsample_target_inliers)
        .def_readwrite("downsample_max_resolution", &Config::downsample_max_resolution)
        .def_readwrite("downsample_start_resolution", &Config::downsample_start_resolution)
        .def_readwrite("max_iter", &Config::max_iter)
        .def_readwrite("voxel_map_resolution", &Config::voxel_map_resolution)
        .def_readwrite("voxel_map_max_layer", &Config::voxel_map_max_layer)
        .def_readwrite("voxel_map_layer_size", &Config::voxel_map_layer_size)
        .def_readwrite("voxel_map_max_point_size", &Config::voxel_map_max_point_size)
        .def_readwrite("voxel_map_plane_thres", &Config::voxel_map_plane_thres)
        .def_readwrite("voxel_map_sliding_en", &Config::voxel_map_sliding_en)
        .def_readwrite("voxel_map_sliding_thresh", &Config::voxel_map_sliding_thresh)
        .def_readwrite("voxel_map_half_size", &Config::voxel_map_half_size)
        .def_readwrite("verbose", &Config::verbose)
        .def_readwrite("residual_floor", &Config::residual_floor)
        .def_readwrite("deskew_cov_en", &Config::deskew_cov_en)
        .def_readwrite("init_fl2", &Config::init_fl2)
        .def_readwrite("const_r", &Config::const_r)
        .def_readwrite("visual_en", &Config::visual_en)
        .def_readwrite("visual", &Config::visual)
        .def_readwrite("cameras", &Config::cameras);

    py::class_<State>(m, "_State")
        .def(py::init<>())
        .def_readonly("stamp", &State::stamp)
        .def_readonly("num_inliers", &State::num_inliers)
        .def_readonly("residual", &State::residual)
        .def_readonly("pose", &State::pose)
        .def_readonly("vel", &State::vel)
        .def_readonly("bg", &State::bg)
        .def_readonly("ba", &State::ba)
        .def_readonly("grav", &State::grav)
        .def_readonly("covariance", &State::covariance);

    py::class_<SE3LIOWrapper>(m, "_SE3LIO")
        .def(py::init<const Config &, const Eigen::Matrix4d &>(), "config"_a,
             "lidar_extrinsic"_a)
        .def("_register_frame", &SE3LIOWrapper::RegisterFrame, "points"_a, "point_times"_a,
             "imu"_a, "frame_stamp"_a, "lidar_idx"_a = py::array_t<double>(), "end_time"_a = 0.0,
             "grays"_a = std::vector<Gray>())
        .def("_register_multi", &SE3LIOWrapper::RegisterMulti, "points_list"_a, "times_list"_a,
             "stamps"_a, "imu"_a, "end_time"_a = 0.0, "grays"_a = std::vector<Gray>())
        .def("_merge_lidars", &SE3LIOWrapper::MergeLidars, "points_list"_a, "times_list"_a,
             "stamps"_a)
        .def("num_tracked", &SE3LIOWrapper::NumTracked);

    using HBAParams = se3_lio::hba::Params;
    py::class_<HBAParams>(m, "_HBAParams")
        .def(py::init<>())
        .def_readwrite("voxel_size", &HBAParams::voxel_size)
        .def_readwrite("downsample_size", &HBAParams::downsample_size)
        .def_readwrite("eigen_ratio", &HBAParams::eigen_ratio)
        .def_readwrite("reject_ratio", &HBAParams::reject_ratio)
        .def_readwrite("max_iter", &HBAParams::max_iter)
        .def_readwrite("layers", &HBAParams::layers)
        .def_readwrite("threads", &HBAParams::threads)
        .def_readwrite("every", &HBAParams::every)
        .def_readwrite("hess_const", &HBAParams::hess_const)
        .def_readwrite("gravity_sigma_deg", &HBAParams::gravity_sigma_deg)
        .def_readwrite("gravity_file", &HBAParams::gravity_file);

    py::class_<OnlineHBAWrapper>(m, "_OnlineHBA")
        .def(py::init<const HBAParams &>(), "params"_a)
        .def("push", &OnlineHBAWrapper::Push, "q_xyzw"_a, "p"_a, "xyz"_a, "stamp"_a, "acc"_a, "ba"_a, "grav"_a)
        .def("finish", &OnlineHBAWrapper::Finish)
        .def("stats", &OnlineHBAWrapper::Stats)
        .def("pending_ms", &OnlineHBAWrapper::PendingMs);
}
