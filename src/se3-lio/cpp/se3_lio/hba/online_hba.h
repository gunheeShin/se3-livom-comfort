// Online HBA (hku-mars/HBA a0cdd47): the lower layers' window BAs run on a worker thread while scans
// arrive; finish() solves the tail windows, the top-layer BA and the PGO exactly as the offline hba.cpp.
#pragma once

#include <Eigen/Core>
#include <memory>
#include <vector>

namespace se3_lio {
namespace hba {

struct Params {
    double voxel_size = 1.0;
    double downsample_size = 0.0;
    double eigen_ratio = 0.1;
    double reject_ratio = 0.05;
    int max_iter = 10;
    int layers = 3;
    int threads = 16;  // threads of one window BA (HBA thd_num)
};

struct Pose {
    Eigen::Matrix3d R;
    Eigen::Vector3d p;
};

struct WindowStat {
    int layer;   // 1 = scans
    int index;   // window index in that layer
    double ms;
    int pushed;  // scans pushed when the window finished
};

class OnlineHBA {
public:
    explicit OnlineHBA(const Params &params);
    ~OnlineHBA();
    // Deskewed body-frame scan (float32 xyz, n rows) with its LIO pose. The pose comes as a quaternion, the form
    // the offline HBA reads from pose.json, so both start from the same numbers.
    void push(const Eigen::Vector4d &q_xyzw, const Eigen::Vector3d &p, const float *xyz, int n);
    // Tail windows, top-layer BA, PGO. One pose per pushed scan, world frame re-anchored at the first LIO pose.
    std::vector<Pose> finish();
    const std::vector<WindowStat> &stats() const;
    double finish_ms() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace hba
}  // namespace se3_lio
