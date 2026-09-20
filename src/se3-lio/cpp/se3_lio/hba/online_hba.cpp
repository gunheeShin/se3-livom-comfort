#include "hba/online_hba.h"

#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>

#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <stdexcept>
#include <thread>

// ba.hpp / tools.hpp define globals (thd_num, layer_limit, ...): include them from this TU only.
#include "hba/ba.hpp"
#include "hba/mypcl.hpp"

namespace se3_lio {
namespace hba {
namespace {

using Cloud = pcl::PointCloud<PointType>;

struct Layer {
    vector<mypcl::pose> pose_vec;
    vector<Cloud::Ptr> pcds;      // layer 1: scans; above: merged keyframes. Freed once no window needs them.
    PLV(6) hessians;              // layer 1: per window (hba.cpp parallel_comp); top: all pairs (global_ba)
    int next_window = 0;
};

// hba.cpp cut_voxel
void cut_voxel(unordered_map<VOXEL_LOC, OCTO_TREE_ROOT *> &feat_map, pcl::PointCloud<PointType> &feat_pt,
               Eigen::Quaterniond q, Eigen::Vector3d t, int fnum, double voxel_size, int window_size,
               float eigen_ratio) {
    float loc_xyz[3];
    for (PointType &p_c : feat_pt.points) {
        Eigen::Vector3d pvec_orig(p_c.x, p_c.y, p_c.z);
        Eigen::Vector3d pvec_tran = q * pvec_orig + t;

        for (int j = 0; j < 3; j++) {
            loc_xyz[j] = pvec_tran[j] / voxel_size;
            if (loc_xyz[j] < 0) loc_xyz[j] -= 1.0;
        }

        VOXEL_LOC position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
        auto iter = feat_map.find(position);
        if (iter != feat_map.end()) {
            iter->second->vec_orig[fnum].push_back(pvec_orig);
            iter->second->vec_tran[fnum].push_back(pvec_tran);

            iter->second->sig_orig[fnum].push(pvec_orig);
            iter->second->sig_tran[fnum].push(pvec_tran);
        } else {
            OCTO_TREE_ROOT *ot = new OCTO_TREE_ROOT(window_size, eigen_ratio);
            ot->vec_orig[fnum].push_back(pvec_orig);
            ot->vec_tran[fnum].push_back(pvec_tran);
            ot->sig_orig[fnum].push(pvec_orig);
            ot->sig_tran[fnum].push(pvec_tran);

            ot->voxel_center[0] = (0.5 + position.x) * voxel_size;
            ot->voxel_center[1] = (0.5 + position.y) * voxel_size;
            ot->voxel_center[2] = (0.5 + position.z) * voxel_size;
            ot->quater_length = voxel_size / 4.0;
            ot->layer = 0;
            feat_map[position] = ot;
        }
    }
}

// One window of hba.cpp parallel_comp: frames [start, start+w) of `layer` (layer_num as in HBA, 1 = scans).
void solve_window(const Params &prm, Layer &layer, int layer_num, int start, int w, PLV(6) & hess_out,
                  Cloud::Ptr &keyframe) {
    vector<Cloud::Ptr> src_pc(w), raw_pc(w);

    double residual_cur = 0, residual_pre = 0;
    vector<IMUST> x_buf(w);
    for (int j = 0; j < w; j++) {
        x_buf[j].R = layer.pose_vec[start + j].q.toRotationMatrix();
        x_buf[j].p = layer.pose_vec[start + j].t;
    }

    if (layer_num != 1)
        for (int j = 0; j < w; j++) src_pc[j] = (*layer.pcds[start + j]).makeShared();

    size_t mem_cost = 0;
    for (int loop = 0; loop < prm.max_iter; loop++) {
        if (layer_num == 1)
            for (int j = 0; j < w; j++) {
                if (loop == 0) raw_pc[j] = layer.pcds[start + j];
                src_pc[j] = (*raw_pc[j]).makeShared();
            }

        unordered_map<VOXEL_LOC, OCTO_TREE_ROOT *> surf_map;

        for (int j = 0; j < w; j++) {
            if (prm.downsample_size > 0) downsample_voxel(*src_pc[j], prm.downsample_size);
            cut_voxel(surf_map, *src_pc[j], Eigen::Quaterniond(x_buf[j].R), x_buf[j].p, j, prm.voxel_size, w,
                      prm.eigen_ratio);
        }
        for (auto iter = surf_map.begin(); iter != surf_map.end(); ++iter) iter->second->recut();

        VOX_HESS voxhess(w);
        for (auto iter = surf_map.begin(); iter != surf_map.end(); iter++) iter->second->tras_opt(voxhess);

        VOX_OPTIMIZER opt_lsv(w);
        opt_lsv.remove_outlier(x_buf, voxhess, prm.reject_ratio);
        PLV(6) hess_vec;
        opt_lsv.damping_iter(x_buf, voxhess, residual_cur, hess_vec, mem_cost);

        for (auto iter = surf_map.begin(); iter != surf_map.end(); ++iter) delete iter->second;

        if (loop > 0 && abs(residual_pre - residual_cur) / abs(residual_cur) < 0.05 || loop == prm.max_iter - 1) {
            hess_out = hess_vec;
            break;
        }
        residual_pre = residual_cur;
    }

    keyframe.reset(new Cloud);
    for (int j = 0; j < w; j++) {
        Eigen::Quaterniond q_tmp;
        Eigen::Vector3d t_tmp;
        assign_qt(q_tmp, t_tmp, Eigen::Quaterniond(x_buf[0].R.inverse() * x_buf[j].R),
                  x_buf[0].R.inverse() * (x_buf[j].p - x_buf[0].p));

        Cloud::Ptr pc_oneframe(new Cloud);
        mypcl::transform_pointcloud(*src_pc[j], *pc_oneframe, t_tmp, q_tmp);
        keyframe = mypcl::append_cloud(keyframe, *pc_oneframe);
    }
    downsample_voxel(*keyframe, 0.05);
}

// hba.cpp global_ba (FULL_HESS)
void global_ba(const Params &prm, Layer &layer) {
    int window_size = layer.pose_vec.size();
    vector<IMUST> x_buf(window_size);
    for (int i = 0; i < window_size; i++) {
        x_buf[i].R = layer.pose_vec[i].q.toRotationMatrix();
        x_buf[i].p = layer.pose_vec[i].t;
    }

    vector<Cloud::Ptr> src_pc(window_size);
    for (int i = 0; i < window_size; i++) src_pc[i] = (*layer.pcds[i]).makeShared();

    double residual_cur = 0, residual_pre = 0;
    size_t mem_cost = 0;
    for (int loop = 0; loop < prm.max_iter; loop++) {
        unordered_map<VOXEL_LOC, OCTO_TREE_ROOT *> surf_map;

        for (int i = 0; i < window_size; i++) {
            if (prm.downsample_size > 0) downsample_voxel(*src_pc[i], prm.downsample_size);
            cut_voxel(surf_map, *src_pc[i], Eigen::Quaterniond(x_buf[i].R), x_buf[i].p, i, prm.voxel_size,
                      window_size, prm.eigen_ratio * 2);
        }
        for (auto iter = surf_map.begin(); iter != surf_map.end(); ++iter) iter->second->recut();

        VOX_HESS voxhess(window_size);
        for (auto iter = surf_map.begin(); iter != surf_map.end(); ++iter) iter->second->tras_opt(voxhess);

        VOX_OPTIMIZER opt_lsv(window_size);
        opt_lsv.remove_outlier(x_buf, voxhess, prm.reject_ratio);
        PLV(6) hess_vec;
        opt_lsv.damping_iter(x_buf, voxhess, residual_cur, hess_vec, mem_cost);

        for (auto iter = surf_map.begin(); iter != surf_map.end(); ++iter) delete iter->second;

        if (loop > 0 && abs(residual_pre - residual_cur) / abs(residual_cur) < 0.05 || loop == prm.max_iter - 1) {
            layer.hessians = hess_vec;
            break;
        }
        residual_pre = residual_cur;
    }
    for (int i = 0; i < window_size; i++) {
        layer.pose_vec[i].q = Eigen::Quaterniond(x_buf[i].R);
        layer.pose_vec[i].t = x_buf[i].p;
    }
}

// hba.hpp pose_graph_optimization. The result is re-anchored at the input's first pose the way
// mypcl::write_pose + run_hba.sh do (first-frame relative, then the original first pose put back).
vector<Pose> pgo(const Layer &bottom, const Layer &top, int total_layer_num) {
    vector<mypcl::pose> upper_pose = top.pose_vec, init_pose = bottom.pose_vec;
    const PLV(6) &upper_cov = top.hessians, &init_cov = bottom.hessians;

    int cnt = 0;
    gtsam::Values initial;
    gtsam::NonlinearFactorGraph graph;
    gtsam::Vector Vector6(6);
    Vector6 << 1e-6, 1e-6, 1e-6, 1e-8, 1e-8, 1e-8;
    gtsam::noiseModel::Diagonal::shared_ptr priorModel = gtsam::noiseModel::Diagonal::Variances(Vector6);
    initial.insert(0, gtsam::Pose3(gtsam::Rot3(init_pose[0].q.toRotationMatrix()), gtsam::Point3(init_pose[0].t)));
    graph.add(gtsam::PriorFactor<gtsam::Pose3>(
        0, gtsam::Pose3(gtsam::Rot3(init_pose[0].q.toRotationMatrix()), gtsam::Point3(init_pose[0].t)), priorModel));

    for (uint i = 0; i < init_pose.size(); i++) {
        if (i > 0)
            initial.insert(i, gtsam::Pose3(gtsam::Rot3(init_pose[i].q.toRotationMatrix()), gtsam::Point3(init_pose[i].t)));

        if (i % GAP == 0 && cnt < (int)init_cov.size())
            for (int j = 0; j < WIN_SIZE - 1; j++)
                for (int k = j + 1; k < WIN_SIZE; k++) {
                    if (i + j + 1 >= init_pose.size() || i + k >= init_pose.size()) break;

                    cnt++;
                    if (init_cov[cnt - 1].norm() < 1e-20) continue;

                    Eigen::Vector3d t_ab = init_pose[i + j].t;
                    Eigen::Matrix3d R_ab = init_pose[i + j].q.toRotationMatrix();
                    t_ab = R_ab.transpose() * (init_pose[i + k].t - t_ab);
                    R_ab = R_ab.transpose() * init_pose[i + k].q.toRotationMatrix();
                    gtsam::Rot3 R_sam(R_ab);
                    gtsam::Point3 t_sam(t_ab);

                    Vector6 << fabs(1.0 / init_cov[cnt - 1](0)), fabs(1.0 / init_cov[cnt - 1](1)),
                        fabs(1.0 / init_cov[cnt - 1](2)), fabs(1.0 / init_cov[cnt - 1](3)),
                        fabs(1.0 / init_cov[cnt - 1](4)), fabs(1.0 / init_cov[cnt - 1](5));
                    gtsam::noiseModel::Diagonal::shared_ptr odometryNoise =
                        gtsam::noiseModel::Diagonal::Variances(Vector6);
                    gtsam::NonlinearFactor::shared_ptr factor(new gtsam::BetweenFactor<gtsam::Pose3>(
                        i + j, i + k, gtsam::Pose3(R_sam, t_sam), odometryNoise));
                    graph.push_back(factor);
                }
    }

    size_t stride = 1;
    for (int l = 0; l < total_layer_num - 1; l++) stride *= GAP;
    int pose_size = upper_pose.size();
    cnt = 0;
    for (int i = 0; i < pose_size - 1; i++)
        for (int j = i + 1; j < pose_size; j++) {
            cnt++;
            if (upper_cov[cnt - 1].norm() < 1e-20) continue;

            Eigen::Vector3d t_ab = upper_pose[i].t;
            Eigen::Matrix3d R_ab = upper_pose[i].q.toRotationMatrix();
            t_ab = R_ab.transpose() * (upper_pose[j].t - t_ab);
            R_ab = R_ab.transpose() * upper_pose[j].q.toRotationMatrix();
            gtsam::Rot3 R_sam(R_ab);
            gtsam::Point3 t_sam(t_ab);

            Vector6 << fabs(1.0 / upper_cov[cnt - 1](0)), fabs(1.0 / upper_cov[cnt - 1](1)),
                fabs(1.0 / upper_cov[cnt - 1](2)), fabs(1.0 / upper_cov[cnt - 1](3)),
                fabs(1.0 / upper_cov[cnt - 1](4)), fabs(1.0 / upper_cov[cnt - 1](5));
            gtsam::noiseModel::Diagonal::shared_ptr odometryNoise = gtsam::noiseModel::Diagonal::Variances(Vector6);
            gtsam::NonlinearFactor::shared_ptr factor(new gtsam::BetweenFactor<gtsam::Pose3>(
                i * stride, j * stride, gtsam::Pose3(R_sam, t_sam), odometryNoise));
            graph.push_back(factor);
        }

    gtsam::ISAM2Params parameters;
    parameters.relinearizeThreshold = 0.01;
    parameters.relinearizeSkip = 1;
    gtsam::ISAM2 isam(parameters);
    isam.update(graph, initial);
    isam.update();

    gtsam::Values results = isam.calculateEstimate();

    vector<Pose> out(results.size());
    const Eigen::Matrix3d R0o = init_pose[0].q.toRotationMatrix();
    const Eigen::Vector3d p0o = init_pose[0].t;
    gtsam::Pose3 pose0 = results.at(0).cast<gtsam::Pose3>();
    const Eigen::Matrix3d R0r = pose0.rotation().matrix();
    const Eigen::Vector3d p0r = pose0.translation();
    for (uint i = 0; i < results.size(); i++) {
        gtsam::Pose3 pose = results.at(i).cast<gtsam::Pose3>();
        Eigen::Quaterniond q0(R0r), q(pose.rotation().matrix());
        Eigen::Vector3d t = q0.inverse() * (pose.translation() - p0r);
        Eigen::Quaterniond qr = q0.inverse() * q;
        out[i].R = (init_pose[0].q * qr).toRotationMatrix();
        out[i].p = init_pose[0].q * t + p0o;
    }
    return out;
}

}  // namespace

struct OnlineHBA::Impl {
    Params prm;
    vector<Layer> layers;
    std::vector<WindowStat> stats;
    double finish_ms = 0;

    std::mutex mu;
    std::condition_variable cv;
    std::deque<std::pair<mypcl::pose, Cloud::Ptr>> pending;
    bool stop = false;
    std::atomic<int> pushed{0};
    std::thread worker;

    void solve(int l, int i, int w) {
        Layer &L = layers[l];
        double t0 = now_sec();
        PLV(6) hv;
        Cloud::Ptr kf;
        solve_window(prm, L, l + 1, i * GAP, w, hv, kf);
        if (l == 0) {
            L.hessians.resize(i * (WIN_SIZE - 1) * WIN_SIZE / 2 + hv.size());
            for (size_t j = 0; j < hv.size(); j++) L.hessians[i * (WIN_SIZE - 1) * WIN_SIZE / 2 + j] = hv[j];
        }
        layers[l + 1].pose_vec.push_back(L.pose_vec[i * GAP]);
        layers[l + 1].pcds.push_back(kf);
        for (int k = i * GAP; k < i * GAP + w && k < (i + 1) * GAP; k++) L.pcds[k].reset();
        L.next_window = i + 1;
        stats.push_back({l + 1, i, (now_sec() - t0) * 1e3, pushed.load()});
    }

    // Every full window whose frames have arrived, layer by layer (a keyframe may complete a window above).
    void advance() {
        for (int l = 0; l + 1 < prm.layers; l++)
            while ((int)layers[l].pose_vec.size() >= layers[l].next_window * GAP + WIN_SIZE)
                solve(l, layers[l].next_window, WIN_SIZE);
    }

    // hba.cpp parallel_tail: the last, shorter window when (N - WIN_SIZE) % GAP != 0
    void tail(int l) {
        int N = layers[l].pose_vec.size();
        if (N < WIN_SIZE) throw std::runtime_error("OnlineHBA: a layer has fewer than WIN_SIZE frames");
        int gap_num = (N - WIN_SIZE) / GAP, tail = (N - WIN_SIZE) % GAP;
        if (tail > 0) solve(l, gap_num + 1, N - GAP * (gap_num + 1));
        for (int k = 0; k < N; k++) layers[l].pcds[k].reset();
    }

    void run() {
        for (;;) {
            std::deque<std::pair<mypcl::pose, Cloud::Ptr>> batch;
            {
                std::unique_lock<std::mutex> lk(mu);
                cv.wait(lk, [&] { return stop || !pending.empty(); });
                if (pending.empty()) return;
                batch.swap(pending);
            }
            for (auto &e : batch) {
                layers[0].pose_vec.push_back(e.first);
                layers[0].pcds.push_back(e.second);
            }
            advance();
        }
    }
};

OnlineHBA::OnlineHBA(const Params &params) : impl_(new Impl) {
    impl_->prm = params;
    impl_->layers.resize(params.layers);
    thd_num = params.threads;
    impl_->worker = std::thread(&Impl::run, impl_.get());
}

OnlineHBA::~OnlineHBA() {
    if (impl_->worker.joinable()) {
        {
            std::lock_guard<std::mutex> lk(impl_->mu);
            impl_->stop = true;
        }
        impl_->cv.notify_all();
        impl_->worker.join();
    }
}

void OnlineHBA::push(const Eigen::Vector4d &q_xyzw, const Eigen::Vector3d &p, const float *xyz, int n) {
    Cloud::Ptr pc(new Cloud);
    pc->points.resize(n);
    for (int i = 0; i < n; i++) {
        pc->points[i].x = xyz[3 * i];
        pc->points[i].y = xyz[3 * i + 1];
        pc->points[i].z = xyz[3 * i + 2];
    }
    {
        std::lock_guard<std::mutex> lk(impl_->mu);
        impl_->pending.emplace_back(mypcl::pose(Eigen::Quaterniond(q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]), p), pc);
    }
    impl_->pushed++;
    impl_->cv.notify_one();
}

std::vector<Pose> OnlineHBA::finish() {
    {
        std::lock_guard<std::mutex> lk(impl_->mu);
        impl_->stop = true;
    }
    impl_->cv.notify_all();
    impl_->worker.join();

    double t0 = now_sec();
    for (int l = 0; l + 1 < impl_->prm.layers; l++) {
        impl_->advance();
        impl_->tail(l);
    }
    global_ba(impl_->prm, impl_->layers.back());
    auto out = pgo(impl_->layers[0], impl_->layers.back(), impl_->prm.layers);
    impl_->finish_ms = (now_sec() - t0) * 1e3;
    return out;
}

const std::vector<WindowStat> &OnlineHBA::stats() const { return impl_->stats; }
double OnlineHBA::finish_ms() const { return impl_->finish_ms; }

}  // namespace hba
}  // namespace se3_lio
