#include "hba/online_hba.h"

#include <gtsam/geometry/Pose3.h>
#include <gtsam/navigation/AttitudeFactor.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>

#include <atomic>
#include <condition_variable>
#include <deque>
#include <fstream>
#include <mutex>
#include <thread>
#include <cmath>
#include <tuple>
#include <unordered_map>

// ba.hpp / tools.hpp define globals (thd_num, layer_limit, ...): include them from this TU only.
#include "hba/ba.hpp"
#include "hba/mypcl.hpp"

namespace se3_lio {
namespace hba {
namespace {

using Cloud = pcl::PointCloud<PointType>;

VOXEL_LOC root_key(const Eigen::Vector3d &w, double voxel_size) {
    float loc[3];
    for (int j = 0; j < 3; j++) {
        loc[j] = w[j] / voxel_size;
        if (loc[j] < 0) loc[j] -= 1.0;
    }
    return VOXEL_LOC((int64_t)loc[0], (int64_t)loc[1], (int64_t)loc[2]);
}

// hba.cpp cut_voxel over all keyframes on `threads` threads. Pass A (by keyframe) decides which thread owns each
// point's root voxel; pass B (by owner) inserts, keyframe by keyframe, into per-thread maps that are then joined
// thread by thread. Deterministic for a given input; the map order differs from the sequential cut_voxel.
void cut_voxel_parallel(unordered_map<VOXEL_LOC, OCTO_TREE_ROOT *> &feat_map, const vector<Cloud::Ptr> &pcds,
                        const vector<IMUST> &x_buf, double voxel_size, int n, float eigen_ratio, int threads) {
    std::hash<VOXEL_LOC> hasher;
    vector<vector<uint8_t>> owner(n);
    vector<std::thread> pool;
    for (int t = 0; t < threads; t++)
        pool.emplace_back([&, t] {
            for (int i = t; i < n; i += threads) {
                Eigen::Quaterniond q(x_buf[i].R);
                owner[i].resize(pcds[i]->points.size());
                size_t k = 0;
                for (const PointType &p : pcds[i]->points)
                    owner[i][k++] = hasher(root_key(q * Eigen::Vector3d(p.x, p.y, p.z) + x_buf[i].p, voxel_size)) % threads;
            }
        });
    for (auto &th : pool) th.join();
    pool.clear();

    vector<unordered_map<VOXEL_LOC, OCTO_TREE_ROOT *>> maps(threads);
    for (int t = 0; t < threads; t++)
        pool.emplace_back([&, t] {
            for (int i = 0; i < n; i++) {
                Eigen::Quaterniond q(x_buf[i].R);
                size_t k = 0;
                for (const PointType &p : pcds[i]->points) {
                    if (owner[i][k++] != t) continue;
                    Eigen::Vector3d pvec_orig(p.x, p.y, p.z);
                    Eigen::Vector3d pvec_tran = q * pvec_orig + x_buf[i].p;
                    VOXEL_LOC position = root_key(pvec_tran, voxel_size);
                    auto iter = maps[t].find(position);
                    OCTO_TREE_ROOT *ot;
                    if (iter != maps[t].end()) {
                        ot = iter->second;
                    } else {
                        ot = new OCTO_TREE_ROOT(n, eigen_ratio);
                        ot->voxel_center[0] = (0.5 + position.x) * voxel_size;
                        ot->voxel_center[1] = (0.5 + position.y) * voxel_size;
                        ot->voxel_center[2] = (0.5 + position.z) * voxel_size;
                        ot->quater_length = voxel_size / 4.0;
                        ot->layer = 0;
                        maps[t][position] = ot;
                    }
                    ot->vec_orig[i].push_back(pvec_orig);
                    ot->vec_tran[i].push_back(pvec_tran);
                    ot->sig_orig[i].push(pvec_orig);
                    ot->sig_tran[i].push(pvec_tran);
                }
            }
        });
    for (auto &th : pool) th.join();
    for (int t = 0; t < threads; t++)
        for (auto &kv : maps[t]) feat_map[kv.first] = kv.second;
}

void recut_parallel(unordered_map<VOXEL_LOC, OCTO_TREE_ROOT *> &feat_map, int threads) {
    vector<OCTO_TREE_ROOT *> roots;
    for (auto &kv : feat_map) roots.push_back(kv.second);
    vector<std::thread> pool;
    for (int t = 0; t < threads; t++)
        pool.emplace_back([&, t] {
            for (size_t i = t; i < roots.size(); i += threads) roots[i]->recut();
        });
    for (auto &th : pool) th.join();
}

// hba.cpp global_ba over the top-layer keyframes (poses in place, already downsampled clouds), all-pair Hessians out
void global_ba(const Params &prm, vector<mypcl::pose> &poses, const vector<Cloud::Ptr> &pcds, PLV(6) & hess_out, int &iters) {
    int n = poses.size();
    vector<IMUST> x_buf(n);
    for (int i = 0; i < n; i++) {
        x_buf[i].R = poses[i].q.toRotationMatrix();
        x_buf[i].p = poses[i].t;
    }
    double residual_cur = 0, residual_pre = 0;
    size_t mem_cost = 0;
    iters = 0;
    hess_out.clear();
    for (int loop = 0; loop < prm.max_iter; loop++) {
        iters++;
        unordered_map<VOXEL_LOC, OCTO_TREE_ROOT *> surf_map;
        cut_voxel_parallel(surf_map, pcds, x_buf, prm.voxel_size, n, prm.eigen_ratio * 2, prm.threads);
        recut_parallel(surf_map, prm.threads);
        VOX_HESS voxhess(n);
        for (auto &it : surf_map) it.second->tras_opt(voxhess);
        if (voxhess.plvec_voxels.empty()) {
            for (auto &it : surf_map) delete it.second;
            break;
        }
        VOX_OPTIMIZER opt_lsv(n);
        opt_lsv.remove_outlier(x_buf, voxhess, prm.reject_ratio);
        PLV(6) hess_vec;
        opt_lsv.damping_iter(x_buf, voxhess, residual_cur, hess_vec, mem_cost);
        for (auto &it : surf_map) delete it.second;
        if (loop > 0 && fabs(residual_pre - residual_cur) / fabs(residual_cur) < 0.05 || loop == prm.max_iter - 1) {
            hess_out = hess_vec;
            break;
        }
        residual_pre = residual_cur;
    }
    for (int i = 0; i < n; i++) {
        poses[i].q = Eigen::Quaterniond(x_buf[i].R);
        poses[i].t = x_buf[i].p;
    }
}

gtsam::Pose3 P3(const mypcl::pose &x) { return gtsam::Pose3(gtsam::Rot3(x.q.toRotationMatrix()), gtsam::Point3(x.t)); }

// hba.hpp PGO BetweenFactor: measurement = relative pose of (a, b), variance = |1 / h|
gtsam::NonlinearFactor::shared_ptr between(size_t a, size_t b, const mypcl::pose &xa, const mypcl::pose &xb, const VEC(6) & h) {
    Eigen::Matrix3d Ra = xa.q.toRotationMatrix();
    Eigen::Vector3d t_ab = Ra.transpose() * (xb.t - xa.t);
    Eigen::Matrix3d R_ab = Ra.transpose() * xb.q.toRotationMatrix();
    gtsam::Vector v(6);
    for (int k = 0; k < 6; k++) v[k] = fabs(1.0 / h[k]);
    return gtsam::NonlinearFactor::shared_ptr(new gtsam::BetweenFactor<gtsam::Pose3>(
        a, b, gtsam::Pose3(gtsam::Rot3(R_ab), gtsam::Point3(t_ab)), gtsam::noiseModel::Diagonal::Variances(v)));
}

// Window i of a layer (frames i*GAP .. +WIN_SIZE) merged in the first frame's coordinates with the input poses
// (hba.cpp parallel_comp without the window BA), then 0.05 m downsampled as HBA does.
Cloud::Ptr merge_window(const vector<mypcl::pose> &poses, const vector<Cloud::Ptr> &src, int i) {
    Cloud::Ptr kf(new Cloud);
    const mypcl::pose &x0 = poses[i * GAP];
    for (int j = 0; j < WIN_SIZE; j++) {
        const mypcl::pose &xj = poses[i * GAP + j];
        Eigen::Quaterniond q;
        Eigen::Vector3d t;
        assign_qt(q, t, x0.q.inverse() * xj.q, x0.q.inverse() * (xj.t - x0.t));
        Cloud one;
        mypcl::transform_pointcloud(*src[i * GAP + j], one, t, q);
        kf = mypcl::append_cloud(kf, one);
    }
    downsample_voxel(*kf, 0.05);
    return kf;
}

}  // namespace

struct OnlineHBA::Impl {
    Params prm;
    int stride = 1;  // scans per top keyframe

    // layers[0] = scans; layers[l] = keyframes of layer l (pose = input pose of the window's first frame)
    vector<vector<Cloud::Ptr>> pcds;
    vector<vector<mypcl::pose>> kposes;
    vector<int> next_win, freed;
    struct Imu {
        double stamp;
        Eigen::Vector3d acc, ba, grav;
    };
    vector<Imu> imu;  // per scan

    // stationary chunks (gravity_check.py chunks(): 2 s speed < 2 cm/s, ends trimmed 0.5 s, 4 s chunks, remainder >= 3 s)
    int still_eval = 0;     // next scan whose 2 s speed can be judged (needs scan +20)
    bool run_open = false;
    double next_a = 0;      // start of the next chunk inside the open run

    // gravity factors by scan node (from the file)
    struct GravMeas {
        Eigen::Vector3d b_ref;  // up in the body frame
        double sigma;
        Eigen::Vector3d n_z;    // up in the world frame
    };
    unordered_map<int, GravMeas> grav;  // pending, by scan node (attached when the node is inserted)

    // pose graph
    gtsam::ISAM2 isam;
    gtsam::NonlinearFactorGraph graph;
    gtsam::Values values;
    int nodes = 0;
    gtsam::FactorIndices top_idx;
    int solved = 0;  // top keyframes covered by the last applied round
    std::vector<RoundStat> stats;
    double pending_ms = 0;

    // scan queue and the worker
    std::mutex mu;
    std::condition_variable cv;
    std::deque<std::tuple<mypcl::pose, Cloud::Ptr, Imu>> pending;
    bool file_grav = false;
    bool stop = false;
    double stop_time = 0;
    std::atomic<int> pushed{0};
    std::thread worker;

    static gtsam::Pose3AttitudeFactor gravity_factor(int node, const GravMeas &m) {
        return gtsam::Pose3AttitudeFactor(node, gtsam::Unit3(m.n_z), gtsam::noiseModel::Isotropic::Sigma(2, m.sigma), gtsam::Unit3(m.b_ref));
    }

    // A finished chunk [a, e): the accelerometer mean minus the LIO bias, in the chunk's mean attitude, is the body-frame
    // up; every scan node in the chunk gets it rotated by its own attitude, against the LIO's world up (-grav).
    void emit_chunk(double a, double e) {
        vector<int> idx;
        for (int i = 0; i < (int)imu.size(); i++)
            if (imu[i].stamp >= a && imu[i].stamp < e && imu[i].acc.allFinite()) idx.push_back(i);
        if (idx.empty()) return;
        Eigen::Vector3d m = Eigen::Vector3d::Zero();
        Eigen::Vector4d qsum = Eigen::Vector4d::Zero();
        const vector<mypcl::pose> &lio = kposes[0];
        for (int i : idx) {
            m += imu[i].acc;
            Eigen::Vector4d q = lio[i].q.coeffs();
            if (qsum.dot(q) < 0) q = -q;
            qsum += q;
        }
        m /= idx.size();
        Eigen::Quaterniond qbar(qsum.normalized());
        const Imu &last = imu[idx.back()];
        Eigen::Vector3d up_body = (m - last.ba).normalized(), up_w = (-last.grav).normalized();
        double sigma = prm.gravity_sigma_deg * M_PI / 180 * std::sqrt((double)idx.size());
        for (int i : idx) {
            GravMeas g{lio[i].q.toRotationMatrix().transpose() * (qbar * up_body), sigma, up_w};
            if (i < nodes) graph.add(gravity_factor(i, g));
            else grav[i] = g;
        }
        if (graph.size()) {
            isam.update(graph, values);
            graph.resize(0);
            values.clear();
        }
    }

    // Causal version of gravity_check.py chunks(): scan i is "still" when the 2 s motion from it is < 2 cm/s (known once
    // scan i+20 exists). Inside a still run, trimmed 0.5 s at both ends, a chunk is emitted every 4 s; when the run ends
    // a remainder of >= 3 s is one more chunk.
    void detect_still() {
        const double V_STILL = 0.02, CHUNK = 4.0, MIN_LEN = 3.0;
        const vector<mypcl::pose> &lio = kposes[0];
        int N = imu.size();
        while (still_eval + 20 < N) {
            int i = still_eval++;
            double t_end = imu[i + 20].stamp;
            bool still = (lio[i + 20].t - lio[i].t).norm() / (t_end - imu[i].stamp) < V_STILL;
            if (still && !run_open) {
                run_open = true;
                next_a = imu[i].stamp + 0.5;
            }
            if (still) {
                while (t_end - 0.5 >= next_a + CHUNK) {
                    emit_chunk(next_a, next_a + CHUNK);
                    next_a += CHUNK;
                }
            } else if (run_open) {
                double e = imu[i - 1 + 20].stamp - 0.5;
                if (e - next_a >= MIN_LEN) emit_chunk(next_a, e);
                run_open = false;
            }
        }
    }

    void add_nodes(int upto) {
        const vector<mypcl::pose> &lio = kposes[0];
        for (; nodes < upto; nodes++) {
            values.insert(nodes, P3(lio[nodes]));
            if (nodes == 0) {
                gtsam::Vector v(6);
                v << 1e-6, 1e-6, 1e-6, 1e-8, 1e-8, 1e-8;
                graph.add(gtsam::PriorFactor<gtsam::Pose3>(0, P3(lio[0]), gtsam::noiseModel::Diagonal::Variances(v)));
            }
            auto g = grav.find(nodes);
            if (g != grav.end()) {
                graph.add(gravity_factor(nodes, g->second));
                grav.erase(g);
            }
        }
    }

    // Layer-1 window i: every scan pair, measurement = LIO relative pose, constant weight
    void window_factors(int i) {
        add_nodes(i * GAP + WIN_SIZE);
        VEC(6) h;
        for (int k = 0; k < 6; k++) h[k] = prm.hess_const[k];
        const vector<mypcl::pose> &lio = kposes[0];
        for (int j = 0; j < WIN_SIZE - 1; j++)
            for (int k = j + 1; k < WIN_SIZE; k++) graph.push_back(between(i * GAP + j, i * GAP + k, lio[i * GAP + j], lio[i * GAP + k], h));
        isam.update(graph, values);
        graph.resize(0);
        values.clear();
    }

    // Every full window whose frames have arrived, layer by layer; frames no window still needs are freed.
    void advance() {
        int L = prm.layers;
        for (int l = 0; l + 1 < L; l++) {
            while ((int)pcds[l].size() >= next_win[l] * GAP + WIN_SIZE) {
                int i = next_win[l]++;
                Cloud::Ptr kf = merge_window(kposes[l], pcds[l], i);
                if (l + 1 == L - 1 && prm.downsample_size > 0) downsample_voxel(*kf, prm.downsample_size);
                pcds[l + 1].push_back(kf);
                kposes[l + 1].push_back(kposes[l][i * GAP]);
                if (l == 0) window_factors(i);
            }
            for (int j = freed[l]; j < next_win[l] * GAP; j++, freed[l]++) pcds[l][j].reset();
        }
    }

    // One BA round over every top keyframe so far, from the LIO poses; the all-pair factors replace the previous ones.
    void round() {
        const vector<mypcl::pose> &top = kposes.back();
        int n = top.size(), start = pushed.load();
        vector<mypcl::pose> x(top.begin(), top.begin() + n);
        PLV(6) hess;
        int iters;
        double t0 = now_sec();
        global_ba(prm, x, pcds.back(), hess, iters);
        double ba_ms = (now_sec() - t0) * 1e3;
        solved = n;
        bool discard;
        {
            std::lock_guard<std::mutex> lk(mu);
            discard = stop;
            if (discard) pending_ms = (now_sec() - stop_time) * 1e3;
        }
        if (discard) {
            stats.push_back({(int)stats.size() + 1, start, n, iters, ba_ms, 0, -1});
            return;
        }
        double t1 = now_sec();
        gtsam::NonlinearFactorGraph g;
        if (hess.size() == (size_t)n * (n - 1) / 2) {
            int c = 0;
            for (int i = 0; i < n - 1; i++)
                for (int j = i + 1; j < n; j++, c++)
                    if (hess[c].norm() >= 1e-20) g.push_back(between((size_t)i * stride, (size_t)j * stride, x[i], x[j], hess[c]));
        }
        isam.update(gtsam::NonlinearFactorGraph(), gtsam::Values(), top_idx);
        top_idx = isam.update(g).newFactorsIndices;
        stats.push_back({(int)stats.size() + 1, start, n, iters, ba_ms, (now_sec() - t1) * 1e3, pushed.load()});
    }

    void run() {
        for (;;) {
            std::deque<std::tuple<mypcl::pose, Cloud::Ptr, Imu>> batch;
            bool stopping;
            {
                std::unique_lock<std::mutex> lk(mu);
                cv.wait(lk, [&] { return stop || !pending.empty(); });
                if (pending.empty()) return;
                batch.swap(pending);
                stopping = stop;
            }
            for (auto &e : batch) {
                kposes[0].push_back(std::get<0>(e));
                pcds[0].push_back(std::get<1>(e));
                imu.push_back(std::get<2>(e));
            }
            advance();
            if (prm.gravity_sigma_deg > 0 && !file_grav) detect_still();
            if (!stopping && (int)kposes.back().size() >= solved + prm.every) round();
        }
    }
};

OnlineHBA::OnlineHBA(const Params &params) : impl_(new Impl) {
    Impl &im = *impl_;
    im.prm = params;
    if (params.layers < 2) throw std::invalid_argument("OnlineHBA: layers must be >= 2");
    if (params.hess_const.size() != 6) throw std::invalid_argument("OnlineHBA: hess_const needs 6 values");
    for (int l = 1; l < params.layers; l++) im.stride *= GAP;
    im.pcds.resize(params.layers);
    im.kposes.resize(params.layers);
    im.next_win.assign(params.layers, 0);
    im.freed.assign(params.layers, 0);
    if (!params.gravity_file.empty()) {
        std::ifstream gf(params.gravity_file);
        double nx, ny, nz;
        if (!(gf >> nx >> ny >> nz)) throw std::invalid_argument("OnlineHBA: cannot read " + params.gravity_file);
        im.file_grav = true;
        Eigen::Vector3d up_w(nx, ny, nz);
        int node;
        double bx, by, bz, sig;
        while (gf >> node >> bx >> by >> bz >> sig) im.grav[node] = {Eigen::Vector3d(bx, by, bz), sig, up_w};
    }
    gtsam::ISAM2Params ip;
    ip.relinearizeThreshold = 0.01;
    ip.relinearizeSkip = 1;
    im.isam = gtsam::ISAM2(ip);
    thd_num = params.threads;
    im.worker = std::thread(&Impl::run, impl_.get());
}

OnlineHBA::~OnlineHBA() {
    if (impl_->worker.joinable()) {
        {
            std::lock_guard<std::mutex> lk(impl_->mu);
            impl_->stop = true;
            impl_->stop_time = now_sec();
        }
        impl_->cv.notify_all();
        impl_->worker.join();
    }
}

void OnlineHBA::push(const Eigen::Vector4d &q_xyzw, const Eigen::Vector3d &p, const float *xyz, int n, double stamp,
                     const Eigen::Vector3d &acc, const Eigen::Vector3d &ba, const Eigen::Vector3d &grav) {
    Cloud::Ptr pc(new Cloud);
    pc->points.resize(n);
    for (int i = 0; i < n; i++) {
        pc->points[i].x = xyz[3 * i];
        pc->points[i].y = xyz[3 * i + 1];
        pc->points[i].z = xyz[3 * i + 2];
    }
    {
        std::lock_guard<std::mutex> lk(impl_->mu);
        impl_->pending.emplace_back(mypcl::pose(Eigen::Quaterniond(q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]), p), pc, Impl::Imu{stamp, acc, ba, grav});
    }
    impl_->pushed++;
    impl_->cv.notify_one();
}

std::vector<Pose> OnlineHBA::finish() {
    Impl &im = *impl_;
    {
        std::lock_guard<std::mutex> lk(im.mu);
        im.stop = true;
        im.stop_time = now_sec();
    }
    im.cv.notify_all();
    im.worker.join();  // drains queued scans (windows, odometry factors); a round in flight ends and is discarded

    const vector<mypcl::pose> &lio = im.kposes[0];
    int N = lio.size();
    std::vector<Pose> out(N);
    gtsam::Values est = im.isam.calculateEstimate();
    for (int i = 0; i < im.nodes; i++) {
        gtsam::Pose3 p = est.at<gtsam::Pose3>(i);
        out[i] = {p.rotation().matrix(), p.translation()};
    }
    for (int i = im.nodes; i < N; i++) {  // scans past the last odometry window: LIO relative motion on the last node
        if (i == 0) {
            out[i] = {lio[0].q.toRotationMatrix(), lio[0].t};
            continue;
        }
        Eigen::Matrix3d Ra = lio[i - 1].q.toRotationMatrix();
        out[i].R = out[i - 1].R * (Ra.transpose() * lio[i].q.toRotationMatrix());
        out[i].p = out[i - 1].p + out[i - 1].R * (Ra.transpose() * (lio[i].t - lio[i - 1].t));
    }
    return out;
}

const std::vector<RoundStat> &OnlineHBA::stats() const { return impl_->stats; }
double OnlineHBA::pending_ms() const { return impl_->pending_ms; }

}  // namespace hba
}  // namespace se3_lio
