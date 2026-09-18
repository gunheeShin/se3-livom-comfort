// SPDX-License-Identifier: GPL-2.0-only
//
// Adapted from HKU-MARS VoxelMap (https://github.com/hku-mars/VoxelMap),
// Copyright (C) the VoxelMap authors, licensed under GPL-2.0.
// This file inherits the GPL-2.0 license of its origin.

#include "voxel_map_util.h"

#include <cstdlib>

// env SE3_LIO_OMP_THREADS overrides the MP_PROC_NUM default (benchmark sets 1
// for deterministic single-threaded runs).
static int se3_lio_omp_threads() {
    const char *e = std::getenv("SE3_LIO_OMP_THREADS");
    return (e && std::atoi(e) > 0) ? std::atoi(e) : MP_PROC_NUM;
}

void buildVoxelMap(const std::vector<pointWithCov> &input_points,
                   const float voxel_size,
                   const int max_layer,
                   const std::vector<int> &layer_point_size,
                   const int max_points_size,
                   const int max_cov_points_size,
                   const float planer_threshold,
                   std::unordered_map<VOXEL_LOC, OctoTree *> &feat_map) {
    uint plsize = input_points.size();
    for (uint i = 0; i < plsize; i++) {
        const pointWithCov p_v = input_points[i];
        float loc_xyz[3];
        for (int j = 0; j < 3; j++) {
            loc_xyz[j] = p_v.point[j] / voxel_size;
            if (loc_xyz[j] < 0) {
                loc_xyz[j] -= 1.0;
            }
        }

        Eigen::Vector3i loc_xyz_int = (p_v.point / voxel_size).array().round().template cast<int>();

        VOXEL_LOC position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);

        auto iter = feat_map.find(position);
        if (iter != feat_map.end()) {
            feat_map[position]->temp_points_.push_back(p_v);
            feat_map[position]->new_points_num_++;
        } else {
            OctoTree *octo_tree = new OctoTree(max_layer, 0, layer_point_size, max_points_size,
                                               max_cov_points_size, planer_threshold);
            feat_map[position] = octo_tree;
            feat_map[position]->quater_length_ = voxel_size / 4;
            feat_map[position]->voxel_center_[0] = (0.5 + position.x) * voxel_size;
            feat_map[position]->voxel_center_[1] = (0.5 + position.y) * voxel_size;
            feat_map[position]->voxel_center_[2] = (0.5 + position.z) * voxel_size;
            feat_map[position]->temp_points_.push_back(p_v);
            feat_map[position]->new_points_num_++;
            feat_map[position]->layer_point_size_ = layer_point_size;
        }
    }

    for (auto iter = feat_map.begin(); iter != feat_map.end(); ++iter) {
        iter->second->init_octo_tree();
    }
}

void updateVoxelMapOMP(const std::vector<pointWithCov> &input_points,
                       const float voxel_size,
                       const int max_layer,
                       const std::vector<int> &layer_point_size,
                       const int max_points_size,
                       const int max_cov_points_size,
                       const float planer_threshold,
                       std::unordered_map<VOXEL_LOC, OctoTree *> &feat_map) {
    std::unordered_map<VOXEL_LOC, std::vector<pointWithCov>> position_index_map;
    int insert_count = 0, update_count = 0;
    uint plsize = input_points.size();

    // double t_update_start = omp_get_wtime();
    for (uint i = 0; i < plsize; i++) {
        const pointWithCov p_v = input_points[i];
        float loc_xyz[3];
        for (int j = 0; j < 3; j++) {
            loc_xyz[j] = p_v.point[j] / voxel_size;
            if (loc_xyz[j] < 0) {
                loc_xyz[j] -= 1.0;
            }
        }
        VOXEL_LOC position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
        auto iter = feat_map.find(position);
        if (iter != feat_map.end()) {
            // update_count++;
            position_index_map[position].push_back(p_v);
        } else {
            insert_count++;
            OctoTree *octo_tree = new OctoTree(max_layer, 0, layer_point_size, max_points_size,
                                               max_cov_points_size, planer_threshold);
            feat_map[position] = octo_tree;
            feat_map[position]->quater_length_ = voxel_size / 4;
            feat_map[position]->voxel_center_[0] = (0.5 + position.x) * voxel_size;
            feat_map[position]->voxel_center_[1] = (0.5 + position.y) * voxel_size;
            feat_map[position]->voxel_center_[2] = (0.5 + position.z) * voxel_size;
            feat_map[position]->UpdateOctoTree(p_v);
        }
    }
    // #ifdef MP_EN
    omp_set_num_threads(se3_lio_omp_threads());
#pragma omp parallel for default(none) shared(position_index_map, feat_map)
    // #endif
    for (size_t b = 0; b < position_index_map.bucket_count(); b++) {
        for (auto bi = position_index_map.begin(b); bi != position_index_map.end(b); bi++) {
            VOXEL_LOC position = bi->first;
            for (const pointWithCov &p_v : bi->second) {
                feat_map[position]->UpdateOctoTree(p_v);
            }
        }
    }
    // t_update_end = omp_get_wtime();
    // std::printf("Update:  %.4fs\n", t_update_end - t_update_start);

    // std::printf("Insert: %d  Update: %d \n", insert_count, update_count);
}

void build_single_residual(const pointWithCov &pv,
                           const OctoTree *current_octo,
                           const int current_layer,
                           const int max_layer,
                           const double sigma_num,
                           bool &is_sucess,
                           double &prob,
                           ptpl &single_ptpl) {
    double radius_k = 3;
    Eigen::Vector3d p_w = pv.point_world;
    if (current_octo->plane_ptr_->is_plane) {
        Plane &plane = *current_octo->plane_ptr_;
        float dis_to_plane = fabs(plane.normal(0) * p_w(0) + plane.normal(1) * p_w(1) +
                                  plane.normal(2) * p_w(2) + plane.d);
        float dis_to_center = (plane.center(0) - p_w(0)) * (plane.center(0) - p_w(0)) +
                              (plane.center(1) - p_w(1)) * (plane.center(1) - p_w(1)) +
                              (plane.center(2) - p_w(2)) * (plane.center(2) - p_w(2));
        float range_dis = sqrt(dis_to_center - dis_to_plane * dis_to_plane);

        if (range_dis <= radius_k * plane.radius) {
            Eigen::Matrix<double, 1, 6> J_nq;
            J_nq.block<1, 3>(0, 0) = p_w - plane.center;
            J_nq.block<1, 3>(0, 3) = -plane.normal;
            double sigma_l = J_nq * plane.plane_cov * J_nq.transpose();
            sigma_l += plane.normal.transpose() * pv.cov * plane.normal;  // eq(11)
            if (dis_to_plane < sigma_num * sqrt(sigma_l)) {
                is_sucess = true;
                double this_prob =
                    1.0 / (sqrt(sigma_l)) * exp(-0.5 * dis_to_plane * dis_to_plane / sigma_l);
                if (this_prob > prob) {
                    prob = this_prob;
                    single_ptpl.point = pv.point;
                    single_ptpl.point_world = pv.point_world;
                    single_ptpl.plane_cov = plane.plane_cov;
                    single_ptpl.normal = plane.normal;
                    single_ptpl.center = plane.center;
                    single_ptpl.d = plane.d;
                    single_ptpl.layer = current_layer;
                    single_ptpl.cov_lidar = pv.cov_lidar;
                }
                return;
            } else {
                // is_sucess = false;
                return;
            }
        } else {
            // is_sucess = false;
            return;
        }
    } else {
        if (current_layer < max_layer) {
            for (size_t leafnum = 0; leafnum < 8; leafnum++) {
                if (current_octo->leaves_[leafnum] != nullptr) {
                    OctoTree *leaf_octo = current_octo->leaves_[leafnum];
                    build_single_residual(pv, leaf_octo, current_layer + 1, max_layer, sigma_num,
                                          is_sucess, prob, single_ptpl);
                }
            }
            return;
        } else {
            is_sucess = false;
            return;
        }
    }
}

void BuildResidualListOMP(const std::unordered_map<VOXEL_LOC, OctoTree *> &voxel_map,
                          const double voxel_size,
                          const double sigma_num,
                          const int max_layer,
                          const std::vector<pointWithCov> &pv_list,
                          std::vector<ptpl> &ptpl_list,
                          std::vector<Eigen::Vector3d> &non_match) {
    std::mutex mylock;
    ptpl_list.clear();
    std::vector<ptpl> all_ptpl_list(pv_list.size());
    std::vector<bool> useful_ptpl(pv_list.size());
    std::vector<size_t> index(pv_list.size());
    for (size_t i = 0; i < index.size(); ++i) {
        index[i] = i;
        useful_ptpl[i] = false;
    }
    // #ifdef MP_EN
    int num_success = 0;
    // #endif
    omp_set_num_threads(se3_lio_omp_threads());
#pragma omp parallel for
    for (int i = 0; i < index.size(); i++) {
        pointWithCov pv = pv_list[i];
        float loc_xyz[3];
        for (int j = 0; j < 3; j++) {
            loc_xyz[j] = pv.point_world[j] / voxel_size;
            if (loc_xyz[j] < 0) {
                loc_xyz[j] -= 1.0;
            }
        }
        VOXEL_LOC position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
        auto iter = voxel_map.find(position);

        if (iter != voxel_map.end()) {
            OctoTree *current_octo = iter->second;
            ptpl single_ptpl;
            bool is_sucess = false;
            double prob = 0;
            build_single_residual(pv, current_octo, 0, max_layer, sigma_num, is_sucess, prob,
                                  single_ptpl);

            if (!is_sucess) {
                VOXEL_LOC near_position = position;
                if (loc_xyz[0] > (current_octo->voxel_center_[0] + current_octo->quater_length_)) {
                    near_position.x = near_position.x + 1;
                } else if (loc_xyz[0] <
                           (current_octo->voxel_center_[0] - current_octo->quater_length_)) {
                    near_position.x = near_position.x - 1;
                }
                if (loc_xyz[1] > (current_octo->voxel_center_[1] + current_octo->quater_length_)) {
                    near_position.y = near_position.y + 1;
                } else if (loc_xyz[1] <
                           (current_octo->voxel_center_[1] - current_octo->quater_length_)) {
                    near_position.y = near_position.y - 1;
                }
                if (loc_xyz[2] > (current_octo->voxel_center_[2] + current_octo->quater_length_)) {
                    near_position.z = near_position.z + 1;
                } else if (loc_xyz[2] <
                           (current_octo->voxel_center_[2] - current_octo->quater_length_)) {
                    near_position.z = near_position.z - 1;
                }
                auto iter_near = voxel_map.find(near_position);
                if (iter_near != voxel_map.end()) {
                    build_single_residual(pv, iter_near->second, 0, max_layer, sigma_num, is_sucess,
                                          prob, single_ptpl);
                }
            }

            if (is_sucess) {
                mylock.lock();
                useful_ptpl[i] = true;
                all_ptpl_list[i] = std::move(single_ptpl);
                mylock.unlock();
            } else {
                mylock.lock();
                useful_ptpl[i] = false;
                mylock.unlock();
            }
        }
    }
    for (size_t i = 0; i < useful_ptpl.size(); i++) {
        if (useful_ptpl[i]) {
            all_ptpl_list[i].original_index = i;
            ptpl_list.push_back(all_ptpl_list[i]);
        }
    }
}

void GetUpdatePlane(const OctoTree *current_octo,
                    const int pub_max_voxel_layer,
                    std::vector<Plane> &plane_list) {
    if (current_octo->layer_ > pub_max_voxel_layer) {
        return;
    }
    if (current_octo->plane_ptr_->is_update) {
        plane_list.push_back(*current_octo->plane_ptr_);
    }
    if (current_octo->layer_ < current_octo->max_layer_) {
        if (!current_octo->plane_ptr_->is_plane) {
            for (size_t i = 0; i < 8; i++) {
                if (current_octo->leaves_[i] != nullptr) {
                    GetUpdatePlane(current_octo->leaves_[i], pub_max_voxel_layer, plane_list);
                }
            }
        }
    }
    return;
}