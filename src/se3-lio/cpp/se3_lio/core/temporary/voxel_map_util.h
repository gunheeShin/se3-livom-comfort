// SPDX-License-Identifier: GPL-2.0-only
//
// Adapted from HKU-MARS VoxelMap (https://github.com/hku-mars/VoxelMap),
// Copyright (C) the VoxelMap authors, licensed under GPL-2.0.
// This file inherits the GPL-2.0 license of its origin.

#ifndef VOXEL_MAP_UTIL_HPP
#define VOXEL_MAP_UTIL_HPP
// #include "common_lib.hpp"
#include <stdio.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "omp.h"

// #include <boost/filesystem.hpp>
#include <filesystem>

#include "common/data_type.h"

#define MP_EN TRUE
#define MP_PROC_NUM 8
#define HASH_P 116101
#define MAX_N 10000000000

static int plane_id = 0;

// a point to plane matching structure
typedef struct ptpl {
    Eigen::Vector3d point;
    Eigen::Vector3d point_world;
    Eigen::Vector3d normal;
    Eigen::Vector3d center;
    Eigen::Matrix<double, 6, 6> plane_cov;
    double d;
    int layer;
    Eigen::Matrix3d cov_lidar;
    int original_index;
} ptpl;

// 3D point with covariance
typedef struct pointWithCov {
    Eigen::Vector3d point;
    Eigen::Vector3d point_world;
    Eigen::Matrix3d cov;
    Eigen::Matrix3d cov_lidar;

    pointWithCov() {}
    pointWithCov(const Eigen::Vector3d &_point,
                 const Eigen::Vector3d &_point_world,
                 const Eigen::Matrix3d &_cov_lidar,
                 const Eigen::Matrix3d &_cov)
        : point(_point), point_world(_point_world), cov(_cov), cov_lidar(_cov_lidar) {}
} pointWithCov;

// Stored map point. The octree only ever reads .point (world coords, filled by
// transformGlobalPointsWithNoise) and .cov (plane_cov in init_plane);
// point_world/cov_lidar of pointWithCov are never set on the map path, so
// keeping them costs 96 B/point of dead weight.
struct MapPoint {
    Eigen::Vector3d point;
    Eigen::Matrix3d cov;

    MapPoint() {}
    MapPoint(const pointWithCov &pv) : point(pv.point), cov(pv.cov) {}
};

typedef struct Plane {
    Eigen::Vector3d center;
    Eigen::Vector3d normal;
    Eigen::Vector3d y_normal;
    Eigen::Vector3d x_normal;
    Eigen::Matrix3d covariance;
    Eigen::Matrix<double, 6, 6> plane_cov;
    float radius = 0;
    float min_eigen_value = 1;
    float mid_eigen_value = 1;
    float max_eigen_value = 1;
    float d = 0;
    int points_size = 0;

    bool is_plane = false;
    bool is_init = false;
    int id;
    // is_update and last_update_points_size are only for publish plane
    bool is_update = false;
    int last_update_points_size = 0;
    bool update_enable = true;

    // for debug by gunhee
    bool pass_nns = false;

} Plane;

class VOXEL_LOC {
public:
    int64_t x, y, z;

    VOXEL_LOC(int64_t vx = 0, int64_t vy = 0, int64_t vz = 0) : x(vx), y(vy), z(vz) {}

    bool operator==(const VOXEL_LOC &other) const {
        return (x == other.x && y == other.y && z == other.z);
    }

    int64_t operator()(const VOXEL_LOC &s) const {
        // using std::hash;
        // using std::size_t;
        return ((((s.z) * HASH_P) % MAX_N + (s.y)) * HASH_P) % MAX_N + (s.x);
    }
};

// Hash value
namespace std {
template <>
struct hash<VOXEL_LOC> {
    int64_t operator()(const VOXEL_LOC &s) const {
        using std::hash;
        using std::size_t;
        return ((((s.z) * HASH_P) % MAX_N + (s.y)) * HASH_P) % MAX_N + (s.x);
    }
};

}  // namespace std

class OctoTree {
public:
    std::vector<MapPoint> temp_points_;  // all points in an octo tree
    std::vector<MapPoint> new_points_;   // new points in an octo tree
    Plane *plane_ptr_;
    int max_layer_;
    bool indoor_mode_;
    int layer_;
    int octo_state_;  // 0 is end of tree, 1 is not
    OctoTree *leaves_[8];
    double voxel_center_[3];  // x, y, z
    std::vector<int> layer_point_size_;
    float quater_length_;
    float planer_threshold_;
    int max_plane_update_threshold_;
    int update_size_threshold_;
    int all_points_num_;
    int new_points_num_;
    int max_points_size_;
    int max_cov_points_size_;
    bool init_octo_;
    bool update_cov_enable_;
    bool update_enable_;
    OctoTree(int max_layer,
             int layer,
             std::vector<int> layer_point_size,
             int max_point_size,
             int max_cov_points_size,
             float planer_threshold)
        : max_layer_(max_layer),
          layer_(layer),
          layer_point_size_(layer_point_size),
          max_points_size_(max_point_size),
          max_cov_points_size_(max_cov_points_size),
          planer_threshold_(planer_threshold) {
        temp_points_.clear();
        octo_state_ = 0;
        new_points_num_ = 0;
        all_points_num_ = 0;
        // when new points num > 5, do a update
        update_size_threshold_ = 5;
        init_octo_ = false;
        update_enable_ = true;
        update_cov_enable_ = true;
        max_plane_update_threshold_ = layer_point_size_[layer_];  // 이 수를 넘으면 plane update
        for (int i = 0; i < 8; i++) {
            leaves_[i] = nullptr;
        }
        plane_ptr_ = new Plane;
    }

    ~OctoTree() {
        for (int i = 0; i < 8; i++) {
            delete leaves_[i];
        }
        delete plane_ptr_;
    }

    // check is plane , calc plane parameters including plane covariance
    void init_plane(const std::vector<MapPoint> &points, Plane *plane) {
        plane->plane_cov = Eigen::Matrix<double, 6, 6>::Zero();
        plane->covariance = Eigen::Matrix3d::Zero();
        plane->center = Eigen::Vector3d::Zero();
        plane->normal = Eigen::Vector3d::Zero();
        plane->points_size = points.size();
        plane->radius = 0;

        for (auto pv : points) {
            plane->covariance += pv.point * pv.point.transpose();
            plane->center += pv.point;
        }

        plane->center = plane->center / plane->points_size;
        plane->covariance =
            plane->covariance / plane->points_size - plane->center * plane->center.transpose();

        Eigen::EigenSolver<Eigen::Matrix3d> es(plane->covariance);
        Eigen::Matrix3cd evecs = es.eigenvectors();
        Eigen::Vector3cd evals = es.eigenvalues();
        Eigen::Vector3d evalsReal;
        evalsReal = evals.real();
        Eigen::Matrix3f::Index evalsMin, evalsMax;
        evalsReal.rowwise().sum().minCoeff(&evalsMin);
        evalsReal.rowwise().sum().maxCoeff(&evalsMax);
        int evalsMid = 3 - evalsMin - evalsMax;
        Eigen::Vector3d evecMin = evecs.real().col(evalsMin);
        Eigen::Vector3d evecMid = evecs.real().col(evalsMid);
        Eigen::Vector3d evecMax = evecs.real().col(evalsMax);

        // plane covariance calculation
        Eigen::Matrix3d J_Q = 1.0 / plane->points_size * Eigen::Matrix3d::Identity();
        if (evalsReal(evalsMin) < planer_threshold_) {
            std::vector<int> index(points.size());
            std::vector<Eigen::Matrix<double, 6, 6>> temp_matrix(points.size());
            for (int i = 0; i < points.size(); i++) {
                Eigen::Matrix<double, 6, 3> J;
                Eigen::Matrix3d F;
                for (int m = 0; m < 3; m++) {
                    if (m != (int)evalsMin) {
                        Eigen::Matrix<double, 1, 3> F_m =
                            (points[i].point - plane->center).transpose() /
                            ((plane->points_size) * (evalsReal[evalsMin] - evalsReal[m])) *
                            (evecs.real().col(m) * evecs.real().col(evalsMin).transpose() +
                             evecs.real().col(evalsMin) * evecs.real().col(m).transpose());
                        F.row(m) = F_m;
                    } else {
                        Eigen::Matrix<double, 1, 3> F_m;
                        F_m << 0, 0, 0;
                        F.row(m) = F_m;
                    }
                }
                J.block<3, 3>(0, 0) = evecs.real() * F;
                J.block<3, 3>(3, 0) = J_Q;
                plane->plane_cov += J * points[i].cov * J.transpose();
            }

            plane->normal << evecs.real()(0, evalsMin), evecs.real()(1, evalsMin),
                evecs.real()(2, evalsMin);
            plane->y_normal << evecs.real()(0, evalsMid), evecs.real()(1, evalsMid),
                evecs.real()(2, evalsMid);
            plane->x_normal << evecs.real()(0, evalsMax), evecs.real()(1, evalsMax),
                evecs.real()(2, evalsMax);
            plane->min_eigen_value = evalsReal(evalsMin);
            plane->mid_eigen_value = evalsReal(evalsMid);
            plane->max_eigen_value = evalsReal(evalsMax);
            plane->radius = sqrt(evalsReal(evalsMax));
            plane->d = -(plane->normal(0) * plane->center(0) + plane->normal(1) * plane->center(1) +
                         plane->normal(2) * plane->center(2));
            plane->is_plane = true;
            if (plane->last_update_points_size == 0) {
                plane->last_update_points_size = plane->points_size;
                plane->is_update = true;
            } else if (plane->points_size - plane->last_update_points_size > 100) {
                plane->last_update_points_size = plane->points_size;
                plane->is_update = true;
            }

            if (!plane->is_init) {
                plane->id = plane_id;
                plane_id++;
                plane->is_init = true;
            }

        } else {
            if (!plane->is_init) {
                plane->id = plane_id;
                plane_id++;
                plane->is_init = true;
            }
            if (plane->last_update_points_size == 0) {
                plane->last_update_points_size = plane->points_size;
                plane->is_update = true;
            } else if (plane->points_size - plane->last_update_points_size > 100) {
                plane->last_update_points_size = plane->points_size;
                plane->is_update = true;
            }
            plane->is_plane = false;
            plane->normal << evecs.real()(0, evalsMin), evecs.real()(1, evalsMin),
                evecs.real()(2, evalsMin);
            plane->y_normal << evecs.real()(0, evalsMid), evecs.real()(1, evalsMid),
                evecs.real()(2, evalsMid);
            plane->x_normal << evecs.real()(0, evalsMax), evecs.real()(1, evalsMax),
                evecs.real()(2, evalsMax);
            plane->min_eigen_value = evalsReal(evalsMin);
            plane->mid_eigen_value = evalsReal(evalsMid);
            plane->max_eigen_value = evalsReal(evalsMax);
            plane->radius = sqrt(evalsReal(evalsMax));
            plane->d = -(plane->normal(0) * plane->center(0) + plane->normal(1) * plane->center(1) +
                         plane->normal(2) * plane->center(2));
        }
    }

    // only updaye plane normal, center and radius with new points
    void update_plane(const std::vector<MapPoint> &points, Plane *plane) {
        Eigen::Matrix3d old_covariance = plane->covariance;
        Eigen::Vector3d old_center = plane->center;
        Eigen::Matrix3d sum_ppt =
            (plane->covariance + plane->center * plane->center.transpose()) * plane->points_size;
        Eigen::Vector3d sum_p = plane->center * plane->points_size;
        for (size_t i = 0; i < points.size(); i++) {
            Eigen::Vector3d pv = points[i].point;
            sum_ppt += pv * pv.transpose();
            sum_p += pv;
        }
        plane->points_size = plane->points_size + points.size();
        plane->center = sum_p / plane->points_size;
        plane->covariance =
            sum_ppt / plane->points_size - plane->center * plane->center.transpose();
        Eigen::EigenSolver<Eigen::Matrix3d> es(plane->covariance);
        Eigen::Matrix3cd evecs = es.eigenvectors();
        Eigen::Vector3cd evals = es.eigenvalues();
        Eigen::Vector3d evalsReal;
        evalsReal = evals.real();
        Eigen::Matrix3d::Index evalsMin, evalsMax;
        evalsReal.rowwise().sum().minCoeff(&evalsMin);
        evalsReal.rowwise().sum().maxCoeff(&evalsMax);
        int evalsMid = 3 - evalsMin - evalsMax;
        Eigen::Vector3d evecMin = evecs.real().col(evalsMin);
        Eigen::Vector3d evecMid = evecs.real().col(evalsMid);
        Eigen::Vector3d evecMax = evecs.real().col(evalsMax);
        if (evalsReal(evalsMin) < planer_threshold_) {
            plane->normal << evecs.real()(0, evalsMin), evecs.real()(1, evalsMin),
                evecs.real()(2, evalsMin);
            plane->y_normal << evecs.real()(0, evalsMid), evecs.real()(1, evalsMid),
                evecs.real()(2, evalsMid);
            plane->x_normal << evecs.real()(0, evalsMax), evecs.real()(1, evalsMax),
                evecs.real()(2, evalsMax);
            plane->min_eigen_value = evalsReal(evalsMin);
            plane->mid_eigen_value = evalsReal(evalsMid);
            plane->max_eigen_value = evalsReal(evalsMax);
            plane->radius = sqrt(evalsReal(evalsMax));
            plane->d = -(plane->normal(0) * plane->center(0) + plane->normal(1) * plane->center(1) +
                         plane->normal(2) * plane->center(2));

            plane->is_plane = true;
            plane->is_update = true;

        } else {
            plane->normal << evecs.real()(0, evalsMin), evecs.real()(1, evalsMin),
                evecs.real()(2, evalsMin);
            plane->y_normal << evecs.real()(0, evalsMid), evecs.real()(1, evalsMid),
                evecs.real()(2, evalsMid);
            plane->x_normal << evecs.real()(0, evalsMax), evecs.real()(1, evalsMax),
                evecs.real()(2, evalsMax);
            plane->min_eigen_value = evalsReal(evalsMin);
            plane->mid_eigen_value = evalsReal(evalsMid);
            plane->max_eigen_value = evalsReal(evalsMax);
            plane->radius = sqrt(evalsReal(evalsMax));
            plane->d = -(plane->normal(0) * plane->center(0) + plane->normal(1) * plane->center(1) +
                         plane->normal(2) * plane->center(2));
            plane->is_plane = false;
            plane->is_update = true;
        }
    }

    void init_octo_tree() {
        if (temp_points_.size() > max_plane_update_threshold_) {
            init_plane(temp_points_, plane_ptr_);
            if (plane_ptr_->is_plane == true) {
                octo_state_ = 0;
                if (temp_points_.size() > max_cov_points_size_) {
                    update_cov_enable_ = false;
                }
                if (temp_points_.size() > max_points_size_) {
                    update_enable_ = false;
                }
            } else {
                octo_state_ = 1;
                cut_octo_tree();
            }
            init_octo_ = true;
            new_points_num_ = 0;
            //      temp_points_.clear();
        }
    }

    void cut_octo_tree() {
        if (layer_ >= max_layer_) {
            octo_state_ = 0;
            return;
        }
        for (size_t i = 0; i < temp_points_.size(); i++) {
            int xyz[3] = {0, 0, 0};
            if (temp_points_[i].point[0] > voxel_center_[0]) {
                xyz[0] = 1;
            }
            if (temp_points_[i].point[1] > voxel_center_[1]) {
                xyz[1] = 1;
            }
            if (temp_points_[i].point[2] > voxel_center_[2]) {
                xyz[2] = 1;
            }
            int leafnum = 4 * xyz[0] + 2 * xyz[1] + xyz[2];
            if (leaves_[leafnum] == nullptr) {
                leaves_[leafnum] =
                    new OctoTree(max_layer_, layer_ + 1, layer_point_size_, max_points_size_,
                                 max_cov_points_size_, planer_threshold_);
                leaves_[leafnum]->voxel_center_[0] =
                    voxel_center_[0] + (2 * xyz[0] - 1) * quater_length_;
                leaves_[leafnum]->voxel_center_[1] =
                    voxel_center_[1] + (2 * xyz[1] - 1) * quater_length_;
                leaves_[leafnum]->voxel_center_[2] =
                    voxel_center_[2] + (2 * xyz[2] - 1) * quater_length_;
                leaves_[leafnum]->quater_length_ = quater_length_ / 2;
            }
            leaves_[leafnum]->temp_points_.push_back(temp_points_[i]);
            leaves_[leafnum]->new_points_num_++;
        }
        std::vector<MapPoint>().swap(temp_points_);  // handed down to the leaves

        for (uint i = 0; i < 8; i++) {
            if (leaves_[i] != nullptr) {
                if (leaves_[i]->temp_points_.size() > leaves_[i]->max_plane_update_threshold_) {
                    init_plane(leaves_[i]->temp_points_, leaves_[i]->plane_ptr_);

                    if (leaves_[i]->plane_ptr_->is_plane) {
                        leaves_[i]->octo_state_ = 0;
                    } else {
                        leaves_[i]->octo_state_ = 1;
                        leaves_[i]->cut_octo_tree();
                    }

                    leaves_[i]->init_octo_ = true;
                    leaves_[i]->new_points_num_ = 0;
                }
            }
        }
    }

    void UpdateOctoTree(const pointWithCov &pv) {
        if (!init_octo_) {
            new_points_num_++;
            all_points_num_++;
            temp_points_.push_back(pv);
            if (temp_points_.size() > max_plane_update_threshold_) {
                init_octo_tree();
            }
        } else {
            if (plane_ptr_->is_plane) {
                if (update_enable_) {
                    new_points_num_++;
                    all_points_num_++;
                    if (update_cov_enable_) {
                        temp_points_.push_back(pv);
                    } else {
                        new_points_.push_back(pv);
                    }
                    if (new_points_num_ > update_size_threshold_) {
                        if (update_cov_enable_) {
                            init_plane(temp_points_, plane_ptr_);
                        }
                        new_points_num_ = 0;
                    }
                    if (all_points_num_ >= max_cov_points_size_) {
                        update_cov_enable_ = false;
                        std::vector<MapPoint>().swap(temp_points_);
                    }
                    if (all_points_num_ >= max_points_size_) {
                        update_enable_ = false;
                        plane_ptr_->update_enable = false;
                        std::vector<MapPoint>().swap(new_points_);
                    }
                } else {
                    return;
                }
            } else {
                if (layer_ < max_layer_) {
                    if (temp_points_.size() != 0) {
                        std::vector<MapPoint>().swap(temp_points_);
                    }
                    if (new_points_.size() != 0) {
                        std::vector<MapPoint>().swap(new_points_);
                    }

                    int xyz[3] = {0, 0, 0};
                    if (pv.point[0] > voxel_center_[0]) {
                        xyz[0] = 1;
                    }
                    if (pv.point[1] > voxel_center_[1]) {
                        xyz[1] = 1;
                    }
                    if (pv.point[2] > voxel_center_[2]) {
                        xyz[2] = 1;
                    }
                    int leafnum = 4 * xyz[0] + 2 * xyz[1] + xyz[2];

                    if (leaves_[leafnum] != nullptr) {
                        leaves_[leafnum]->UpdateOctoTree(pv);
                    } else {
                        leaves_[leafnum] =
                            new OctoTree(max_layer_, layer_ + 1, layer_point_size_,
                                         max_points_size_, max_cov_points_size_, planer_threshold_);
                        leaves_[leafnum]->layer_point_size_ = layer_point_size_;
                        leaves_[leafnum]->voxel_center_[0] =
                            voxel_center_[0] + (2 * xyz[0] - 1) * quater_length_;
                        leaves_[leafnum]->voxel_center_[1] =
                            voxel_center_[1] + (2 * xyz[1] - 1) * quater_length_;
                        leaves_[leafnum]->voxel_center_[2] =
                            voxel_center_[2] + (2 * xyz[2] - 1) * quater_length_;
                        leaves_[leafnum]->quater_length_ = quater_length_ / 2;
                        leaves_[leafnum]->UpdateOctoTree(pv);
                    }
                } else {
                    if (update_enable_) {
                        new_points_num_++;
                        all_points_num_++;
                        if (update_cov_enable_) {
                            temp_points_.push_back(pv);
                        } else {
                            new_points_.push_back(pv);
                        }

                        if (new_points_num_ > update_size_threshold_) {
                            if (update_cov_enable_) {
                                init_plane(temp_points_, plane_ptr_);
                            } else {
                                update_plane(new_points_, plane_ptr_);
                                new_points_.clear();
                            }
                            new_points_num_ = 0;
                        }

                        if (all_points_num_ >= max_cov_points_size_) {
                            update_cov_enable_ = false;
                            std::vector<MapPoint>().swap(temp_points_);
                        }

                        if (all_points_num_ >= max_points_size_) {
                            update_enable_ = false;
                            plane_ptr_->update_enable = false;
                            std::vector<MapPoint>().swap(new_points_);
                        }
                    }
                }
            }
        }
    }

    // void saveOctoTree(std::string save_path, std::string str_key) {
    //     std::string map_path = save_path + "/" + str_key;
    //     std::filesystem::create_directories(map_path);

    //     std::string map_file = map_path + "/octo_tree.bin";

    //     std::ofstream octo_obj(map_file, std::ios::binary);

    //     // save member variables of OctoTree
    //     octo_obj.write((char *)&max_layer_, sizeof(int));
    //     octo_obj.write((char *)&layer_, sizeof(int));
    //     octo_obj.write((char *)&voxel_center_, sizeof(double) * 3);
    //     octo_obj.write((char *)&quater_length_, sizeof(float));

    //     // save plane_ptr_
    //     octo_obj.write((char *)&plane_ptr_->center, sizeof(double) * 3);
    //     octo_obj.write((char *)&plane_ptr_->normal, sizeof(double) * 3);
    //     octo_obj.write((char *)&plane_ptr_->plane_cov, sizeof(double) * 36);
    //     octo_obj.write((char *)&plane_ptr_->radius, sizeof(float));
    //     octo_obj.write((char *)&plane_ptr_->min_eigen_value, sizeof(float));
    //     octo_obj.write((char *)&plane_ptr_->mid_eigen_value, sizeof(float));
    //     octo_obj.write((char *)&plane_ptr_->max_eigen_value, sizeof(float));
    //     octo_obj.write((char *)&plane_ptr_->d, sizeof(float));
    //     octo_obj.write((char *)&plane_ptr_->is_plane, sizeof(bool));
    //     octo_obj.write((char *)&plane_ptr_->id, sizeof(int));
    //     octo_obj.write((char *)&plane_ptr_->is_update, sizeof(bool));

    //     // save leaves_
    //     std::string child_path = map_path + "/child";
    //     for (int i = 0; i < 8; i++) {
    //         if (leaves_[i] != nullptr) {
    //             leaves_[i]->saveOctoTree(child_path, std::to_string(i));
    //         }
    //     }
    // }

    // bool loadOctoTree(std::string load_path, std::string str_key) {
    //     std::string map_path = load_path + "/" + str_key;
    //     std::string map_file = map_path + "/octo_tree.bin";
    //     // check if the file exists
    //     if (!std::filesystem::exists(map_file)) {
    //         return false;
    //     }

    //     std::ifstream octo_obj(map_file, std::ios::binary);

    //     // load member variables of OctoTree
    //     octo_obj.read((char *)&max_layer_, sizeof(int));
    //     octo_obj.read((char *)&layer_, sizeof(int));
    //     octo_obj.read((char *)&voxel_center_, sizeof(double) * 3);
    //     octo_obj.read((char *)&quater_length_, sizeof(float));

    //     // load plane_ptr_
    //     plane_ptr_ = new Plane;
    //     octo_obj.read((char *)&plane_ptr_->center, sizeof(double) * 3);
    //     octo_obj.read((char *)&plane_ptr_->normal, sizeof(double) * 3);
    //     octo_obj.read((char *)&plane_ptr_->plane_cov, sizeof(double) * 36);
    //     octo_obj.read((char *)&plane_ptr_->radius, sizeof(float));
    //     octo_obj.read((char *)&plane_ptr_->min_eigen_value, sizeof(float));
    //     octo_obj.read((char *)&plane_ptr_->mid_eigen_value, sizeof(float));
    //     octo_obj.read((char *)&plane_ptr_->max_eigen_value, sizeof(float));
    //     octo_obj.read((char *)&plane_ptr_->d, sizeof(float));
    //     octo_obj.read((char *)&plane_ptr_->is_plane, sizeof(bool));
    //     octo_obj.read((char *)&plane_ptr_->id, sizeof(int));
    //     octo_obj.read((char *)&plane_ptr_->is_update, sizeof(bool));

    //     // load leaves_
    //     std::string child_path = map_path + "/child";
    //     for (int i = 0; i < 8; i++) {
    //         OctoTree *child = new OctoTree(0, 0, {0}, 0, 0, 0);
    //         bool child_exist = child->loadOctoTree(child_path, std::to_string(i));
    //         if (child_exist) {
    //             leaves_[i] = child;
    //         } else {
    //             leaves_[i] = nullptr;
    //         }
    //     }

    //     octo_obj.close();
    //     return true;
    // }

    void getPoints(std::vector<CustomPointType> &points) {
        // url::PointType p;
        CustomPointType p;
        p.x = voxel_center_[0];
        p.y = voxel_center_[1];
        p.z = voxel_center_[2];
        points.push_back(p);
        for (int i = 0; i < 8; i++) {
            if (leaves_[i] != nullptr) {
                leaves_[i]->getPoints(points);
            }
        }
    }
};

void buildVoxelMap(const std::vector<pointWithCov> &input_points,
                   const float voxel_size,
                   const int max_layer,
                   const std::vector<int> &layer_point_size,
                   const int max_points_size,
                   const int max_cov_points_size,
                   const float planer_threshold,
                   std::unordered_map<VOXEL_LOC, OctoTree *> &feat_map);

void updateVoxelMapOMP(const std::vector<pointWithCov> &input_points,
                       const float voxel_size,
                       const int max_layer,
                       const std::vector<int> &layer_point_size,
                       const int max_points_size,
                       const int max_cov_points_size,
                       const float planer_threshold,
                       std::unordered_map<VOXEL_LOC, OctoTree *> &feat_map);

void build_single_residual(const pointWithCov &pv,
                           const OctoTree *current_octo,
                           const int current_layer,
                           const int max_layer,
                           const double sigma_num,
                           bool &is_sucess,
                           double &prob,
                           ptpl &single_ptpl);

void BuildResidualListOMP(const std::unordered_map<VOXEL_LOC, OctoTree *> &voxel_map,
                          const double voxel_size,
                          const double sigma_num,
                          const int max_layer,
                          const std::vector<pointWithCov> &pv_list,
                          std::vector<ptpl> &ptpl_list,
                          std::vector<Eigen::Vector3d> &non_match);

void GetUpdatePlane(const OctoTree *current_octo,
                    const int pub_max_voxel_layer,
                    std::vector<Plane> &plane_list);
#endif