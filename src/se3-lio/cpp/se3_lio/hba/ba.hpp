// Copied from hku-mars/HBA (a0cdd47, GPL-2.0), ROS headers and ros::Time removed; thd_num is set by OnlineHBA.
#ifndef BA_HPP
#define BA_HPP

#include <thread>
#include <fstream>
#include <iomanip>
#include <Eigen/Sparse>
#include <Eigen/Eigenvalues>
#include <Eigen/SparseCholesky>
#include <chrono>

#include "tools.hpp"

#define WIN_SIZE 10
#define GAP 5
#define AVG_THR
#define FULL_HESS
// #define ENABLE_RVIZ
// #define ENABLE_FILTER

const double one_three = (1.0 / 3.0);

inline double now_sec()
{
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

int layer_limit = 2;
int MIN_PT = 15;
int thd_num = 16;

// Frames present in a plane voxel (ascending) and their factors, one entry per present frame (sparse: a voxel
// sees a few of the window's frames, so the dense win_size arrays of the original HBA are not kept).
struct VOX_SIGS
{
  const vector<int>* idx;
  const vector<VOX_FACTOR>* sig;
};

class VOX_HESS
{
public:
  vector<VOX_SIGS> plvec_voxels;
  vector<PLV(3)> origin_points;
  int win_size;

  VOX_HESS(int _win_size = WIN_SIZE): win_size(_win_size){origin_points.resize(win_size);}

  ~VOX_HESS()
  {
    vector<VOX_SIGS>().swap(plvec_voxels);
  }

  void get_center(const PLV(3)& vec_orig, PLV(3)& origin_points_)
  {
    size_t pt_size = vec_orig.size();
    for(size_t i = 0; i < pt_size; i++)
      origin_points_.emplace_back(vec_orig[i]);
    return;
  }

  void push_voxel(const vector<int>* idx, const vector<VOX_FACTOR>* sig_orig, const vector<PLV(3)>* vec_orig)
  {
    int process_size = 0;
    for(size_t k = 0; k < idx->size(); k++)
      if((*sig_orig)[k].N != 0)
        process_size++;

    #ifdef ENABLE_FILTER
    if(process_size < 1) return;

    for(size_t k = 0; k < idx->size(); k++)
      if((*sig_orig)[k].N != 0)
        get_center((*vec_orig)[k], origin_points[(*idx)[k]]);
    #endif
    
    if(process_size < 2) return;
    
    plvec_voxels.push_back({idx, sig_orig});
  }

  Eigen::Matrix<double, 6, 1> lam_f(Eigen::Vector3d *u, int m, int n)
  {
    Eigen::Matrix<double, 6, 1> jac;
    jac[0] = u[m][0] * u[n][0];
    jac[1] = u[m][0] * u[n][1] + u[m][1] * u[n][0];
    jac[2] = u[m][0] * u[n][2] + u[m][2] * u[n][0];
    jac[3] = u[m][1] * u[n][1];
    jac[4] = u[m][1] * u[n][2] + u[m][2] * u[n][1];
    jac[5] = u[m][2] * u[n][2];
    return jac;
  }

  void acc_evaluate2(const vector<IMUST>& xs, int head, int end,
                     Eigen::MatrixXd& Hess, Eigen::VectorXd& JacT, double& residual)
  {
    Hess.setZero(); JacT.setZero(); residual = 0;
    vector<VOX_FACTOR> sig_tran(win_size);
    const int kk = 0;

    PLV(3) viRiTuk(win_size);
    PLM(3) viRiTukukT(win_size);

    vector<Eigen::Matrix<double, 3, 6>, Eigen::aligned_allocator<Eigen::Matrix<double, 3, 6>>> Auk(win_size);
    Eigen::Matrix3d umumT;

    for(int a = head; a < end; a++)
    {
      const vector<int>& idx = *plvec_voxels[a].idx;
      const vector<VOX_FACTOR>& sig_orig = *plvec_voxels[a].sig;
      const int K = idx.size();

      VOX_FACTOR sig;
      for(int k = 0; k < K; k++)
        if(sig_orig[k].N != 0)
        {
          int i = idx[k];
          sig_tran[i].transform(sig_orig[k], xs[i]);
          sig += sig_tran[i];
        }
      
      const Eigen::Vector3d& vBar = sig.v / sig.N;
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(sig.P/sig.N - vBar * vBar.transpose());
      const Eigen::Vector3d& lmbd = saes.eigenvalues();
      const Eigen::Matrix3d& U = saes.eigenvectors();
      int NN = sig.N;
      
      Eigen::Vector3d u[3] = {U.col(0), U.col(1), U.col(2)};

      const Eigen::Vector3d& uk = u[kk];
      Eigen::Matrix3d ukukT = uk * uk.transpose();
      umumT.setZero();
      for(int i = 0; i < 3; i++)
        if(i != kk)
          umumT += 2.0/(lmbd[kk] - lmbd[i]) * u[i] * u[i].transpose();

      for(int k = 0; k < K; k++)
        if(sig_orig[k].N != 0)
        {
          int i = idx[k];
          Eigen::Matrix3d Pi = sig_orig[k].P;
          Eigen::Vector3d vi = sig_orig[k].v;
          Eigen::Matrix3d Ri = xs[i].R;
          double ni = sig_orig[k].N;

          Eigen::Matrix3d vihat; vihat << SKEW_SYM_MATRX(vi);
          Eigen::Vector3d RiTuk = Ri.transpose() * uk;
          Eigen::Matrix3d RiTukhat; RiTukhat << SKEW_SYM_MATRX(RiTuk);

          Eigen::Vector3d PiRiTuk = Pi * RiTuk;
          viRiTuk[i] = vihat * RiTuk;
          viRiTukukT[i] = viRiTuk[i] * uk.transpose();
          
          Eigen::Vector3d ti_v = xs[i].p - vBar;
          double ukTti_v = uk.dot(ti_v);

          Eigen::Matrix3d combo1 = hat(PiRiTuk) + vihat * ukTti_v;
          Eigen::Vector3d combo2 = Ri*vi + ni*ti_v;
          Auk[i].block<3, 3>(0, 0) = (Ri*Pi + ti_v*vi.transpose()) * RiTukhat - Ri*combo1;
          Auk[i].block<3, 3>(0, 3) = combo2 * uk.transpose() + combo2.dot(uk) * I33;
          Auk[i] /= NN;

          const Eigen::Matrix<double, 6, 1> &jjt = Auk[i].transpose() * uk;
          JacT.block<6, 1>(6*i, 0) += jjt;

          const Eigen::Matrix3d &HRt = 2.0/NN * (1.0-ni/NN) * viRiTukukT[i];
          Eigen::Matrix<double, 6, 6> Hb = Auk[i].transpose() * umumT * Auk[i];
          Hb.block<3, 3>(0, 0) +=
            2.0/NN*(combo1-RiTukhat*Pi)*RiTukhat - 2.0/NN/NN*viRiTuk[i]*viRiTuk[i].transpose() - 0.5*hat(jjt.block<3, 1>(0, 0));
          Hb.block<3, 3>(0, 3) += HRt;
          Hb.block<3, 3>(3, 0) += HRt.transpose();
          Hb.block<3, 3>(3, 3) += 2.0/NN * (ni - ni*ni/NN) * ukukT;

          Hess.block<6, 6>(6*i, 6*i) += Hb;
        }
      
      for(int ka = 0; ka < K-1; ka++)
        if(sig_orig[ka].N != 0)
        {
          int i = idx[ka];
          double ni = sig_orig[ka].N;
          for(int kb = ka+1; kb < K; kb++)
            if(sig_orig[kb].N != 0)
            {
              int j = idx[kb];
              double nj = sig_orig[kb].N;
              Eigen::Matrix<double, 6, 6> Hb = Auk[i].transpose() * umumT * Auk[j];
              Hb.block<3, 3>(0, 0) += -2.0/NN/NN * viRiTuk[i] * viRiTuk[j].transpose();
              Hb.block<3, 3>(0, 3) += -2.0*nj/NN/NN * viRiTukukT[i];
              Hb.block<3, 3>(3, 0) += -2.0*ni/NN/NN * viRiTukukT[j].transpose();
              Hb.block<3, 3>(3, 3) += -2.0*ni*nj/NN/NN * ukukT;

              Hess.block<6, 6>(6*i, 6*j) += Hb;
            }
        }
      
      residual += lmbd[kk];
    }

    for(int i = 1; i < win_size; i++)
      for(int j = 0; j < i; j++)
        Hess.block<6, 6>(6*i, 6*j) = Hess.block<6, 6>(6*j, 6*i).transpose();
    
  }

  void evaluate_only_residual(const vector<IMUST>& xs, double& residual)
  {
    residual = 0;
    vector<VOX_FACTOR> sig_tran(win_size);
    int kk = 0; // The kk-th lambda value

    int gps_size = plvec_voxels.size();

    for(int a = 0; a < gps_size; a++)
    {
      const vector<int>& idx = *plvec_voxels[a].idx;
      const vector<VOX_FACTOR>& sig_orig = *plvec_voxels[a].sig;
      VOX_FACTOR sig;

      for(size_t k = 0; k < idx.size(); k++)
      {
        int i = idx[k];
        sig_tran[i].transform(sig_orig[k], xs[i]);
        sig += sig_tran[i];
      }

      Eigen::Vector3d vBar = sig.v / sig.N;
      Eigen::Matrix3d cmt = sig.P/sig.N - vBar * vBar.transpose();

      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(cmt);
      Eigen::Vector3d lmbd = saes.eigenvalues();

      residual += lmbd[kk];
    }
  }

  std::vector<double> evaluate_residual(const vector<IMUST>& xs)
  {
    /* for outlier removal usage */
    std::vector<double> residuals;
    vector<VOX_FACTOR> sig_tran(win_size);
    int kk = 0; // The kk-th lambda value
    int gps_size = plvec_voxels.size();

    for(int a = 0; a < gps_size; a++)
    {
      const vector<int>& idx = *plvec_voxels[a].idx;
      const vector<VOX_FACTOR>& sig_orig = *plvec_voxels[a].sig;
      VOX_FACTOR sig;

      for(size_t k = 0; k < idx.size(); k++)
      {
        int i = idx[k];
        sig_tran[i].transform(sig_orig[k], xs[i]);
        sig += sig_tran[i];
      }

      Eigen::Vector3d vBar = sig.v / sig.N;
      Eigen::Matrix3d cmt = sig.P / sig.N - vBar * vBar.transpose();

      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(cmt);
      Eigen::Vector3d lmbd = saes.eigenvalues();

      residuals.push_back(lmbd[kk]);
    }

    return residuals;
  }

  void remove_residual(const vector<IMUST>& xs, double threshold, double reject_num)
  {
    vector<VOX_FACTOR> sig_tran(win_size);
    int kk = 0; // The kk-th lambda value
    int rej_cnt = 0;
    size_t i = 0;
    for(; i < plvec_voxels.size();)
    {
      const vector<int>& idx = *plvec_voxels[i].idx;
      const vector<VOX_FACTOR>& sig_orig = *plvec_voxels[i].sig;
      VOX_FACTOR sig;

      for(size_t k = 0; k < idx.size(); k++)
      {
        int j = idx[k];
        sig_tran[j].transform(sig_orig[k], xs[j]);
        sig += sig_tran[j];
      }

      Eigen::Vector3d vBar = sig.v / sig.N;
      Eigen::Matrix3d cmt = sig.P / sig.N - vBar * vBar.transpose();
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(cmt);
      Eigen::Vector3d lmbd = saes.eigenvalues();

      if(lmbd[kk] >= threshold)
      {
        plvec_voxels.erase(plvec_voxels.begin()+i);
        rej_cnt++;
        continue;
      }
      i++;
      if(rej_cnt == reject_num) break;
    }
  }
};

int BINGO_CNT = 0;
enum OCTO_STATE {UNKNOWN, MID_NODE, PLANE};
class OCTO_TREE_NODE
{
public:
  OCTO_STATE octo_state;
  int layer, win_size;
  vector<int> idx;                        // frames with points here, ascending; the vectors below are parallel to it
  vector<PLV(3)> vec_orig, vec_tran;
  vector<VOX_FACTOR> sig_orig, sig_tran;

  OCTO_TREE_NODE* leaves[8];
  float voxel_center[3];
  float quater_length;
  float eigen_thr;

  Eigen::Vector3d center, direct, value_vector;
  double eigen_ratio;
  
  #ifdef ENABLE_RVIZ
  ros::NodeHandle nh;
  ros::Publisher pub_residual = nh.advertise<sensor_msgs::PointCloud2>("/residual", 1000);
  ros::Publisher pub_direct = nh.advertise<visualization_msgs::MarkerArray>("/direct", 1000);
  #endif

  OCTO_TREE_NODE(int _win_size = WIN_SIZE, float _eigen_thr = 1.0/10):
    win_size(_win_size), eigen_thr(_eigen_thr)
  {
    octo_state = UNKNOWN; layer = 0;
    for(int i = 0; i < 8; i++)
      leaves[i] = nullptr;
  }

  // A point of frame `fnum`; frames must arrive in ascending order (cut_voxel walks the frames in order).
  void add(int fnum, const Eigen::Vector3d& orig, const Eigen::Vector3d& tran)
  {
    if(idx.empty() || idx.back() != fnum)
    {
      idx.push_back(fnum);
      vec_orig.emplace_back(); vec_tran.emplace_back();
      sig_orig.emplace_back(); sig_tran.emplace_back();
    }
    vec_orig.back().push_back(orig);
    vec_tran.back().push_back(tran);
    sig_orig.back().push(orig);
    sig_tran.back().push(tran);
  }

  virtual ~OCTO_TREE_NODE()
  {
    for(int i = 0; i < 8; i++)
      if(leaves[i] != nullptr)
        delete leaves[i];
  }

  bool judge_eigen()
  {
    VOX_FACTOR covMat;
    for(size_t k = 0; k < idx.size(); k++)
      if(sig_tran[k].N > 0)
        covMat += sig_tran[k];
    
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covMat.cov());
    value_vector = saes.eigenvalues();
    center = covMat.v / covMat.N;
    direct = saes.eigenvectors().col(0);

    eigen_ratio = saes.eigenvalues()[0] / saes.eigenvalues()[2]; // [0] is the smallest
    if(eigen_ratio > eigen_thr) return 0;

    double eva0 = saes.eigenvalues()[0];
    double sqr_eva0 = sqrt(eva0);
    Eigen::Vector3d center_turb = center + 5 * sqr_eva0 * direct;
    vector<VOX_FACTOR> covMats(8);
    for(size_t i = 0; i < idx.size(); i++)
    {
      for(Eigen::Vector3d ap: vec_tran[i])
      {
        int xyz[3] = {0, 0, 0};
        for(int k = 0; k < 3; k++)
          if(ap(k) > center_turb[k])
            xyz[k] = 1;

        Eigen::Vector3d pvec(ap(0), ap(1), ap(2));
        
        int leafnum = 4*xyz[0] + 2*xyz[1] + xyz[2];
        covMats[leafnum].push(pvec);
      }
    }

    double ratios[2] = {1.0/(3.0*3.0), 2.0*2.0};
    int num_all = 0, num_qua = 0;
    for(int i = 0; i < 8; i++)
      if(covMats[i].N > 10)
      {
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covMats[i].cov());
        double child_eva0 = (saes.eigenvalues()[0]);
        if(child_eva0 > ratios[0]*eva0 && child_eva0 < ratios[1]*eva0)
          num_qua++;
        num_all++;
      }

    double prop = 1.0 * num_qua / num_all;

    if(prop < 0.5) return 0;
    return 1;
  }

  void cut_func(int ci)   // entry ci (frame idx[ci]) of this node into the leaves
  {
    PLV(3)& pvec_orig = vec_orig[ci];
    PLV(3)& pvec_tran = vec_tran[ci];
    const int fnum = idx[ci];

    uint a_size = pvec_tran.size();
    for(uint j = 0; j < a_size; j++)
    {
      int xyz[3] = {0, 0, 0};
      for(uint k = 0; k < 3; k++)
        if(pvec_tran[j][k] > voxel_center[k])
          xyz[k] = 1;
      int leafnum = 4*xyz[0] + 2*xyz[1] + xyz[2];
      if(leaves[leafnum] == nullptr)
      {
        leaves[leafnum] = new OCTO_TREE_NODE(win_size, eigen_thr);
        leaves[leafnum]->voxel_center[0] = voxel_center[0] + (2*xyz[0]-1)*quater_length;
        leaves[leafnum]->voxel_center[1] = voxel_center[1] + (2*xyz[1]-1)*quater_length;
        leaves[leafnum]->voxel_center[2] = voxel_center[2] + (2*xyz[2]-1)*quater_length;
        leaves[leafnum]->quater_length = quater_length / 2.0;
        leaves[leafnum]->layer = layer + 1;
      }

      leaves[leafnum]->add(fnum, pvec_orig[j], pvec_tran[j]);
    }

    PLV(3)().swap(pvec_orig);
    PLV(3)().swap(pvec_tran);
  }

  void recut()
  {
    if(octo_state == UNKNOWN)
    {
      int point_size = 0;
      for(size_t k = 0; k < idx.size(); k++)
        point_size += sig_orig[k].N;
      
      if(point_size < MIN_PT)
      {
        octo_state = MID_NODE;
        vector<int>().swap(idx);
        vector<PLV(3)>().swap(vec_orig);
        vector<PLV(3)>().swap(vec_tran);
        vector<VOX_FACTOR>().swap(sig_orig);
        vector<VOX_FACTOR>().swap(sig_tran);
        return;
      }

      if(judge_eigen())
      {
        octo_state = PLANE;
        #ifndef ENABLE_FILTER
        #ifndef ENABLE_RVIZ
        vector<PLV(3)>().swap(vec_orig);
        vector<PLV(3)>().swap(vec_tran);
        #endif
        #endif
        vector<VOX_FACTOR>().swap(sig_tran);
        return;
      }
      else
      {
        if(layer == layer_limit)
        {
          octo_state = MID_NODE;
          vector<int>().swap(idx);
          vector<PLV(3)>().swap(vec_orig);
          vector<PLV(3)>().swap(vec_tran);
          vector<VOX_FACTOR>().swap(sig_orig);
          vector<VOX_FACTOR>().swap(sig_tran);
          return;
        }
        vector<VOX_FACTOR>().swap(sig_orig);
        vector<VOX_FACTOR>().swap(sig_tran);
        for(size_t k = 0; k < idx.size(); k++)
          cut_func(k);
        vector<int>().swap(idx);
        vector<PLV(3)>().swap(vec_orig);
        vector<PLV(3)>().swap(vec_tran);
      }
    }
    
    for(int i = 0; i < 8; i++)
      if(leaves[i] != nullptr)
        leaves[i]->recut();
  }

  void tras_opt(VOX_HESS& vox_opt)
  {
    if(octo_state == PLANE)
      vox_opt.push_voxel(&idx, &sig_orig, &vec_orig);
    else
      for(int i = 0; i < 8; i++)
        if(leaves[i] != nullptr)
          leaves[i]->tras_opt(vox_opt);
  }

  void tras_display(int layer = 0)
  {
    float ref = 255.0*rand()/(RAND_MAX + 1.0f);
    pcl::PointXYZINormal ap;
    ap.intensity = ref;

    if(octo_state == PLANE)
    {
      // std::vector<unsigned int> colors;
			// colors.push_back(static_cast<unsigned int>(rand() % 256));
			// colors.push_back(static_cast<unsigned int>(rand() % 256));
			// colors.push_back(static_cast<unsigned int>(rand() % 256));
      pcl::PointCloud<pcl::PointXYZINormal> color_cloud;

      for(size_t i = 0; i < idx.size(); i++)
      {
        for(size_t j = 0; j < vec_tran[i].size(); j++)
        {
          Eigen::Vector3d& pvec = vec_tran[i][j];
          ap.x = pvec.x();
          ap.y = pvec.y();
          ap.z = pvec.z();
          // ap.b = colors[0];
          // ap.g = colors[1];
          // ap.r = colors[2];
          ap.normal_x = sqrt(value_vector[1] / value_vector[0]);
          ap.normal_y = sqrt(value_vector[2] / value_vector[0]);
          ap.normal_z = sqrt(value_vector[0]);
          // ap.curvature = total;
          color_cloud.push_back(ap);
        }
      }

      #ifdef ENABLE_RVIZ
      sensor_msgs::PointCloud2 dbg_msg;
      pcl::toROSMsg(color_cloud, dbg_msg);
      dbg_msg.header.frame_id = "camera_init";
      pub_residual.publish(dbg_msg);

      visualization_msgs::Marker marker;
      visualization_msgs::MarkerArray marker_array;
      marker.header.frame_id = "camera_init";
      marker.header.stamp = ros::Time::now();
      marker.ns = "basic_shapes";
      marker.id = BINGO_CNT; BINGO_CNT++;
      marker.action = visualization_msgs::Marker::ADD;
      marker.type = visualization_msgs::Marker::ARROW;
      marker.color.a = 1;
      marker.color.r = layer==0?1:0;
      marker.color.g = layer==1?1:0;
      marker.color.b = layer==2?1:0;
      marker.scale.x = 0.01;
      marker.scale.y = 0.05;
      marker.scale.z = 0.05;
      marker.lifetime = ros::Duration();
      geometry_msgs::Point apoint;
      apoint.x = center(0); apoint.y = center(1); apoint.z = center(2);
      marker.points.push_back(apoint);
      apoint.x += 0.2*direct(0); apoint.y += 0.2*direct(1); apoint.z += 0.2*direct(2);
      marker.points.push_back(apoint);
      marker_array.markers.push_back(marker);
      pub_direct.publish(marker_array);
      #endif
    }
    else
    {
      if(layer == layer_limit) return;
      layer++;
      for(int i = 0; i < 8; i++)
        if(leaves[i] != nullptr)
          leaves[i]->tras_display(layer);
    }
  }
};

class OCTO_TREE_ROOT: public OCTO_TREE_NODE
{
public:
  OCTO_TREE_ROOT(int _winsize, float _eigen_thr): OCTO_TREE_NODE(_winsize, _eigen_thr){}
};

class VOX_OPTIMIZER
{
public:
  int win_size, jac_leng, imu_leng;
  VOX_OPTIMIZER(int _win_size = WIN_SIZE): win_size(_win_size)
  {
    jac_leng = DVEL * win_size;
    imu_leng = DIM * win_size;
  }

  double divide_thread(vector<IMUST>& x_stats, VOX_HESS& voxhess, vector<IMUST>& x_ab,
                       Eigen::MatrixXd& Hess,Eigen::VectorXd& JacT)
  {
    double residual = 0;
    Hess.setZero(); JacT.setZero();
    PLM(-1) hessians(thd_num);
    PLV(-1) jacobins(thd_num);

    for(int i = 0; i < thd_num; i++)
    {
      hessians[i].resize(jac_leng, jac_leng);
      jacobins[i].resize(jac_leng);
    }

    int tthd_num = thd_num;
    vector<double> resis(tthd_num, 0);
    int g_size = voxhess.plvec_voxels.size();
    if(g_size < tthd_num) tthd_num = 1;

    vector<thread*> mthreads(tthd_num);
    double part = 1.0 * g_size / tthd_num;
    for(int i = 0; i < tthd_num; i++)
      mthreads[i] = new thread(&VOX_HESS::acc_evaluate2, &voxhess, x_stats, part*i, part*(i+1),
                               ref(hessians[i]), ref(jacobins[i]), ref(resis[i]));

    for(int i = 0; i < tthd_num; i++)
    {
      mthreads[i]->join();
      Hess += hessians[i];
      JacT += jacobins[i];
      residual += resis[i];
      delete mthreads[i];
    }
    #ifdef AVG_THR
    return residual/g_size;
    #else
    return residual;
    #endif
  }

  double only_residual(vector<IMUST>& x_stats, VOX_HESS& voxhess, vector<IMUST>& x_ab, bool is_avg = false)
  {
    double residual2 = 0;
    voxhess.evaluate_only_residual(x_stats, residual2);
    if(is_avg) return residual2 / voxhess.plvec_voxels.size();
    return residual2;
  }

  void remove_outlier(vector<IMUST>& x_stats, VOX_HESS& voxhess, double ratio)
  {
    std::vector<double> residuals = voxhess.evaluate_residual(x_stats);
    std::sort(residuals.begin(), residuals.end()); // sort in ascending order
    double threshold = residuals[std::floor((1-ratio)*voxhess.plvec_voxels.size())-1];
    int reject_num = std::floor(ratio * voxhess.plvec_voxels.size());
    // std::cout << "vox_num before " << voxhess.plvec_voxels.size();
    // std::cout << ", reject threshold " << std::setprecision(3) << threshold << ", rejected " << reject_num;
    voxhess.remove_residual(x_stats, threshold, reject_num);
    // std::cout << ", vox_num after " << voxhess.plvec_voxels.size() << std::endl;
  }

  void damping_iter(vector<IMUST>& x_stats, VOX_HESS& voxhess, double& residual,
                    PLV(6)& hess_vec, size_t& mem_cost)
  {
    double u = 0.01, v = 2;
    Eigen::MatrixXd D(jac_leng, jac_leng), Hess(jac_leng, jac_leng),
                    HessuD(jac_leng, jac_leng);
    Eigen::VectorXd JacT(jac_leng), dxi(jac_leng), new_dxi(jac_leng);

    D.setIdentity();
    double residual1, residual2, q;
    bool is_calc_hess = true;
    vector<IMUST> x_stats_temp;

    vector<IMUST> x_ab(win_size);
    x_ab[0] = x_stats[0];
    for(int i=1; i<win_size; i++)
    {
      x_ab[i].p = x_stats[i-1].R.transpose() * (x_stats[i].p - x_stats[i-1].p);
      x_ab[i].R = x_stats[i-1].R.transpose() * x_stats[i].R;
    }

    double hesstime = 0;
    double solvtime = 0;
    size_t max_mem = 0;
    double loop_num = 0;
    for(int i = 0; i < 10; i++)
    {
      if(is_calc_hess)
      {
        double tm = now_sec();
        residual1 = divide_thread(x_stats, voxhess, x_ab, Hess, JacT);
        hesstime += now_sec() - tm;
      }

      double tm = now_sec();
      D.diagonal() = Hess.diagonal();
      HessuD = Hess + u*D;
      double t1 = now_sec();
      Eigen::SparseMatrix<double> A1_sparse(jac_leng, jac_leng);
      std::vector<Eigen::Triplet<double>> tripletlist;
      for(int a = 0; a < jac_leng; a++)
        for(int b = 0; b < jac_leng; b++)
          if(HessuD(a, b) != 0)
          {
            tripletlist.push_back(Eigen::Triplet<double>(a, b, HessuD(a, b)));
            //A1_sparse.insert(a, b) = HessuD(a, b);
          }
      A1_sparse.setFromTriplets(tripletlist.begin(), tripletlist.end());
      A1_sparse.makeCompressed();
      Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> Solver_sparse;
      Solver_sparse.compute(A1_sparse);
      size_t temp_mem = check_mem();
      if(temp_mem > max_mem) max_mem = temp_mem;
      dxi = Solver_sparse.solve(-JacT);
      temp_mem = check_mem();
      if(temp_mem > max_mem) max_mem = temp_mem;
      solvtime += now_sec() - tm;
      // new_dxi = Solver_sparse.solve(-JacT);
      // printf("new solve time cost %f\n",now_sec() - t1);
      // relative_err = ((Hess + u*D)*dxi + JacT).norm()/JacT.norm();
      // absolute_err = ((Hess + u*D)*dxi + JacT).norm();
      // std::cout<<"relative error "<<relative_err<<std::endl;
      // std::cout<<"absolute error "<<absolute_err<<std::endl;
      // std::cout<<"delta x\n"<<(new_dxi-dxi).transpose()/dxi.norm()<<std::endl;

      x_stats_temp = x_stats;
      for(int j = 0; j < win_size; j++)
      {
        x_stats_temp[j].R = x_stats[j].R * Exp(dxi.block<3, 1>(DVEL*j, 0));
        x_stats_temp[j].p = x_stats[j].p + dxi.block<3, 1>(DVEL*j+3, 0);
      }

      double q1 = 0.5*dxi.dot(u*D*dxi-JacT);
      #ifdef AVG_THR
      residual2 = only_residual(x_stats_temp, voxhess, x_ab, true);
      q1 /= voxhess.plvec_voxels.size();
      #else
      residual2 = only_residual(x_stats_temp, voxhess, x_ab);
      #endif
      residual = residual2;
      q = (residual1-residual2);
      // printf("iter%d: (%lf %lf) u: %lf v: %lf q: %lf %lf %lf\n",
      //        i, residual1, residual2, u, v, q/q1, q1, q);
      loop_num = i+1;
      // if(hesstime/loop_num > 1) printf("Avg. Hessian time: %lf ", hesstime/loop_num);
      // if(solvtime/loop_num > 1) printf("Avg. solve time: %lf\n", solvtime/loop_num);
      // if(double(max_mem/1048576.0) > 2.0) printf("Max mem: %lf\n", double(max_mem/1048576.0));
      
      if(q > 0)
      {
        x_stats = x_stats_temp;
        q = q / q1;
        v = 2;
        q = 1 - pow(2*q-1, 3);
        u *= (q<one_three ? one_three:q);
        is_calc_hess = true;
      }
      else
      {
        u = u * v;
        v = 2 * v;
        is_calc_hess = false;
      }
      #ifdef AVG_THR
      if((fabs(residual1-residual2)/residual1) < 0.05 || i == 9)
      {
        if(mem_cost < max_mem) mem_cost = max_mem;
        for(int j = 0; j < win_size-1; j++)
          for(int k = j+1; k < win_size; k++)
            hess_vec.push_back(Hess.block<DVEL, DVEL>(DVEL*j, DVEL*k).diagonal().segment<DVEL>(0));
        break;
      }
      #else
      if(fabs(residual1-residual2)<1e-9) break;
      #endif
    }
  }

  size_t check_mem()
  {
    FILE* file = fopen("/proc/self/status", "r");
    int result = -1;
    char line[128];

    while(fgets(line, 128, file) != nullptr)
    {
      if(strncmp(line, "VmRSS:", 6) == 0)
      {
        int len = strlen(line);

        const char* p = line;
        for(; std::isdigit(*p) == false; ++p){}

        line[len - 3] = 0;
        result = atoi(p);

        break;
      }
    }
    fclose(file);

    return result;
  }
};

#endif