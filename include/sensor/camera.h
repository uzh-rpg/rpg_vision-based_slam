/* This file is part of RPG vision-based SLAM.
Copyright (C) 2022 Giovanni Cioffi <cioffi at ifi dot uzh dot ch>
(Robotics and Perception Group, University of Zurich, Switzerland).

This file is subject to the terms and conditions defined in the file
'LICENSE', which is part of this source code package.
*/

#include <iostream>
#include <unordered_map>

#include <glog/logging.h>

#include "sensor/camera_calib.h"
#include "util/eigen_utils.h"
#include "util/endian.h"


class Camera
{
    public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    uint32_t cam_id;
    int model_id;
    uint64_t w, h;
    double fx, fy;
    double cx, cy;
    std::vector<double> dist_coeffs;
    Sophus::SE3d T_b_c; // imu-cam transformation
    Sophus::SE3d T_c_b; // cam-imu transformation
    double t_offset_cam_imu = 0.0; // t_imu = t_cam + t_offset_cam_imu
    
    DistortionModel* dist_model_ptr_ = new DistortionModel;

    Camera(){};
    ~Camera(){};

    void setCameraId(uint32_t id) {cam_id = id;}
    void setModelId(int id) 
    {
        model_id = id; 
        dist_coeffs.resize(numberParameters(id), 0);
        
    }
    void setWidth(uint64_t a) {w = a;}
    void setHeight(uint64_t a) {h = a;}
    void setFocalLength(double x, double y) {fx = x; fy = y;}
    void setPrincipalPoint(double x, double y) {cx = x; cy = y;}
    void setDistortionModel()
    {
        switch (model_id) 
        {
            // opencv full camera model (= pinhole, rad-tan)
            case 4:
            dist_model_ptr_->setDistortionModel(model_id);

            //opencv fisheye
            case 5:
            dist_model_ptr_->setDistortionModel(model_id);
            break;

            //fov
            case 7:
            dist_model_ptr_->setDistortionModel(model_id);
            break;
            
            default:
            GVI_FUSION_ASSERT_STREAM(false, "Camera model id " << model_id << " not supported!");
        }
    }
    void setTbc(Sophus::SE3d& T) 
    {
        T_b_c = T;
        T_c_b = T.inverse();
    }
    void setTimeOffsetCamImu(double t) {t_offset_cam_imu = t;}

    template <typename T>
    void project(const Eigen::Matrix<T, 3, 1>& p_C, Eigen::Matrix<T, 2, 1>& uv) const
    {
        Eigen::Matrix<T, 2, 1> uv_distorted;
        uv_distorted[0] = p_C[0] / p_C[2];
        uv_distorted[1] = p_C[1] / p_C[2];

        dist_model_ptr_->distort(dist_coeffs, uv_distorted);
        
        uv[0] = fx * uv_distorted[0] + cx;
        uv[1] = fy * uv_distorted[1] + cy;
    }
};


struct Point2D
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    uint32_t id;
    Eigen::Vector2d uv;
    uint64_t point3d_id;
    bool is_reproj_valid;
    Eigen::Vector2d velocity;

    Point2D(){};
    Point2D(uint32_t i, Eigen::Vector2d p, uint64_t i_3D) : 
    id(i), uv(p), point3d_id(i_3D), is_reproj_valid(false) 
    {
        velocity.setZero();
    };
    ~Point2D(){};
};


struct Frame 
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    std::string name;
    uint32_t id, cam_id;
    uint64_t w, h;
    Eigen::Quaterniond q_cw_colmap;
    Eigen::Vector3d p_cw_colmap;
    Eigen::aligned_vector<Point2D> points2D;
    std::vector<uint64_t> points3D_ids;
    double timestamp;
    /*
    Features on the frame are split in a grid of squared cells.
    The grid is defined as grid[idx] = [feature_idx, feature_idx, ..],
    - idx is the index of the cell. Cells are indexed column-wise from top-left to 
    bottom-right. 
    For example, for a 3x3 grid indexes are organized as =  [0 1 2;
                                                             3 4 5;
                                                             6 7 8].
    - feature_idx is the index of the feature in the vector points2D
    */
    // Not working when cell_size_px = 1
    int cell_size_px;
    int n_grid_columns;
    int n_grid_rows;
    std::vector<std::vector<size_t>> grid; 
    int n_valid_reprojections = 0;
    Eigen::Vector2d avg_pixel_vel;

    Frame()
    {
        avg_pixel_vel.setZero();
    };
    ~Frame(){};

    void setImageId(uint32_t im_id) {id = im_id;}
    void setWidth(uint64_t a) {w = a;}
    void setHeight(uint64_t a) {h = a;}
    void setColmapQcw(Eigen::Quaterniond q) {q_cw_colmap = q;}
    void setColmapPcw(Eigen::Vector3d p) {p_cw_colmap = p;}
    void setCameraId(uint32_t i) {cam_id = i;}
    void setGridCellSize(int v) {cell_size_px = v;}
    void addPoint2D(uint32_t id, Eigen::Vector2d p, uint64_t id_p3D) 
    {
        Point2D point(id, p, id_p3D);
        points2D.emplace_back(point);

        // add to grid
        int idx_clmn = p.x() / cell_size_px;
        int idx_row = p.y() / cell_size_px;
        int idx = idx_clmn + idx_row * n_grid_columns;
        size_t idx_in_points2D = points2D.size() - 1;
        grid[idx].push_back(idx_in_points2D);
    }
    void addPoint3DId(uint64_t id) {points3D_ids.emplace_back(id);}
    size_t numPoints2D() {return points2D.size();}
    void setTimestamp(double ts) {timestamp = ts;}
    void initializeGrid()
    {
        n_grid_columns = w / cell_size_px;
        int r = w % cell_size_px;
        if (r > 0) { n_grid_columns+=1; }

        n_grid_rows = h / cell_size_px;
        r = h % cell_size_px;
        if (r > 0) { n_grid_rows+=1; }

        int n_cells = n_grid_rows * n_grid_columns;
        grid.resize(n_cells);
    }

    void computeAveragePixelVelocity()
    {
        int n_points_with_no_zero_vel = 0;
        double avg_vel_u = 0.0;
        double avg_vel_v = 0.0;
        for (const auto& p : points2D)
        {
            // points not tracked on next frame will have 0 vel.
            if (p.velocity.norm() > 1e-6)
            {
                avg_vel_u += p.velocity[0];
                avg_vel_v += p.velocity[1];
                n_points_with_no_zero_vel++;
            }
        }

        if (n_points_with_no_zero_vel > 0)
        {
            avg_vel_u /= n_points_with_no_zero_vel;
            avg_vel_v /= n_points_with_no_zero_vel;
        }

        avg_pixel_vel << avg_vel_u, avg_vel_v;
    }
};


struct Point3D
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    uint64_t id;
    Eigen::Vector3d p_wl; // World frame
    Eigen::Vector3d p_gl; // Colmap frame
    // mean reprojection error
    double error;
    // image ids where the point reprojects
    std::vector<uint32_t> img_ids;
    // corresponding feature ids
    std::vector<uint32_t> points2D_ids;

    Point3D(){};
    ~Point3D(){};

    void setId(uint64_t i) {id = i;}
    void setPositionInColmapFrame(Eigen::Vector3d p) {p_gl = p;}
    void setPositionInWorldFrame(Eigen::Vector3d p) {p_wl = p;}
    void setError(double e) {error = e;}
    void addImgId(uint32_t id) {img_ids.emplace_back(id);}
    void addPoint2DId(uint32_t id) {points2D_ids.emplace_back(id);}
};


/*
The following code has been adapted from: 
https://github.com/colmap/colmap/blob/99c7904c4a3c99c17d80fee1d8b57ed7fe5d3753/src/base/reconstruction.cc
*/


void readCamerasBinary(const std::string& path, Eigen::aligned_vector<Camera>& cameras) 
{
    std::ifstream file(path, std::ios::binary);
    CHECK(file.is_open()) << path;

    const size_t num_cameras = readBinaryLittleEndian<uint64_t>(&file);

    for (size_t i = 0; i < num_cameras; i++)
    {   
        Camera camera;
        camera.setCameraId(readBinaryLittleEndian<uint32_t>(&file));
        camera.setModelId(readBinaryLittleEndian<int>(&file));
        camera.setWidth(readBinaryLittleEndian<uint64_t>(&file));
        camera.setHeight(readBinaryLittleEndian<uint64_t>(&file));
        double fx = readBinaryLittleEndian<double>(&file);
        double fy = readBinaryLittleEndian<double>(&file);
        camera.setFocalLength(fx, fy);
        double cx = readBinaryLittleEndian<double>(&file);
        double cy = readBinaryLittleEndian<double>(&file);
        camera.setPrincipalPoint(cx, cy);
        readBinaryLittleEndian<double>(&file, &camera.dist_coeffs);
        camera.setDistortionModel();

        cameras.emplace_back(camera);
    }
}


void readImagesAndPoints3DBinary(
    const std::string& path_images, 
    const std::string& path_points3d, 
    const Eigen::aligned_vector<Camera>& cameras, 
    Eigen::aligned_vector<Frame>& frames, 
    Eigen::aligned_unordered_map<uint64_t, Point3D>& points3D, 
    const int grid_cell_size = 30
    )
{
  std::ifstream file_images(path_images, std::ios::binary);
  CHECK(file_images.is_open()) << path_images;

  const uint64_t kInvalidPoint3DId = std::numeric_limits<uint64_t>::max();

  const size_t num_reg_images = readBinaryLittleEndian<uint64_t>(&file_images);
  for (size_t i = 0; i < num_reg_images; ++i) 
  {
    Frame frame;

    uint32_t im_id = readBinaryLittleEndian<uint32_t>(&file_images);
    frame.setImageId(im_id - 1);

    double qw = readBinaryLittleEndian<double>(&file_images);
    double qx = readBinaryLittleEndian<double>(&file_images);
    double qy = readBinaryLittleEndian<double>(&file_images);
    double qz = readBinaryLittleEndian<double>(&file_images);
    Eigen::Quaterniond q(qw, qx, qy, qz);
    q.normalize();
    frame.setColmapQcw(q);

    double tx = readBinaryLittleEndian<double>(&file_images);
    double ty = readBinaryLittleEndian<double>(&file_images);
    double tz = readBinaryLittleEndian<double>(&file_images);
    Eigen::Vector3d t(tx, ty, tz);
    frame.setColmapPcw(t);

    frame.setCameraId(readBinaryLittleEndian<uint32_t>(&file_images));
    size_t cam_idx = static_cast<size_t>(frame.cam_id) - 1;
    uint64_t w = cameras[cam_idx].w;
    uint64_t h = cameras[cam_idx].h;
    frame.setWidth(w);
    frame.setHeight(h);

    frame.setGridCellSize(grid_cell_size);
    frame.initializeGrid();

    char name_char;
    do {
      file_images.read(&name_char, 1);
      if (name_char != '\0') {
        frame.name += name_char;
      }
    } while (name_char != '\0');

    const size_t num_potential_points2D = readBinaryLittleEndian<uint64_t>(&file_images);

    /* We noted that:
    1- COLMAP sometimes saves 2d points twice.
    The corresponding 3d points are not exactly the same but very similar 
    (norm difference in the order of sub-mm).
    We discard one of the two points since they would introduce unecessary redundancy 
    in the optimization.
    2- There are usually multiple 2d points on an image that correspond to the same 3d point.
    E.g.: point 1 = (383.38, 18.55) and point 2 = (385.93, 18.97) both reproject to the 
    same 3d point.
    In a similr case, we would only include point 1 on the image. 
    This is done by checking that the 3D point id has not already been included.*/
    std::vector<Eigen::Vector2d> points2D;
    std::vector<uint32_t> point2D_ids;
    std::vector<uint64_t> point3D_ids;

    /* We store x and y coordinates as key in a harsh table (i.e., unordered_map).
    The value is the 2d point index as in the vector points2D.
    In this way, we can efficiently evaluate if the 2d point has already been stored.
    */
    std::unordered_map<double, size_t> x_coords;
    std::unordered_map<double, size_t> y_coords;

    for (size_t j = 0; j < num_potential_points2D; ++j) 
    {
        const double x = readBinaryLittleEndian<double>(&file_images);
        const double y = readBinaryLittleEndian<double>(&file_images);
        const uint64_t point3D_id = readBinaryLittleEndian<uint64_t>(&file_images);

        if (point3D_id == kInvalidPoint3DId) { continue; }

        // See note 1 above
        auto search_x = x_coords.find(x);
        auto search_y = y_coords.find(y);     
        if ( (search_x != x_coords.end()) && (search_y != y_coords.end()) && 
        (search_x->second == search_y->second) )
        {
            //LOG(INFO) << "A point 2D on image " << im_id 
            //<< " has been discarded because duplicate!";
            continue;
        }

        // See note 2 above
        std::vector<uint64_t>::iterator it = std::find(point3D_ids.begin(), point3D_ids.end(), point3D_id);
        if (it != point3D_ids.end())
        {
            //LOG(INFO) << "A point 2D on image " << im_id 
            //<< " has been discarded because projects to an already seen 3D point!";
            continue;
        }
        
        points2D.emplace_back(x, y);
        point2D_ids.emplace_back(static_cast<uint32_t>(j));
        point3D_ids.emplace_back(point3D_id);

        x_coords.insert_or_assign(x, j);
        y_coords.insert_or_assign(y, j);
    }
    /*LOG(INFO) << "On image " << im_id << ", " 
    << (num_potential_points2D - points2D.size()) << " 2D points 2D have been discarded!";*/

    for (size_t point2D_idx = 0; point2D_idx < points2D.size(); ++point2D_idx) 
    {
        frame.addPoint2D(point2D_ids[point2D_idx], points2D[point2D_idx], point3D_ids[point2D_idx]);
        frame.addPoint3DId(point3D_ids[point2D_idx]);
    }

    frames.emplace_back(frame);
  }
  file_images.close();

  std::ifstream file_points3d(path_points3d, std::ios::binary);
  CHECK(file_points3d.is_open()) << path_points3d;

  const size_t num_points3D = readBinaryLittleEndian<uint64_t>(&file_points3d);

  for (size_t i = 0; i < num_points3D; ++i) 
  {
    Point3D point3D;

    const uint64_t id = readBinaryLittleEndian<uint64_t>(&file_points3d);
    
    point3D.setId(id);

    double tx = readBinaryLittleEndian<double>(&file_points3d);
    double ty = readBinaryLittleEndian<double>(&file_points3d);
    double tz = readBinaryLittleEndian<double>(&file_points3d);
    Eigen::Vector3d t(tx, ty, tz);
    point3D.setPositionInColmapFrame(t);

    // read r,g,b values
    readBinaryLittleEndian<uint8_t>(&file_points3d);
    readBinaryLittleEndian<uint8_t>(&file_points3d);
    readBinaryLittleEndian<uint8_t>(&file_points3d);
    //double r = readBinaryLittleEndian<uint8_t>(&file_points3d);
    //double g = readBinaryLittleEndian<uint8_t>(&file_points3d);
    //double b = readBinaryLittleEndian<uint8_t>(&file_points3d);
    
    point3D.setError(readBinaryLittleEndian<double>(&file_points3d));

    const size_t track_length = readBinaryLittleEndian<uint64_t>(&file_points3d);
    for (size_t j = 0; j < track_length; ++j) 
    {
        const uint32_t image_id = readBinaryLittleEndian<uint32_t>(&file_points3d);
        const uint32_t point2D_idx = readBinaryLittleEndian<uint32_t>(&file_points3d);

        point3D.addImgId(image_id - 1);
        point3D.addPoint2DId(point2D_idx);
    }

    points3D.insert({id, point3D});
  }
   
}

