/* This file is part of RPG vision-based SLAM.
Copyright (C) 2022 Giovanni Cioffi <cioffi at ifi dot uzh dot ch>
(Robotics and Perception Group, University of Zurich, Switzerland).

This file is subject to the terms and conditions defined in the file
'LICENSE', which is part of this source code package.
*/

/*
This script is based on:
https://github.com/uzh-rpg/rpg_svo_pro/blob/feature/global_map/svo_ceres_backend/include/svo/ceres_backend/error_interface.hpp
*/

#pragma once

#include <map>


enum class ErrorType: uint8_t
{
  ReprojectionError,
  SpeedAndBiasError,
  kPoseError,
  IMUError
};

extern const std::map<ErrorType, std::string> kErrorToStr;

/// @brief Simple interface class the errors implemented here should inherit from.
class ErrorInterface
{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /// @brief Constructor
  ErrorInterface() = default;
  /// @brief Destructor (does nothing).
  virtual ~ErrorInterface() = default;

  /// @name Sizes
  /// @{

  /// @brief Get dimension of residuals.
  /// @return The residual dimension.
  virtual size_t residualDim() const = 0;

  /// @brief Get the number of parameter blocks this is connected to.
  /// @return The number of parameter blocks.
  virtual size_t parameterBlocks() const = 0;

  /**
   * @brief get the dimension of a parameter block this is connected to.
   * @param parameter_block_idx The index of the parameter block of interest.
   * @return Its dimension.
   */
  virtual size_t parameterBlockDim(size_t parameter_block_idx) const = 0;

  /// @}
  // Error and Jacobian computation
  /**
   * @brief This evaluates the error term and additionally computes
   *        the Jacobians in the minimal internal representation.
   * @param parameters Pointer to the parameters (see ceres)
   * @param residuals Pointer to the residual vector (see ceres)
   * @param jacobians Pointer to the Jacobians (see ceres)
   * @param jacobians_minimal Pointer to the minimal Jacobians (equivalent to jacobians).
   * @return Success of the evaluation.
   */
  virtual bool EvaluateWithMinimalJacobians(
      double const* const * parameters, double* residuals, double** jacobians,
      double** jacobians_minimal) const = 0;

  /// @brief Residual block type as string
  virtual ErrorType typeInfo() const = 0;
};

