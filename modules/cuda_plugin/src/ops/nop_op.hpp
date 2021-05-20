// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda_operation_base.hpp>

namespace CUDAPlugin {

/**
 * @brief NOP - no operation. Common implementation for all operations which
 * do nothing.
 *
 * These operations are at least the following: Reshape, Squeeze, Unsqueeze,
 * Constant.
 *
 * The purpose of having NOP operations in execution queue is to make them
 * transparent for the rest of plugin implementation, so they don't require
 * special handling to skip their execution.
 *
 * Note, that Reshape-like operations do not need to perform any data copying
 * because their input and output data tensors reuse the same memory allocation.
 * Constants also have nothing to do, because at the time of execution their
 * values are already copied to device side and linked with all dependent
 * consumer operations.
 */
class NopOp : public OperationBase {
public:
  using OperationBase::OperationBase;

  gsl::span<const unsigned> GetInputIds() const override {
    return gsl::span<const unsigned> {};
  };

  gsl::span<const unsigned> GetOutputIds() const override {
    return gsl::span<const unsigned> {};
  };

  void Execute(const InferenceRequestContext& context,
               Inputs inputTensors,
               Outputs outputTensors) override {}
};

} // namespace CUDAPlugin
