// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "multiply.hpp"

#include <cuda_operation_registry.hpp>

namespace CUDAPlugin {

MultiplyOp::MultiplyOp(const CUDA::Device& device, const std::shared_ptr<ngraph::Node>& node,
                       IndexCollection&& inputIds, IndexCollection&& outputIds)
    : CuDnnTensorOpBase{device, node, move(inputIds), move(outputIds),
                         cudnnOpTensorOp_t::CUDNN_OP_TENSOR_MUL} {}

OPERATION_REGISTER(MultiplyOp, Multiply);

}  // namespace CUDAPlugin
