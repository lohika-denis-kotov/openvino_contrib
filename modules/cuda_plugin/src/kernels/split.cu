// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cuda.h>
#if CUDA_VERSION >= 11000
#include <cuda_bf16.h>
#endif
#include <cuda_fp16.h>
#include <fmt/format.h>

#include <gsl/gsl_assert>

#include "split.hpp"

namespace CUDAPlugin {
namespace kernel {

template <typename T>
static __global__ void split(
    const size_t numSplitChunks, const size_t splitStepSize, const size_t numSplits, const T *x, T **y) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned chunkIdx = i / splitStepSize;
    const unsigned chunkOffset = i % splitStepSize;
    if (chunkIdx < numSplitChunks) {
        const unsigned splitIdx = chunkIdx % numSplits;
        const unsigned splitStepIdx = chunkIdx / numSplits;
        auto src = &x[chunkIdx * splitStepSize];
        auto dest = &y[splitIdx][splitStepIdx * splitStepSize];
        dest[chunkOffset] = src[chunkOffset];
    }
}

Split::Split(Type_t element_type,
             size_t num_splits,
             size_t num_split_chunks,
             size_t split_step_size,
             unsigned num_blocks,
             unsigned threads_per_block)
    : element_type_{element_type},
      num_splits_{num_splits},
      num_split_chunks_{num_split_chunks},
      split_step_size_{split_step_size},
      num_blocks_{num_blocks},
      threads_per_block_{threads_per_block} {}

void Split::operator()(cudaStream_t stream, const void *src, void **dst) const {
    switch (element_type_) {
        case Type_t::boolean:
            return Call<bool>(stream, src, dst);
#if CUDA_VERSION >= 11000
        case Type_t::bf16:
            return Call<__nv_bfloat16>(stream, src, dst);
#endif
        case Type_t::f16:
            return Call<__half>(stream, src, dst);
        case Type_t::f32:
            return Call<float>(stream, src, dst);
        case Type_t::f64:
            return Call<double>(stream, src, dst);
        case Type_t::i8:
            return Call<int8_t>(stream, src, dst);
        case Type_t::i16:
            return Call<int16_t>(stream, src, dst);
        case Type_t::i32:
            return Call<int32_t>(stream, src, dst);
        case Type_t::i64:
            return Call<int64_t>(stream, src, dst);
        case Type_t::u8:
            return Call<uint8_t>(stream, src, dst);
        case Type_t::u16:
            return Call<uint16_t>(stream, src, dst);
        case Type_t::u32:
            return Call<uint32_t>(stream, src, dst);
        case Type_t::u64:
            return Call<uint64_t>(stream, src, dst);
        default:
            throwIEException(
                fmt::format("Input element type = {} is not supported by Split operation "
                            "!!",
                            static_cast<Type_t>(element_type_)));
    }
}

template <typename T>
void Split::Call(cudaStream_t stream, const void *src, void **dst) const {
    split<T><<<num_blocks_, threads_per_block_, 0, stream>>>(
        num_split_chunks_, split_step_size_, num_splits_, static_cast<const T *>(src), reinterpret_cast<T **>(dst));
}

}  // namespace kernel
}  // namespace CUDAPlugin
