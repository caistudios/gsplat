#ifndef GSPLAT_CUDA_HELPERS_H
#define GSPLAT_CUDA_HELPERS_H

#include "types.cuh"

#include <cooperative_groups.h>
#ifdef USE_ROCM
#include <hip/amd_detail/amd_hip_cooperative_groups_reduce.h>
#else
#include <cooperative_groups/reduce.h>
#endif

#include <ATen/Dispatch.h>
#include <ATen/cuda/Atomic.cuh>

#ifdef USE_ROCM
// HIP's cooperative groups have no labeled_partition. The backward kernels
// only use it to pre-reduce gradients across lanes that share a gaussian /
// camera id before one lane does the atomicAdd. Fall back to a degenerate
// "group" of a single thread: no reduction, every lane does its own
// atomicAdd. Slower, but numerically equivalent.
namespace cooperative_groups {
struct gsplat_single_thread_group {
    __device__ unsigned int thread_rank() const { return 0; }
    __device__ unsigned int size() const { return 1; }
};
template <class GroupT, class LabelT>
__device__ inline gsplat_single_thread_group
labeled_partition(const GroupT &, LabelT) {
    return gsplat_single_thread_group();
}
} // namespace cooperative_groups
#endif

namespace gsplat {

namespace cg = cooperative_groups;

#ifdef USE_ROCM
// Sum over a full wave32 (all 32 lanes active); every lane gets the total. DPP adds inside each 16-lane row, then
// the two rows swap. HIP's cg::reduce on a thread_block_tile is a generic loop of ds_bpermute with exec-mask
// bookkeeping, several times slower, and this kernel does nine of them per Gaussian and warp.
__device__ __forceinline__ float wave32_sum(float v) {
    v += __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), 0xB1, 0xF, 0xF, true));  // quad_perm 1,0,3,2
    v += __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), 0x4E, 0xF, 0xF, true));  // quad_perm 2,3,0,1
    v += __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), 0x141, 0xF, 0xF, true)); // row_half_mirror
    v += __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), 0x140, 0xF, 0xF, true)); // row_mirror
    v += __int_as_float(__builtin_amdgcn_permlanex16(0, __float_as_int(v), 0x76543210, 0xfedcba98, true, false));
    return v;
}
#endif

template <class WarpT, class T>
inline __device__ T groupSum(WarpT &warp, T val) {
    return cg::reduce(warp, val, cg::plus<T>());
}

#ifdef USE_ROCM
template <class T>
inline __device__ T groupSum(cg::gsplat_single_thread_group &, T val) {
    return val;
}
#endif

template <uint32_t DIM, class T, class WarpT>
inline __device__ void warpSum(T *val, WarpT &warp) {
    GSPLAT_PRAGMA_UNROLL
    for (uint32_t i = 0; i < DIM; i++) {
        val[i] = groupSum(warp, val[i]);
    }
}

template <class WarpT, class ScalarT>
inline __device__ void warpSum(ScalarT &val, WarpT &warp) {
    val = groupSum(warp, val);
}

template <class WarpT, class ScalarT>
inline __device__ void warpSum(vec4<ScalarT> &val, WarpT &warp) {
    val.x = groupSum(warp, val.x);
    val.y = groupSum(warp, val.y);
    val.z = groupSum(warp, val.z);
    val.w = groupSum(warp, val.w);
}

template <class WarpT, class ScalarT>
inline __device__ void warpSum(vec3<ScalarT> &val, WarpT &warp) {
    val.x = groupSum(warp, val.x);
    val.y = groupSum(warp, val.y);
    val.z = groupSum(warp, val.z);
}

template <class WarpT, class ScalarT>
inline __device__ void warpSum(vec2<ScalarT> &val, WarpT &warp) {
    val.x = groupSum(warp, val.x);
    val.y = groupSum(warp, val.y);
}

template <class WarpT, class ScalarT>
inline __device__ void warpSum(mat4<ScalarT> &val, WarpT &warp) {
    warpSum(val[0], warp);
    warpSum(val[1], warp);
    warpSum(val[2], warp);
    warpSum(val[3], warp);
}

template <class WarpT, class ScalarT>
inline __device__ void warpSum(mat3<ScalarT> &val, WarpT &warp) {
    warpSum(val[0], warp);
    warpSum(val[1], warp);
    warpSum(val[2], warp);
}

template <class WarpT, class ScalarT>
inline __device__ void warpSum(mat2<ScalarT> &val, WarpT &warp) {
    warpSum(val[0], warp);
    warpSum(val[1], warp);
}

template <class WarpT, class ScalarT>
inline __device__ void warpMax(ScalarT &val, WarpT &warp) {
    val = cg::reduce(warp, val, cg::greater<ScalarT>());
}

template <typename T> __forceinline__ __device__ T sum(vec3<T> a) {
    return a.x + a.y + a.z;
}

} // namespace gsplat

#endif // GSPLAT_CUDA_HELPERS_H
