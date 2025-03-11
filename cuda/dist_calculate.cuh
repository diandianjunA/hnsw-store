#pragma once

#define mul(x, y) ( x * y )
#define add(x, y) ( x + y )
#define sub(x, y) ( x - y )
#define gt(x, y) ( x > y )
#define ge(x, y) ( x >= y )
#define lt(x, y) ( x < y )
#define le(x, y) ( x <= y )
#define out_scalar(x) ( x )
#define conversion(x) ( x )

#define WARP_SIZE 32

__inline__ __device__
float warp_reduce_sum(float val) {
  #if __CUDACC_VER_MAJOR__ >= 9
  // __shfl_down is deprecated with cuda 9+. use newer variants
  unsigned int active = __activemask();
  #pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
      val = add(val, __shfl_down_sync(active, val, offset));
  }
  #else
  #pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
      val = add(val, __shfl_down(val, offset));
  }
  #endif
  return val;
}

__inline__ __device__
float squaresum(const float * a, const float * b, const int num_dims) {
  __syncthreads();
  static __shared__ float shared[32];

  // figure out the warp/ position inside the warp
  int warp =  threadIdx.x / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;

  // partially reduce the dot product inside each warp using a shuffle
  float val = 0;
  for (int i = threadIdx.x; i < num_dims; i += blockDim.x) {
    float _val = sub(a[i], b[i]);
    val = add(val, mul(_val, _val));
  }
  __syncthreads();
  val = warp_reduce_sum(val);

  // write out the partial reduction to shared memory if appropiate
  if (lane == 0) {
    shared[warp] = val;
  }
  __syncthreads();

  // if we we don't have multiple warps, we're done
  if (blockDim.x <= WARP_SIZE) {
    return shared[0];
  }

  // otherwise reduce again in the first warp
  val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane]: conversion(0.0f);
  if (warp == 0) {
    val = warp_reduce_sum(val);
    // broadcast back to shared memory
    if (threadIdx.x == 0) {
        shared[0] = val;
    }
  }
  __syncthreads();
  return shared[0];
}

__inline__ __device__
float GetDistanceByVec(const float* src_vec, const float* dst_vec, const int num_dims) {
  float dist = 0;
  dist = squaresum(src_vec, dst_vec, num_dims);
  return dist;
}