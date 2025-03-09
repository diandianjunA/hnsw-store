#ifndef _GPU_MEM_UTIL_H_
#define _GPU_MEM_UTIL_H_

#define HAVE_CUDA

#ifdef HAVE_CUDA
/* "/usr/local/cuda/include/" is added to build include path in the Makefile */
#include "cuda.h"
#endif // HAVE_CUDA

#ifdef __cplusplus
extern "C" {
#endif

struct cuda_memory_ctx {
	int device_id;
	char *device_bus_id;
	CUdevice cuDevice;
	CUcontext cuContext;
	bool use_dmabuf;
	bool use_data_direct;
};

/*
 * Memory allocation on CPU or GPU according to HAVE_CUDA pre-compile option and use_cuda flag
 *
 * returns: a pointer to the allocated buffer or NULL on error
 */
void *work_buffer_alloc(size_t length, int use_cuda, const char *bdf);

/*
 * CPU or GPU memory free, according to HAVE_CUDA pre-compile option and use_cuda flag
 */
void work_buffer_free(void *buff, int use_cuda);


#ifdef __cplusplus
}
#endif

#endif /* _GPU_MEM_UTIL_H_ */
