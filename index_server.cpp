#include "include/rdma_service.h"
#include "cuda/print.cuh"
#include "gpu_mem_util.h"
#include <chrono>
#include <iostream>

int main() {
    RdmaService *rdma_service = new RdmaService("127.0.0.1", 70000000);
    rdma_service->resources_create();
    rdma_service->connect_qp();

    auto start = std::chrono::high_resolution_clock::now();
    rdma_service->post_send(IBV_WR_RDMA_READ);
    rdma_service->poll_completion();
    auto end = std::chrono::high_resolution_clock::now();
    // 输出间隔时间，单位为纳秒
    std::cout << "Time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() << "ns" << std::endl;

    print_gpu((const char*)rdma_service->get_buf());

    delete rdma_service;
    return 0;
}