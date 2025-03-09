#include "include/rdma_service.h"

int main() {
    RdmaService *rdma_service = new RdmaService(nullptr, 70000000);
    rdma_service->resources_create();
    char* buf = rdma_service->get_buf();
    memcpy(buf, "Hello World!", 12);
    rdma_service->connect_qp();

    getchar();

    delete rdma_service;
    return 0;
}