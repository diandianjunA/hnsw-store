#include "index_node.h"
#include "cuda/search_kernel.cuh"
#include <iostream>

IndexNode::IndexNode(int dim, int num_data, char* server_name, int M, int ef_construction,
                     int memory_size) {
    this->dim = dim;
    this->num_data = num_data;
    this->M = M;
    this->ef_construction = ef_construction;

    data_size = dim * sizeof(float);
    size_data_per_element_ =
        4 + M * sizeof(unsigned int) + dim * sizeof(float) + sizeof(size_t);
    offset_data = 4 + M * sizeof(unsigned int);

    if (memory_size == 0) {
        memory_size =
            num_data * size_data_per_element_;
    }
    this->rdma_service_ = new RdmaService(server_name, memory_size);
    rdma_service_->resources_create();
}

IndexNode::~IndexNode() { delete rdma_service_; }

void IndexNode::init_gpu() {

    cuda_init(dim, rdma_service_->get_buf(), size_data_per_element_,
              offset_data, M, 50, num_data, data_size);
}

void IndexNode::check() {}

std::pair<std::vector<int>, std::vector<float>>
IndexNode::search_vectors(const std::vector<float> &query, int k,
                          int ef_search) {
    int num_query = query.size() / dim;
    std::vector<int> inner_index(k * num_query);
    std::vector<float> distances(k * num_query);
    std::vector<int> found_cnt(num_query);

    cuda_search(106, query.data(), num_query, ef_search, k, inner_index.data(),
                distances.data(), found_cnt.data());

    return {inner_index, distances};
}

void IndexNode::connect() {
    rdma_service_->connect_qp(); 
    rdma_service_->post_receive();
    rdma_service_->poll_completion();
}

void IndexNode::sync() {
    if (rdma_service_->post_send(IBV_WR_RDMA_READ)) {
        std::cerr << "Failed to post send" << std::endl;
    } 
    if (rdma_service_->poll_completion()) {
        std::cerr << "Failed to poll completion" << std::endl;
    }
}