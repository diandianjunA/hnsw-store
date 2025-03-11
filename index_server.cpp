#include "gpu_mem_util.h"
#include "index_node.h"
#include <chrono>
#include <iostream>

int main() {
    int dim = 128;
    int num_data = 1000;
    IndexNode *index_node = new IndexNode(dim, num_data, "192.168.6.201");
    index_node->connect();

    index_node->init_gpu();

    int num_query = 1;
    int k = 5;
    std::vector<float> query(num_query * dim);
    srand(time(nullptr));
    for (int j = 0; j < num_query; ++j) {
        for (int k = 0; k < dim; ++k) {
            query[j * dim + k] = static_cast<float>(rand()) / RAND_MAX;
        }
    }
    auto result = index_node->search_vectors(query, k);

    // 输出查询结果，包括id和距离
    for (int i = 0; i < num_query; ++i) {
        std::cout << "Query " << i << std::endl;
        for (int j = 0; j < k; ++j) {
            std::cout << "id: " << result.first[i * k + j] << ", distance: " << result.second[i * k + j] << std::endl;
        }
    }
    
    return 0;
}