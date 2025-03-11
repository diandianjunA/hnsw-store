#include <cstdint>
#include <vector>
#include "rdma_service.h"

class IndexNode {
  public:
    // 构造函数
    IndexNode(int dim, int num_data, char* server_name, int M = 32, int ef_construction = 200, int memory_size = 1024 * 1024);
    ~IndexNode();

    void connect();
    void sync();

    void init_gpu();

    void check();

    // 查询向量
    std::pair<std::vector<int>, std::vector<float>>
    search_vectors(const std::vector<float> &query, int k, int ef_search = 50);

  private:
    int dim;
    int num_data;
    int M;
    int ef_construction;
    RdmaService *rdma_service_;
    size_t data_size;
    size_t size_data_per_element_;
    size_t offset_data;
};
