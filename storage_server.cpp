#include "storage_node.h"

// 生成指定大小的随机128维浮点向量
void generate_random_vectors(size_t num_vectors, size_t dimension,
                             std::vector<float> &vectors,
                             std::vector<size_t> &ids) {
    // 随机数生成器
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0); // 随机数范围 -1 to 1

    // 生成随机向量和ID
    vectors.resize(num_vectors * dimension);
    ids.resize(num_vectors);

    for (size_t i = 0; i < num_vectors; ++i) {
        ids[i] = i; // ID设置为序号
        for (size_t j = 0; j < dimension; ++j) {
            vectors[i * dimension + j] = dis(gen); // 每个维度生成一个随机数
        }
    }
}

int main() {
    int dim = 128;
    int num_data = 1000;
    StorageNode *storage_node = new StorageNode("db", dim, num_data);

    std::vector<float> vectors;
    std::vector<size_t> ids;
    generate_random_vectors(num_data, dim, vectors, ids);
    storage_node->insert_vectors_batch(vectors, ids);
    storage_node->connect();

    // generate_random_vectors(num_data, dim, vectors, ids);
    // storage_node->insert_vectors_batch(vectors, ids);

    getchar();
    return 0;
}