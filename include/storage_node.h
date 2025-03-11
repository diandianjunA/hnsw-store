#include <string>
#include <vector>
#include <rapidjson/document.h>
#include <rocksdb/db.h>
#include "rdma_service.h"
#include "hnswlib/hnswlib.h"

class StorageNode {
  public:
    StorageNode(const std::string& db_path, int dim, int num_data, int M = 16, int ef_construction = 200, int memory_size = 1024 * 1024);
    ~StorageNode();

    void connect();

    void insert(long id, const rapidjson::Document &data);
    void insert_batch(std::vector<long> ids, const rapidjson::Document &data);
    rapidjson::Document query(long id);

    void insert_vectors(const float* data, uint64_t label);
    void insert_vectors_batch(const std::vector<float>& data, const std::vector<uint64_t>& ids);

  private:
    rocksdb::DB *db_;
    RdmaService *rdma_service_;

    int dim;
    hnswlib::SpaceInterface<float>* space;
    hnswlib::HierarchicalNSW<float>* index;
};