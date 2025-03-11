#include "storage_node.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"
#include "constant.h"
#include <thread>
#include <iostream>

StorageNode::StorageNode(const std::string& db_path, int dim, int num_data, int M, int ef_construction, int) : dim(dim) {
    // rocksdb::DB* db;
    // rocksdb::Options options;
    // options.create_if_missing = true;
    // rocksdb::Status status = rocksdb::DB::Open(options, db_path, &db);
    // if (!status.ok()) {
    //     throw std::runtime_error("rocksdb open error");
    // }
    // db_ = db;
    space = new hnswlib::L2Space(dim);
    index = new hnswlib::HierarchicalNSW<float>(space, num_data, M, ef_construction);
    rdma_service_ = new RdmaService(nullptr, index->max_elements_ * index->size_data_per_element_, index->data_level0_memory_);
    rdma_service_->resources_create();
}

StorageNode::~StorageNode() {
    // delete db_;
    delete rdma_service_;
}

void StorageNode::insert(long id, const rapidjson::Document& data) {
    const rapidjson::Value& object = data[REQUEST_OBJECT];
    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    object.Accept(writer);
    std::string value = buffer.GetString();
    db_->Put(rocksdb::WriteOptions(), std::to_string(id), value);
}
    
    
rapidjson::Document StorageNode::query(long id) {
    std::string value;
    db_->Get(rocksdb::ReadOptions(), std::to_string(id), &value);
    rapidjson::Document data;
    data.Parse(value.c_str());
    return data;
}

void StorageNode::insert_batch(std::vector<long> ids, const rapidjson::Document& data) {
    const rapidjson::Value& objects = data[REQUEST_OBJECTS];
    if (!objects.IsArray()) {
        throw std::runtime_error("objects type not match");
    }
    
    for (int i = 0; i < ids.size(); i++) {
        const rapidjson::Value& row = objects[i];
        long id = ids[i];
        rapidjson::StringBuffer buffer;
        rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
        row.Accept(writer);
        std::string value = buffer.GetString();
        db_->Put(rocksdb::WriteOptions(), std::to_string(id), value);
    }
}

template<class Function>
inline void ParallelFor(size_t start, size_t end, size_t numThreads, Function fn) {
    if (numThreads <= 0) {
        numThreads = std::thread::hardware_concurrency();
    }

    if (numThreads == 1) {
        for (size_t id = start; id < end; id++) {
            fn(id, 0);
        }
    } else {
        std::vector<std::thread> threads;
        std::atomic<size_t> current(start);

        // keep track of exceptions in threads
        // https://stackoverflow.com/a/32428427/1713196
        std::exception_ptr lastException = nullptr;
        std::mutex lastExceptMutex;

        for (size_t threadId = 0; threadId < numThreads; ++threadId) {
            threads.push_back(std::thread([&, threadId] {
                while (true) {
                    size_t id = current.fetch_add(1);

                    if (id >= end) {
                        break;
                    }

                    try {
                        fn(id, threadId);
                    } catch (...) {
                        std::unique_lock<std::mutex> lastExcepLock(lastExceptMutex);
                        lastException = std::current_exception();
                        /*
                         * This will work even when current is the largest value that
                         * size_t can fit, because fetch_add returns the previous value
                         * before the increment (what will result in overflow
                         * and produce 0 instead of current + 1).
                         */
                        current = end;
                        break;
                    }
                }
            }));
        }
        for (auto &thread : threads) {
            thread.join();
        }
        if (lastException) {
            std::rethrow_exception(lastException);
        }
    }
}

void StorageNode::insert_vectors(const float* data, uint64_t label) {
    index->addPoint(data, label);
}

void StorageNode::insert_vectors_batch(const std::vector<float>& data, const std::vector<uint64_t>& ids) {
    ParallelFor(0, ids.size(), 0, [&](size_t i, size_t threadId) {
        insert_vectors(data.data() + i * dim, ids[i]);
    });
}

void StorageNode::connect() {
    std::cout << "size_data_per_element_: " << index->size_data_per_element_ << std::endl;
    std::cout << "offset_data: " << index->offsetData_ << std::endl;
    std::cout << "data_size_: " << index->data_size_ << std::endl;
    std::cout << "entry_point_node_: " << index->enterpoint_node_ << std::endl;

    rdma_service_->connect_qp();
    rdma_service_->post_send(IBV_WR_SEND);
    rdma_service_->poll_completion();
}