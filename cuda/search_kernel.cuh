#pragma once

void cuda_init(int dims_, char *data_, size_t size_data_per_element_,
               size_t offsetData_, int max_m_, int ef_search_, int num_data_,
               size_t data_size_);

void cuda_search(int entry_node, const float *query_data, int num_query,
                 int ef_search_, int k, int *nns, float *distances,
                 int *found_cnt);

void cuda_search_hierarchical(int entry_node, const float *query_data,
                              int num_query, int ef_search_, int k, int *nns,
                              float *distances, int *found_cnt);