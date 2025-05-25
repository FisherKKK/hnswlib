#pragma once

#include <iostream>
#include <memory>
#include <mutex>

#include "pnm_engine_def.h"
#include "pnmesdk_client_c.h"

// #define USE_ALIFLASH 1

struct AliFlashClient {

  static std::shared_ptr<AliFlashClient> GetInstance(size_t dim) {
    static auto client = std::make_shared<AliFlashClient>(dim);
    return client;
  }

  AliFlashClient(size_t dim) {
      config_ = new pnmesdk_conf();
      context_ = new database_context();

      ret = pnmesdk_init(config_);

      context_->data_type = FP32;
      context_->vecdim = dim;
      vecdim_ = dim;
  }

  ~AliFlashClient() {
      std::cout << "pre uinit and context delete" << std::endl;
      pnmesdk_uninit(config_);
      delete context_;
      delete config_;
      std::cout << "uinit and context delete" << std::endl;
  }

  void open() {
#ifdef MULTICASE
      static bool init = false;
      if (init) return;
#endif
      database_id_ = pnmesdk_db_open(context_, database_id_, database_name_);
      if (database_id_ == 0) {
          std::cout << "open or create new database failed\n";
          exit(1);
      }
#ifdef MULTICASE
      init = true;
#endif
  }

  void upload(float *base_data, size_t vecsize) {
#ifdef MULTICASE
      static bool init = false;
      if (init) return;
#endif

      vecsize_ = vecsize;

      std::cout << "Upload ready, " << "vecsize: " << vecsize_
                << ", vecdim: " << vecdim_
                << std::endl;

      // block size
      size_t offset = BLOCK_SIZE / (vecdim_ * FP32);
      std::cout << "Offset size: " << std::dec << offset << std::endl;

      // fold upload
      size_t i;
      for (i = 0; i + offset <= vecsize_; i += offset) {
          std::cout << "Offset #" << i << std::endl;
          ret = pnmesdk_db_storage(context_,
                                  (char*)(base_data + i * vecdim_),
                                  offset * vecdim_ * FP32);
      }

      // tail
      if (i < vecsize_) {
          std::cout << "Deal with tail: " << vecsize_ - i << std::endl;
          ret = pnmesdk_db_storage(context_, (char*)(base_data + i * vecdim_), (vecsize_ - i) * vecdim_ * FP32);
      }

      if (ret) {
          std::cout << "pnmesdk_db_storage failed, [data upload failed] ret = "
                    << ret << std::endl;
          exit(1);
      }

      std::cout << "Upload succeed" << std::endl;
#ifdef MULTICASE
      init = true;
#endif
  }

  void cal_single(void *query, uint64_t single_id, float *dist, int hnsw_query_id) {
    calculate_config calc_config;
    calc_config.target_vector = query;
    calc_config.target_vector_size = vecdim_ * FP32;
    calc_config.ids_list = &single_id;
    calc_config.ids_size = 1;
    calc_config.result_list = dist;
    calc_config.hnsw_query_id = hnsw_query_id;
    ret = database_context_cal(context_, &calc_config);
    if (ret) {
        printf("current vector cal task execute failed, ret = %d\n", ret);
        exit(1);
    }
  }

  void cal_multi(void *query, uint64_t* ids, float *dist, uint32_t size, int hnsw_query_id) {
    calculate_config calc_config;
    calc_config.target_vector = query;
    calc_config.target_vector_size = vecdim_ * FP32;
    calc_config.ids_list = ids;
    calc_config.ids_size = size;
    calc_config.result_list = dist;
    calc_config.hnsw_query_id = hnsw_query_id;
    ret = database_context_cal(context_, &calc_config);
    if (ret) {
      printf("current vector cal task execute failed, ret = %d\n", ret);
      exit(1);
    }
  }


  int begin_single() {
    // std::unique_lock<std::mutex> lk(mutex_);
    return pnme_get_search_query_id();
  }

  void end_single(int query_id) {
    // std::unique_lock<std::mutex> lk(mutex_);
    pnme_hnsw_search_end(query_id);
  }


  int ret = 0;
  database_context* context_ = nullptr;
  pnmesdk_conf *config_ = nullptr;
  uint64_t database_id_ = 0;
  char* database_name_ = "ant-hnsw";
  size_t vecsize_ = 0;
  size_t vecdim_ = 0;
  std::mutex mutex_;

  constexpr static uint64_t BLOCK_SIZE = 100000UL * 1024 * 4;
   // 1 * 1024 * 1024 * 1024; // bytes
};
