
// Copyright 2024-present the vsag project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "./build_eval_case.h"

#include <algorithm>
#include <filesystem>
#include <omp.h>
#include <utility>

#include "../monitor/duration_monitor.h"
#include "../monitor/memory_peak_monitor.h"

namespace vsag::eval {

BuildEvalCase::BuildEvalCase(const std::string& dataset_path,
                             const std::string& index_path,
                             IndexPtr index,
                             EvalConfig config)
    : EvalCase(dataset_path, index_path, index), config_(std::move(config)) {
    this->init_monitors();
}

void
BuildEvalCase::init_monitors() {
    if (config_.enable_memory) {
        auto memory_peak_monitor = std::make_shared<MemoryPeakMonitor>();
        this->monitors_.emplace_back(std::move(memory_peak_monitor));
    }
    if (config_.enable_tps) {
        auto duration_monitor = std::make_shared<DurationMonitor>();
        this->monitors_.emplace_back(std::move(duration_monitor));
    }
}

JsonType
BuildEvalCase::Run() {
    this->do_build();
    std::cout << "Index built succeed" << std::endl;
    this->serialize();
    std::cout << "Index serialized" << std::endl;
    auto result = this->process_result();
    return result;
}
void
BuildEvalCase::do_build() {
    int64_t total_base = this->dataset_ptr_->GetNumberOfBase();
    int64_t dim = this->dataset_ptr_->GetDim();
    std::vector<int64_t> ids(total_base);
    std::iota(ids.begin(), ids.end(), 0);

    //TODO only support float32
    const float *base = (const float*) this->dataset_ptr_->GetTrain();
    for (auto& monitor : monitors_) {
        monitor->Start();
    }

    omp_set_num_threads(this->config_.num_threads_building);
    std::cout << "Using " << this->config_.num_threads_building << " threads build HNSW" << std::endl;

#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < total_base; ++i) {
        index_->addPoint(base + i * dim, ids[i]);
        if (i % (total_base / 10) == 0)
            std::cout << "Index i: " << i << std::endl;
    }


    for (auto& monitor : monitors_) {
        monitor->Record();
        monitor->Stop();
    }
}
void
BuildEvalCase::serialize() {
    std::filesystem::path dir_path(index_path_);
    dir_path = dir_path.parent_path();
    if (!std::filesystem::exists(dir_path)) {
        std::filesystem::create_directories(dir_path);
    }
    std::ofstream outfile(this->index_path_, std::ios::binary);
    this->index_->saveIndex(this->index_path_);
}

JsonType
BuildEvalCase::process_result() {
    JsonType result;
    JsonType eval_result;
    for (auto& monitor : this->monitors_) {
        const auto& one_result = monitor->GetResult();
        EvalCase::MergeJsonType(one_result, eval_result);
    }
    result = eval_result;
    result["tps"] = double(this->dataset_ptr_->GetNumberOfBase()) / double(result["duration(s)"]);
    EvalCase::MergeJsonType(this->basic_info_, result);
    result["index_info"] = JsonType::parse(config_.build_param);
    result["action"] = "build";
    return result;
}

}  // namespace vsag::eval
