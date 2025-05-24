find_package(OpenMP REQUIRED)

set (EVAL_SRC
        case/eval_case.cpp
        case/search_eval_case.cpp
        case/build_eval_case.cpp
        eval_config.cpp
        eval_dataset.cpp
        eval_job.cpp
        monitor/monitor.cpp
        monitor/latency_monitor.cpp
        monitor/recall_monitor.cpp
        monitor/memory_peak_monitor.cpp
        monitor/duration_monitor.cpp)
add_library (eval_obj OBJECT ${EVAL_SRC})
target_compile_options (eval_obj PRIVATE -fopenmp)
add_dependencies (eval_obj hdf5 spdlog yaml-cpp)

add_executable (eval_performance main.cpp)
target_compile_options (eval_performance PRIVATE -fopenmp)
target_link_libraries (eval_performance
  PRIVATE
  eval_obj
  yaml-cpp
  argparse::argparse
  tabulate
  libhdf5_cpp.a
  libhdf5.a
  nlohmann_json::nlohmann_json
  z
  OpenMP::OpenMP_CXX
)
