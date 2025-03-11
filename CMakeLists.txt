cmake_minimum_required(VERSION 3.2) #设置cmake的最低版本
project(hnsw_store LANGUAGES CXX C) #设置项目名称 vdb
 
# 设置C++标准  基本不用改
set(CMAKE_CXX_STANDARD 17)  
set(CMAKE_CXX_STANDARD_REQUIRED True)  


include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

FIND_PACKAGE(OpenMP REQUIRED)
if(OPENMP_FOUND)
message("OPENMP FOUND")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

add_subdirectory(cuda)
set (EXTRA_LIBS ${EXTRA_LIBS} gpu)

set (CMAKE_CXX_LINK_EXECUTABLE "${CMAKE_CXX_LINK_EXECUTABLE} -ldl") 

# file自定义搜索源文件，塞给集合SOURCES 
file(GLOB INDEX_SERVER_SOURCES rdma/*.cpp gpu_utils/*.c node/index_node.cpp)
file(GLOB STORAGE_SERVER_SOURCES rdma/*.cpp gpu_utils/*.c node/storage_node.cpp)

include_directories("/usr/local/include/")
include_directories("/usr/local/cuda-12.5/targets/x86_64-linux/include")
include_directories("/usr/local/cuda-12.2/targets/x86_64-linux/include")
link_directories("./ThirdParty/OpenCV/lib")
link_directories("/usr/lib/x86_64-linux-gnu")
link_directories("/usr/local/lib")
link_directories("~/anaconda3/lib")
link_directories("./lib")
include_directories("/usr/local/cuda-12.5/lib64")

add_executable(index_server index_server.cpp ${INDEX_SERVER_SOURCES})
add_executable(storage_server storage_server.cpp ${STORAGE_SERVER_SOURCES})

target_link_libraries(index_server PRIVATE ${EXTRA_LIBS}
    ibverbs cuda
)
target_link_libraries(storage_server PRIVATE
    ibverbs cuda rocksdb snappy z bz2 zstd lz4
)