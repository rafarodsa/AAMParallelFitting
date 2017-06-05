cmake_minimum_required(VERSION 2.6)
project (AAM_OpenMP)

list(APPEND CMAKE_CXX_FLAGS "-std=c++0x -O2")
find_package(OpenCV REQUIRED)

add_subdirectory(src)
add_subdirectory(${PROJECT_SOURCE_DIR}/aamlibrary/src)
