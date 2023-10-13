cmake_minimum_required(VERSION 3.27)
project(sycl-tests VERSION 0.1 DESCRIPTION "SyCL tests playground" LANGUAGES CXX)

#set(CMAKE_C_COMPILER icx)
#set(CMAKE_CXX_COMPILER "icpx")

find_package(IntelSYCL REQUIRED)

macro(exe_from_single_source BASE_FILE_NAME)
    add_executable(${BASE_FILE_NAME} ${BASE_FILE_NAME}.cpp)
    add_sycl_to_target(TARGET ${BASE_FILE_NAME} SOURCES ${BASE_FILE_NAME}.cpp)
endmacro()

exe_from_single_source(hello_world)
exe_from_single_source(sycl_device)
exe_from_single_source(simple_codeplay_sycl_app)
