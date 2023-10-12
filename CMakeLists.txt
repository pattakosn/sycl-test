cmake_minimum_required(VERSION 3.23)

project(sycl-tests VERSION 0.1 DESCRIPTION "SyCL tests playground" LANGUAGES CXX)

find_package(IntelSYCL REQUIRED)

macro(exe_from_single_source BASE_FILE_NAME)
    add_executable(${BASE_FILE_NAME} ${BASE_FILE_NAME}.cpp)
    add_sycl_to_target(TARGET ${BASE_FILE_NAME} SOURCES ${BASE_FILE_NAME}.cpp)
endmacro()

exe_from_single_source(hello_world)
exe_from_single_source(sycl_device)
