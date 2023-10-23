cmake_minimum_required(VERSION 3.26)
project(sycl-tests VERSION 0.1 DESCRIPTION "SyCL tests playground" LANGUAGES CXX)

#set(CMAKE_C_COMPILER icx)
#set(CMAKE_CXX_COMPILER icpx)
#message("++NIKOS++ cxx compiler id:" ${CMAKE_CXX_COMPILER_ID})

find_package(IntelSYCL REQUIRED)

macro(exe_from_single_source BASE_FILE_NAME)
    add_executable(${BASE_FILE_NAME} ${BASE_FILE_NAME}.cpp)
    if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
        target_compile_options(${BASE_FILE_NAME} PRIVATE /debug:all)
        target_link_options(${BASE_FILE_NAME} PRIVATE /debug:all /Zi)
    endif()
    add_sycl_to_target(TARGET ${BASE_FILE_NAME} SOURCES ${BASE_FILE_NAME}.cpp)
endmacro()

exe_from_single_source(hello_world)
exe_from_single_source(sycl_device)
exe_from_single_source(simple_codeplay_sycl_app)
exe_from_single_source(start_computing)
exe_from_single_source(create_and_add_buffers)