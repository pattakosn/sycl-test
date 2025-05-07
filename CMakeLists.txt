cmake_minimum_required(VERSION 3.23)

#en set(CMAKE_C_COMPILER icx)
set(CMAKE_CXX_COMPILER icpx)
#message("++NIKOS++ cxx compiler id:" ${CMAKE_CXX_COMPILER_ID})

project(sycl-tests VERSION 0.1 DESCRIPTION "SyCL tests playground" LANGUAGES CXX)

find_package(IntelSYCL REQUIRED)

macro(exe_from_single_source BASE_FILE_NAME)
    add_executable(${BASE_FILE_NAME}
        ${CMAKE_CURRENT_SOURCE_DIR}/${BASE_FILE_NAME}.cpp
        #PROPERTIES
        #ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
        #LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
        #RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
    if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
        target_compile_options(${BASE_FILE_NAME} PRIVATE /debug:all)
        target_link_options(${BASE_FILE_NAME} PRIVATE /debug:all /Zi)
    endif()
    add_sycl_to_target(TARGET ${BASE_FILE_NAME} SOURCES ${BASE_FILE_NAME}.cpp)
endmacro()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
#set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
#set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_subdirectory(src)