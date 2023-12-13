add_library(my_axpy_sycl
            ${CMAKE_CURRENT_SOURCE_DIR}/my_axpy_sycl.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/my_axpy_sycl.h
)
add_sycl_to_target(TARGET my_axpy_sycl SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/my_axpy_sycl.cpp)
set_target_properties(my_axpy_sycl PROPERTIES PUBLIC_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/my_axpy_sycl.h)
set_target_properties(my_axpy_sycl PROPERTIES VERSION ${PROJECT_VERSION})
set_target_properties(my_axpy_sycl PROPERTIES SOVERSION ${PROJECT_VERSION_MAJOR}) # symlink libfoo.so.1 to libfoo.so.1.0.0
target_include_directories(my_axpy_sycl PRIVATE ..)

include(GNUInstallDirs)
configure_file(my_axpy_sycl.pc.in mylib.pc @ONLY)
install(TARGETS my_axpy_sycl
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES ${CMAKE_BINARY_DIR}/mylib.pc DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/pkgconfig)
