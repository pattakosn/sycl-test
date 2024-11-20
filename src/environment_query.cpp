#include <sycl/sycl.hpp>
#include <iostream>

int main() {
    for (const auto& platform : sycl::platform::get_platforms()) {
        std::cout << "Platform: " << platform.get_info<sycl::info::platform::name>() << "\n";
        std::cout << "Vendor: " << platform.get_info<sycl::info::platform::vendor>() << "\n";
        std::cout << "Version: " << platform.get_info<sycl::info::platform::version>() << "\n\n";
    }

    //sycl::queue queue;
    //auto device = queue.get_device();
    //sycl::device device = sycl::device::get_devices()[0]; // Pick the first device.
    for (const auto& device : sycl::device::get_devices()) {
        std::cout << "Device: " << device.get_info<sycl::info::device::name>() << "\n";
        std::cout << "Vendor: " << device.get_info<sycl::info::device::vendor>() << "\n";
        std::cout << "Driver Version: " << device.get_info<sycl::info::device::driver_version>() << "\n";
        std::cout << "Global Memory Size: " << device.get_info<sycl::info::device::global_mem_size>() / 1e6 << " MB\n";
        std::cout << "Local Memory Size: " << device.get_info<sycl::info::device::local_mem_size>() / 1e3 << " KB\n";
        std::cout << "Max Compute Units: " << device.get_info<sycl::info::device::max_compute_units>() << "\n";
        std::cout << "Max Work-group Size: " << device.get_info<sycl::info::device::max_work_group_size>() << "\n";
        std::cout << "Max Clock Frequency: " << device.get_info<sycl::info::device::max_clock_frequency>() << "\n";
        std::cout << "Max Mem Allocation Size: " << device.get_info<sycl::info::device::max_mem_alloc_size>() / 1e6 << " MB\n";
        std::cout << "Native Vector Width Char: " << device.get_info<sycl::info::device::native_vector_width_char>() << "\n";
        std::cout << "Native Vector Width Int: " << device.get_info<sycl::info::device::native_vector_width_int>() << "\n\n";
        if (device.has(sycl::aspect::fp16))
            std::cout << "Device supports half-precision floating-point.\n";
        if (device.has(sycl::aspect::atomic64))
            std::cout << "Device supports 64-bit atomics.\n";
        if (device.has(sycl::aspect::accelerator))
            std::cout << "Device supports accelerator.\n";
    }
    return 0;
}
