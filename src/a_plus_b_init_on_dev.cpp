#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <cstdio>
#include "my_timer.hpp"

class vector_add;

int main(void)
{
	constexpr size_t N = 1024 * 1024 * 16;
try{
	{
		auto q = sycl::queue{sycl::default_selector_v};
		std::cout << "default device on this system: \"" << q.get_device().get_info<sycl::info::device::name>() << "\"" << std::endl;
	}
	{
		auto q = sycl::queue{sycl::gpu_selector_v};
	 	std::cout << "default GPU device on this system: \"" << q.get_device().get_info<sycl::info::device::name>() << "\"" << std::endl;
	}
	{
		auto q = sycl::queue{sycl::cpu_selector_v};
		std::cout << "default CPU device on this system: \"" << q.get_device().get_info<sycl::info::device::name>() << "\"" << std::endl;
	}
	
	auto q = sycl::queue{sycl::gpu_selector_v};
	std::cout << "\nChosen device: \"" << q.get_device().get_info<sycl::info::device::name>() 
		<< "\", max compute units: " << q.get_device().get_info<sycl::info::device::max_compute_units>()
		<< std::endl;
	
	my_timer time_all;
	sycl::buffer<int> bufA{sycl::range{N}};
	sycl::buffer<int> bufB{sycl::range{N}};
	sycl::buffer<int> bufR{sycl::range{N}};

	time_all.start();
	q.submit([&](sycl::handler &cgh) {
		auto aA = sycl::accessor{bufA, cgh, sycl::write_only, sycl::no_init};
		auto aB = sycl::accessor{bufB, cgh, sycl::write_only, sycl::no_init};
		//auto aR = sycl::accessor{bufR, cgh, sycl::write_only, sycl::no_init};
		cgh.parallel_for(N, [=](sycl::id<1> id) {
			aA[id] = id;
			aB[id] = 2*id;
		});
	}).wait_and_throw();
	time_all.stop("Data initialization: ");
//for (int i =0; i < 1024;i++)
	q.submit([&](sycl::handler &cgh) {
		auto aA = sycl::accessor{bufA, cgh, sycl::read_only};
		auto aB = sycl::accessor{bufB, cgh, sycl::read_only};
		auto aR = sycl::accessor{bufR, cgh, sycl::read_write};
		cgh.parallel_for(N, [=](sycl::id<1> id) {
			aR[id] = aA[id] + aB[id];
		});
	}).wait_and_throw();
	time_all.stop("After crunching: ");
	
	sycl::host_accessor end_A{bufA, sycl::read_only};
	sycl::host_accessor end_B{bufB, sycl::read_only};
	sycl::host_accessor end_R{bufR, sycl::read_only};
	
	for( int i = 0; i < 8*16; i+=16 )//N; i+= (1024*1024))
		std::cout << " a[" << i << "]= " << end_A[i] << "\t+ b[" << i << "]= " << end_B[i] << "\t= c["<< i << "]= " << end_R[i] << std::endl;
} catch(const sycl::exception &e) {
 	std::cout << "SyCL exception caught: " << e.what() << std::endl;
} catch(const std::exception& e) {
	std::cout << "STD exception caught: " << e.what() << std::endl;
}
	return EXIT_SUCCESS;
}
