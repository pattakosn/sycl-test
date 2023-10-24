#include <sycl/sycl.hpp>
#include <vector>
#include <cstdio>
#include <chrono>

class vector_add;
class my_timer {
public:
	my_timer() {
		t0 = std::chrono::steady_clock::now();
		t1 = std::chrono::steady_clock::now();
	};
	void start(void) {
		t0 = std::chrono::steady_clock::now();
	}
	void stop (std::string msg) {
		t1 = std::chrono::steady_clock::now();
		std::cout << msg << std::chrono::duration_cast<std::chrono::milliseconds> (t1 - t0).count() << "[ms]" << std::endl;
		t0 = t1;
	}
private:
	std::chrono::steady_clock::time_point t0;
	std::chrono::steady_clock::time_point t1;
};

int main(void)
{
	constexpr size_t N = 1024 * 1024 * 16;
	my_timer time_all;

try{
	for ( size_t i = 0.; i < N; ++i ) {
		a[i] = i;
		b[i] = 2. * i;
	}
	time_all.stop("Data initialization: ");

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

	time_all.start();
	auto q = sycl::queue{sycl::gpu_selector_v};
	std::cout << "\nChosen device: \"" << q.get_device().get_info<sycl::info::device::name>() 
		<< "\", max compute units: " << q.get_device().get_info<sycl::info::device::max_compute_units>()
		<< std::endl;

	auto bufA = sycl::buffer{a.data(), sycl::range{a.size()}};
	auto bufB = sycl::buffer{b.data(), sycl::range{b.size()}};
	auto bufR = sycl::buffer{r.data(), sycl::range{r.size()}};
//for (int i =0; i < 1024;i++)
	q.submit([&](sycl::handler &cgh) {
		auto accA = sycl::accessor{bufA, cgh, sycl::read_only};
		auto accB = sycl::accessor{bufB, cgh, sycl::read_only};
		auto accR = sycl::accessor{bufR, cgh, sycl::write_only};

		cgh.parallel_for<vector_add>(sycl::range(N), [=](sycl::id<1> id) { accR[id] = accA[id] + accB[id]; });
	});
	
	q.wait_and_throw();
	time_all.stop("Data crunching: ");
} catch(const sycl::exception &e) {
 	std::cout << "SyCL exception caught: " << e.what() << std::endl;
} catch(const std::exception& e) {
	std::cout << "STD exception caught: " << e.what() << std::endl;
}
	for( int i = 0; i < 8*16; i+=16 )//N; i+= (1024*1024))
		std::cout << " a[" << i << "]= " << a[i] << "\t+ b[" << i << "]= " << b[i] << "\t= c["<< i << "]= " << r[i] << std::endl;
	return EXIT_SUCCESS;
}
