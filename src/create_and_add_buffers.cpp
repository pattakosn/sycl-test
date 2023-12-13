#include <sycl/sycl.hpp>
#include <vector>
#include <cstdio>
#include <chrono>

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
try{
	constexpr size_t N = 1024 * 1024 * 16;
	constexpr int result = 42;

	sycl::buffer<int> bufA{sycl::range{N}};
	sycl::buffer<int> bufB{sycl::range{N}};
	sycl::buffer<int> bufR{sycl::range{N}};
	sycl::accessor pC{bufR};

	auto q = sycl::queue{sycl::cpu_selector_v};
	std::cout << "\nChosen device: \"" << q.get_device().get_info<sycl::info::device::name>() << "\""	<< std::endl;
	// Initialize data directly on device
	my_timer time_all;
	time_all.start();
	q.submit([&](sycl::handler &cgh) {
		auto aA = sycl::accessor{bufA, cgh, sycl::write_only, sycl::no_init};
		auto aB = sycl::accessor{bufB, cgh, sycl::write_only, sycl::no_init};
		auto aR = sycl::accessor{bufR, cgh, sycl::write_only, sycl::no_init};
		cgh.parallel_for(N, [=](sycl::id<1> id) {
			aA[id] = 1;
			aB[id] = 40;
			aR[id] = 0;
		});
	});//.wait_and_throw();
	//time_all.stop("Data creation: ");
	// add
	q.submit([&](sycl::handler &cgh) {
		auto aA = sycl::accessor{bufA, cgh, sycl::read_only};
		auto aB = sycl::accessor{bufB, cgh, sycl::read_only};
		auto aR = sycl::accessor{bufR, cgh, sycl::read_write};
		cgh.parallel_for(N, [=](sycl::id<1> id) {
			aR[id] += aA[id] + aB[id];
		});
	});
	// increase by one via predeclared accessor
	q.submit([&](sycl::handler &cgh) {
		cgh.require(pC);
		cgh.parallel_for(N, [=](sycl::id<1> i) {
			pC[i]++;
		});
	});

	sycl::host_accessor end_data{bufR, sycl::read_only};
	time_all.stop("After crunching: ");
	for (int i = 0; i < N; i++)
		assert(end_data[i] == result);
	time_all.stop("Data verification on host: ");
} catch(const sycl::exception &e) {
 	std::cout << "SyCL exception caught: " << e.what() << std::endl;
} catch(const std::exception& e) {
	std::cout << "STD exception caught: " << e.what() << std::endl;
}
	return EXIT_SUCCESS;
}
