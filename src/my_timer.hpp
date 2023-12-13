#ifndef MY_TIMER_HPP
#define MY_TIMER_HPP
#include <iostream>
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
	void stop (std::string msg = {"Dt (in ms)"}) {
		t1 = std::chrono::steady_clock::now();
		std::cout << msg << std::chrono::duration_cast<std::chrono::milliseconds> (t1 - t0).count() << "[ms]" << std::endl;
		t0 = t1;
	}
private:
	std::chrono::steady_clock::time_point t0;
	std::chrono::steady_clock::time_point t1;
};
#endif