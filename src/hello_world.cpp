#include <sycl/sycl.hpp>

class hello_world;

int main(void) {
  try {

    auto asyncHandler = [&](sycl::exception_list exceptionList) {
      for (auto& e : exceptionList) {
        std::rethrow_exception(e);
      }
    };
    auto defaultQueue = sycl::queue{asyncHandler};

    auto buf = sycl::buffer<int>(sycl::range{1});

    defaultQueue
      .submit([&](sycl::handler& cgh) {
          auto os = sycl::stream{128, 128, cgh};

          cgh.single_task<hello_world>([=]() { os << "Hello World!\n"; });
          })
    .wait();
    defaultQueue.submit([&](sycl::handler& cgh) {
        // This throws an exception: an accessor has a range which is
        // outside the bounds of its buffer.
        auto acc = buf.get_access(cgh, sycl::range{2}, sycl::read_write);
        });
    defaultQueue.wait_and_throw();
  } catch (const sycl::exception& e) {
    std::cout << "Exception caught: " << e.what() << std::endl;
  }

  return EXIT_SUCCESS;
}

