#ifndef TIMER_H
#define TIMER_H

#include <chrono>
#include <iostream>

class Timer {
 public:
  Timer() {
    is_activated = false;
  }

  void start() {
    if (!is_activated) {
      std::cout << "Timer start..." << std::endl;
      start_point = std::chrono::system_clock::now();
      is_activated = true;
    } else {
      std::cerr << "Error: can not start when timer is already running" << std::endl;
    }
  }

  uint32_t count() {
    if (is_activated) {
      auto duration = std::chrono::system_clock::now() - start_point;
      return std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
    } else {
      std::cerr << "Error: count when timer has not been started!" << std::endl;
      return 0;
    }
  }

  uint32_t stop() {
    if (is_activated) {
      end_point = std::chrono::system_clock::now();
      auto duration = end_point - start_point;
      is_activated = false;
      auto res = std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
      std::cout << "Timer stopped: took " << res << " us" << std::endl;
      return res;
    } else {
      std::cerr << "Error: stop when timer has not been started!" << std::endl;
      return 0;
    }
  }

 private:
  bool is_activated;
  std::chrono::system_clock::time_point start_point;
  std::chrono::system_clock::time_point end_point;
};

#endif
