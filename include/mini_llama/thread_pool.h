#pragma once

#include <thread>
#include <vector>
#include <functional>

namespace mini_llama {

// Get the configured number of threads for parallel operations.
// Defaults to std::thread::hardware_concurrency().
int get_thread_count();

// Set the number of threads for parallel operations.
// Pass 0 to reset to hardware_concurrency().
void set_thread_count(int n);

// Simple parallel for: split [0, n) into roughly equal chunks.
// Uses get_thread_count() to decide how many threads to spawn.
void parallel_for(int n, const std::function<void(int begin, int end)>& fn);

} // namespace mini_llama
