// Minimal C++ test runner header (declarations only)
#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <functional>
#include <cmath>

// Test registry
struct TestCase {
    std::string name;
    std::function<bool()> fn;
};

std::vector<TestCase>& test_registry();
void register_test(const std::string& name, std::function<bool()> fn);

// Assertion helpers (inlined to avoid duplicate symbols)
inline bool assert_true(bool cond, const char* expr, const char* file, int line) {
    if (!cond) {
        std::cerr << "  ASSERTION FAILED: " << expr
                  << " at " << file << ":" << line << std::endl;
        return false;
    }
    return true;
}

inline bool assert_eq_float(float a, float b, float tol, const char* expr_a, const char* expr_b,
                            const char* file, int line) {
    if (std::fabs(a - b) > tol) {
        std::cerr << "  ASSERTION FAILED: |" << expr_a << " - " << expr_b
                  << "| = |" << a << " - " << b << "| = " << std::fabs(a - b)
                  << " > " << tol
                  << " at " << file << ":" << line << std::endl;
        return false;
    }
    return true;
}

#define ASSERT_TRUE(cond) do { if (!assert_true((cond), #cond, __FILE__, __LINE__)) return false; } while(0)
#define ASSERT_EQ(a, b) do { if (!assert_true((a) == (b), #a " == " #b, __FILE__, __LINE__)) return false; } while(0)
#define ASSERT_NEAR(a, b, tol) do { if (!assert_eq_float((a), (b), (tol), #a, #b, __FILE__, __LINE__)) return false; } while(0)
#define ASSERT_FAIL(msg) do { std::cerr << "  ASSERTION FAILED: " << msg << " at " << __FILE__ << ":" << __LINE__ << std::endl; return false; } while(0)
