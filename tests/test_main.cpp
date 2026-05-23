// Minimal C++ test runner (definitions only)
#include "test_main.h"

std::vector<TestCase>& test_registry() {
    static std::vector<TestCase> registry;
    return registry;
}

void register_test(const std::string& name, std::function<bool()> fn) {
    test_registry().push_back({name, fn});
}

int main(int argc, char** argv) {
    (void)argc; (void)argv;
    const auto& tests = test_registry();
    int passed = 0;
    int failed = 0;

    std::cout << "Running " << tests.size() << " tests..." << std::endl;
    std::cout << "========================================" << std::endl;

    for (const auto& t : tests) {
        std::cout << "[RUN] " << t.name << std::endl;
        bool ok = t.fn();
        if (ok) {
            std::cout << "[PASS] " << t.name << std::endl;
            ++passed;
        } else {
            std::cout << "[FAIL] " << t.name << std::endl;
            ++failed;
        }
    }

    std::cout << "========================================" << std::endl;
    std::cout << "Results: " << passed << " passed, " << failed << " failed" << std::endl;
    return failed > 0 ? 1 : 0;
}
