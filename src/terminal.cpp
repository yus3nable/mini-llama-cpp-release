#include "mini_llama/terminal.h"
#include <iostream>
#include <iomanip>

namespace mini_llama {

void Terminal::print_user_prompt() const {
    std::cout << "mini-llama > ";
    std::cout.flush();
}

std::string Terminal::read_line() const {
    std::string line;
    if (std::getline(std::cin, line)) {
        return line;
    }
    return "";
}

void Terminal::print_assistant_prefix() const {
    std::cout << "assistant  > ";
    std::cout.flush();
}

void Terminal::print_token_text(const std::string& text) const {
    std::cout << text;
}

void Terminal::flush() const {
    std::cout.flush();
}

void Terminal::new_line() const {
    std::cout << "\n";
}

void Terminal::print_help() const {
    std::cout << "Commands:\n"
              << "  /help    Show this help message\n"
              << "  /clear   Clear chat history and context\n"
              << "  /stats   Show session statistics\n"
              << "  /params  Show current sampling parameters\n"
              << "  /exit    Exit chat\n";
}

void Terminal::print_stats(const ChatSession& session) const {
    std::cout << "Session stats:\n"
              << "  messages:           " << session.messages.size() << "\n"
              << "  context tokens:     " << session.token_history.size() << "\n"
              << "  total prompt tokens: " << session.total_prompt_tokens << "\n"
              << "  total generated:     " << session.total_generated_tokens << "\n"
              << "  total time:          " << std::fixed << std::setprecision(2)
              << session.total_time_ms << " ms\n"
              << "  tokens/s:            ";
    if (session.total_time_ms > 0.0) {
        std::cout << std::fixed << std::setprecision(2)
                  << (session.total_generated_tokens * 1000.0 / session.total_time_ms);
    } else {
        std::cout << "N/A";
    }
    std::cout << std::defaultfloat << "\n";
}

void Terminal::print_params(const SamplingParams& params) const {
    std::cout << std::defaultfloat
              << "Sampling params:\n"
              << "  temperature: " << params.temperature << "\n"
              << "  top_k:       " << params.top_k << "\n"
              << "  seed:        " << params.seed << "\n";
}

void Terminal::print_message(const std::string& msg) const {
    std::cout << msg << "\n";
}

} // namespace mini_llama
