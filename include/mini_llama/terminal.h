#pragma once

#include "mini_llama/chat.h"
#include "mini_llama/model.h"
#include "mini_llama/tokenizer.h"
#include <string>

namespace mini_llama {

// Handles interactive terminal I/O for the chat mode.
class Terminal {
public:
    Terminal() = default;

    // Print the user prompt (e.g., "mini-llama > ").
    void print_user_prompt() const;

    // Read a line from stdin. Returns empty string on EOF.
    std::string read_line() const;

    // Print assistant prefix (e.g., "assistant > ").
    void print_assistant_prefix() const;

    // Print a single decoded token (no newline, no flush).
    void print_token_text(const std::string& text) const;

    // Flush stdout.
    void flush() const;

    // Print a newline.
    void new_line() const;

    // Print the /help text.
    void print_help() const;

    // Print session stats.
    void print_stats(const ChatSession& session) const;

    // Print current sampling params.
    void print_params(const SamplingParams& params) const;

    // Print a generic message.
    void print_message(const std::string& msg) const;
};

} // namespace mini_llama
