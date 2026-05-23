#pragma once

#include "mini_llama/context.h"
#include "mini_llama/sampler.h"
#include <string>
#include <vector>

namespace mini_llama {

struct ChatMessage {
    std::string role;     // "system", "user", "assistant"
    std::string content;
};

// Manages the state of an interactive chat session.
class ChatSession {
public:
    std::vector<ChatMessage> messages;
    std::vector<int> token_history;
    SamplingParams sampling_params;

    // Stats
    int total_prompt_tokens = 0;
    int total_generated_tokens = 0;
    double total_time_ms = 0.0;

    ChatSession() = default;

    // Reset messages and stats.
    void clear();

    // Add a message to the history.
    void add_message(const std::string& role, const std::string& content);

    // Increment stats after a turn.
    void record_turn(int prompt_tokens, int generated_tokens, double time_ms);

    // Replace the current token history for the latest evaluated context.
    void set_token_history(const std::vector<int>& tokens);

    // Append one generated token to the tracked context.
    void append_token(int token);
};

} // namespace mini_llama
