#include "mini_llama/chat.h"

namespace mini_llama {

void ChatSession::clear() {
    messages.clear();
    token_history.clear();
    total_prompt_tokens = 0;
    total_generated_tokens = 0;
    total_time_ms = 0.0;
}

void ChatSession::add_message(const std::string& role, const std::string& content) {
    messages.push_back({role, content});
}

void ChatSession::record_turn(int prompt_tokens, int generated_tokens, double time_ms) {
    total_prompt_tokens += prompt_tokens;
    total_generated_tokens += generated_tokens;
    total_time_ms += time_ms;
}

void ChatSession::set_token_history(const std::vector<int>& tokens) {
    token_history = tokens;
}

void ChatSession::append_token(int token) {
    token_history.push_back(token);
}

} // namespace mini_llama
