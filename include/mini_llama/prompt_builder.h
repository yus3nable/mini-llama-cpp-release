#pragma once

#include "mini_llama/chat.h"
#include <string>
#include <vector>

namespace mini_llama {

// Builds a text prompt from a chat message history.
// Supports two modes:
//   1. Plain text (default): System:/User:/Assistant: format
//   2. Chat template: applies a model-specific template (currently Qwen2)
//
// The trailing assistant prefix tells the model to start generating.
class PromptBuilder {
public:
    PromptBuilder() = default;

    // Set a chat template. If empty, uses plain text mode.
    void set_chat_template(const std::string& template_str);

    // Build prompt from messages.
    std::string build(const std::vector<ChatMessage>& messages) const;

private:
    std::string chat_template_;

    std::string build_plain(const std::vector<ChatMessage>& messages) const;
    std::string build_qwen2(const std::vector<ChatMessage>& messages) const;
};

// Load chat template from a GGUF file. Returns the raw Jinja2 template string
// if present, or "qwen2" as a fallback for known model families.
std::string load_chat_template_from_gguf(const std::string& gguf_path);

} // namespace mini_llama
