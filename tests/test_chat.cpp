#include "test_main.h"
#include "mini_llama/chat.h"
#include "mini_llama/prompt_builder.h"
#include "mini_llama/terminal.h"
#include "mini_llama/sampler.h"

using namespace mini_llama;

// ---------------------------------------------------------------------------
// ChatMessage
// ---------------------------------------------------------------------------
static bool test_chat_message_stores_role_and_content() {
    ChatMessage msg{"user", "hello"};
    ASSERT_TRUE(msg.role == "user");
    ASSERT_TRUE(msg.content == "hello");
    return true;
}

// ---------------------------------------------------------------------------
// ChatSession
// ---------------------------------------------------------------------------
static bool test_chat_session_starts_empty() {
    ChatSession session;
    ASSERT_TRUE(session.messages.empty());
    ASSERT_TRUE(session.token_history.empty());
    ASSERT_EQ(session.total_prompt_tokens, 0);
    ASSERT_EQ(session.total_generated_tokens, 0);
    ASSERT_NEAR(session.total_time_ms, 0.0, 1e-6);
    return true;
}

static bool test_chat_session_add_message_appends() {
    ChatSession session;
    session.add_message("system", "You are helpful.");
    session.add_message("user", "hi");
    ASSERT_EQ(session.messages.size(), 2);
    ASSERT_TRUE(session.messages[0].role == "system");
    ASSERT_TRUE(session.messages[0].content == "You are helpful.");
    ASSERT_TRUE(session.messages[1].role == "user");
    ASSERT_TRUE(session.messages[1].content == "hi");
    return true;
}

static bool test_chat_session_clear_resets() {
    ChatSession session;
    session.add_message("user", "test");
    session.set_token_history({1, 2, 3});
    session.record_turn(10, 5, 100.0);
    session.clear();
    ASSERT_TRUE(session.messages.empty());
    ASSERT_TRUE(session.token_history.empty());
    ASSERT_EQ(session.total_prompt_tokens, 0);
    ASSERT_EQ(session.total_generated_tokens, 0);
    ASSERT_NEAR(session.total_time_ms, 0.0, 1e-6);
    return true;
}

static bool test_chat_session_tracks_token_history() {
    ChatSession session;
    session.set_token_history({1, 10, 11});
    session.append_token(12);
    ASSERT_EQ(session.token_history.size(), 4);
    ASSERT_EQ(session.token_history[0], 1);
    ASSERT_EQ(session.token_history[3], 12);
    return true;
}

static bool test_chat_session_record_turn_accumulates() {
    ChatSession session;
    session.record_turn(10, 5, 100.0);
    session.record_turn(20, 8, 200.0);
    ASSERT_EQ(session.total_prompt_tokens, 30);
    ASSERT_EQ(session.total_generated_tokens, 13);
    ASSERT_NEAR(session.total_time_ms, 300.0, 1e-6);
    return true;
}

// ---------------------------------------------------------------------------
// PromptBuilder
// ---------------------------------------------------------------------------
static bool test_prompt_builder_empty_history() {
    PromptBuilder builder;
    std::vector<ChatMessage> messages;
    std::string prompt = builder.build(messages);
    ASSERT_TRUE(prompt == "Assistant:");
    return true;
}

static bool test_prompt_builder_multi_turn() {
    PromptBuilder builder;
    std::vector<ChatMessage> messages = {
        {"system", "You are helpful."},
        {"user", "hi"},
        {"assistant", "Hello!"},
        {"user", "bye"}
    };
    std::string prompt = builder.build(messages);
    ASSERT_TRUE(prompt == "System: You are helpful.\n"
                          "User: hi\n"
                          "Assistant: Hello!\n"
                          "User: bye\n"
                          "Assistant:");
    return true;
}

static bool test_prompt_builder_ignores_unknown_roles() {
    PromptBuilder builder;
    std::vector<ChatMessage> messages = {
        {"unknown", "ignored"},
        {"user", "hello"}
    };
    std::string prompt = builder.build(messages);
    ASSERT_TRUE(prompt == "User: hello\nAssistant:");
    return true;
}

static bool test_prompt_builder_qwen2_template_injects_system_and_prefix() {
    PromptBuilder builder;
    builder.set_chat_template("qwen2");
    std::vector<ChatMessage> messages = {
        {"user", "你好"}
    };
    std::string prompt = builder.build(messages);
    ASSERT_TRUE(prompt == "<|im_start|>system\n"
                          "You are a helpful assistant.<|im_end|>\n"
                          "<|im_start|>user\n"
                          "你好<|im_end|>\n"
                          "<|im_start|>assistant\n");
    return true;
}

static bool test_prompt_builder_qwen2_template_keeps_existing_system() {
    PromptBuilder builder;
    builder.set_chat_template("qwen2");
    std::vector<ChatMessage> messages = {
        {"system", "Be concise."},
        {"user", "hi"}
    };
    std::string prompt = builder.build(messages);
    ASSERT_TRUE(prompt == "<|im_start|>system\n"
                          "Be concise.<|im_end|>\n"
                          "<|im_start|>user\n"
                          "hi<|im_end|>\n"
                          "<|im_start|>assistant\n");
    return true;
}

// ---------------------------------------------------------------------------
// Terminal (smoke tests — just verify no crash)
// ---------------------------------------------------------------------------
static bool test_terminal_print_user_prompt() {
    Terminal term;
    term.print_user_prompt();
    return true;
}

static bool test_terminal_print_assistant_prefix() {
    Terminal term;
    term.print_assistant_prefix();
    return true;
}

static bool test_terminal_print_help() {
    Terminal term;
    term.print_help();
    return true;
}

static bool test_terminal_print_stats() {
    Terminal term;
    ChatSession session;
    session.record_turn(10, 5, 250.0);
    term.print_stats(session);
    return true;
}

static bool test_terminal_print_params() {
    Terminal term;
    SamplingParams params;
    params.temperature = 0.7f;
    params.top_k = 40;
    params.seed = 42;
    term.print_params(params);
    return true;
}

static bool test_terminal_print_message() {
    Terminal term;
    term.print_message("test message");
    return true;
}

static bool test_terminal_new_line() {
    Terminal term;
    term.new_line();
    return true;
}

// ---------------------------------------------------------------------------
// Auto-register
// ---------------------------------------------------------------------------
static struct ChatTestRegistrar {
    ChatTestRegistrar() {
        register_test("chat_message_stores_role_and_content", test_chat_message_stores_role_and_content);
        register_test("chat_session_starts_empty", test_chat_session_starts_empty);
        register_test("chat_session_add_message_appends", test_chat_session_add_message_appends);
        register_test("chat_session_clear_resets", test_chat_session_clear_resets);
        register_test("chat_session_record_turn_accumulates", test_chat_session_record_turn_accumulates);
        register_test("chat_session_tracks_token_history", test_chat_session_tracks_token_history);
        register_test("prompt_builder_empty_history", test_prompt_builder_empty_history);
        register_test("prompt_builder_multi_turn", test_prompt_builder_multi_turn);
        register_test("prompt_builder_ignores_unknown_roles", test_prompt_builder_ignores_unknown_roles);
        register_test("prompt_builder_qwen2_template_injects_system_and_prefix", test_prompt_builder_qwen2_template_injects_system_and_prefix);
        register_test("prompt_builder_qwen2_template_keeps_existing_system", test_prompt_builder_qwen2_template_keeps_existing_system);
        register_test("terminal_print_user_prompt", test_terminal_print_user_prompt);
        register_test("terminal_print_assistant_prefix", test_terminal_print_assistant_prefix);
        register_test("terminal_print_help", test_terminal_print_help);
        register_test("terminal_print_stats", test_terminal_print_stats);
        register_test("terminal_print_params", test_terminal_print_params);
        register_test("terminal_print_message", test_terminal_print_message);
        register_test("terminal_new_line", test_terminal_new_line);
    }
} chat_test_registrar;
