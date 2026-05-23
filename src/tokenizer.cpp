#include "mini_llama/tokenizer.h"

namespace mini_llama {

std::vector<int> AsciiTokenizer::encode(const std::string& text) const {
    std::vector<int> tokens;
    tokens.push_back(bos_id());
    for (unsigned char c : text) {
        int id = static_cast<int>(c);
        if (id >= VOCAB_SIZE) {
            tokens.push_back(unk_id());
        } else {
            tokens.push_back(id);
        }
    }
    return tokens;
}

std::string AsciiTokenizer::decode_token(int token) const {
    if (token == bos_id()) {
        return "<bos>";
    }
    if (token == eos_id()) {
        return "<eos>";
    }
    if (token == unk_id()) {
        return "<unk>";
    }
    if (token >= VOCAB_SIZE) {
        return "<unk>";
    }
    if (token < 32 || token == 127) {
        // Control characters: represent as empty
        return "";
    }
    return std::string(1, static_cast<char>(token));
}

std::string AsciiTokenizer::decode(const std::vector<int>& tokens) const {
    std::string result;
    for (int token : tokens) {
        result += decode_token(token);
    }
    return result;
}

} // namespace mini_llama
