// ============================================================================
//  toy_llm.hpp  —  shared plumbing for all four rungs (v1..v4)
// ----------------------------------------------------------------------------
//  Everything here is IDENTICAL across the four programs: the math, the
//  tokenizer, reading a training file, saving/loading a model, and the
//  autocomplete loop. The only thing that differs from rung to rung is the
//  model itself, which lives in each rung's own .cpp. Keeping the plumbing in
//  one place is deliberate: it means the diff between v1 and v4 is purely the
//  idea that changed, not boilerplate.
//
//  Model file format is plain text on purpose, so you can open it and read it.
// ============================================================================
#pragma once
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <string>
#include <map>
#include <cmath>
#include <random>
#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <limits>

using Vector = std::vector<double>;
using Matrix = std::vector<Vector>;

// ---------------------------- math --------------------------------------------
inline double dotProduct(const Vector& a, const Vector& b) {
    double s = 0.0; for (size_t i = 0; i < a.size(); ++i) s += a[i]*b[i]; return s;
}
inline Vector softmax(const Vector& logits) {
    double m = *std::max_element(logits.begin(), logits.end());
    Vector p(logits.size()); double t = 0.0;
    for (size_t i = 0; i < logits.size(); ++i) { p[i] = std::exp(logits[i]-m); t += p[i]; }
    for (double& x : p) x /= t;
    return p;
}
inline double crossEntropyLoss(const Vector& p, int correct) { return -std::log(p[correct] + 1e-12); }
inline int argmax(const Vector& v) { return (int)(std::max_element(v.begin(), v.end()) - v.begin()); }
inline int sampleFrom(const Vector& p, std::mt19937& rng) {
    std::discrete_distribution<int> d(p.begin(), p.end()); return d(rng);
}

// ---------------------------- text --------------------------------------------
inline std::string readFile(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open file: " + path);
    std::stringstream ss; ss << f.rdbuf(); return ss.str();
}

// Split into lowercase word tokens. Sentence punctuation (. ! ?) becomes an
// <eos>/<bos> boundary. addTrailingEos closes the final sentence (used for
// training); generation leaves the sentence open so the model can continue it.
inline std::vector<std::string> tokenizeCore(const std::string& text, bool addTrailingEos) {
    std::vector<std::string> tokens; std::string word;
    auto flush = [&]{ if (!word.empty()) { tokens.push_back(word); word.clear(); } };
    tokens.push_back("<bos>");
    for (char ch : text) {
        if (std::isalnum((unsigned char)ch)) word += (char)std::tolower((unsigned char)ch);
        else { flush(); if (ch=='.'||ch=='!'||ch=='?') { tokens.push_back("<eos>"); tokens.push_back("<bos>"); } }
    }
    flush();
    if (!tokens.empty() && tokens.back()=="<bos>") tokens.pop_back();
    if (addTrailingEos && !tokens.empty() && tokens.back()!="<eos>") tokens.push_back("<eos>");
    return tokens;
}
inline std::vector<std::string> tokenize(const std::string& text)       { return tokenizeCore(text, true); }
inline std::vector<std::string> tokenizePrompt(const std::string& text) {
    auto t = tokenizeCore(text, false);
    while (!t.empty() && t.back()=="<eos>") t.pop_back();
    return t;
}

// A tiny built-in corpus, so every rung runs with no data file (`train --builtin`).
inline const std::string CORPUS =
    "the capital of france is paris. "
    "the capital of japan is tokyo. "
    "the capital of egypt is cairo. "
    "the capital of italy is rome. "
    "the capital of spain is madrid. "
    "the capital of germany is berlin.";

// ------------------- model serialization (human-readable) ---------------------
inline void expectToken(std::istream& in, const std::string& kw) {
    std::string got; in >> got;
    if (got != kw) throw std::runtime_error("model file: expected '" + kw + "', got '" + got + "'");
}
inline void saveVocab(std::ostream& out, const std::vector<std::string>& vocab) {
    out << "VOCAB " << vocab.size() << "\n";
    for (auto& t : vocab) out << t << "\n";        // tokens have no spaces
}
inline std::vector<std::string> loadVocab(std::istream& in) {
    expectToken(in, "VOCAB"); size_t n; in >> n;
    std::vector<std::string> v(n);
    for (size_t i = 0; i < n; ++i) in >> v[i];
    return v;
}
inline void saveMatrix(std::ostream& out, const std::string& name, const Matrix& M) {
    int rows = (int)M.size(), cols = rows ? (int)M[0].size() : 0;
    out << "MATRIX " << name << " " << rows << " " << cols << "\n";
    out << std::setprecision(9);
    for (auto& r : M) { for (size_t j = 0; j < r.size(); ++j) { if (j) out << ' '; out << r[j]; } out << "\n"; }
}
inline Matrix loadMatrix(std::istream& in, const std::string& name) {
    expectToken(in, "MATRIX"); std::string nm; int rows, cols; in >> nm >> rows >> cols;
    if (nm != name) throw std::runtime_error("model file: expected matrix '" + name + "', got '" + nm + "'");
    Matrix M(rows, Vector(cols));
    for (int i = 0; i < rows; ++i) for (int j = 0; j < cols; ++j) in >> M[i][j];
    return M;
}
inline void saveVector(std::ostream& out, const std::string& name, const Vector& v) {
    out << "VECTOR " << name << " " << v.size() << "\n" << std::setprecision(9);
    for (size_t j = 0; j < v.size(); ++j) { if (j) out << ' '; out << v[j]; } out << "\n";
}
inline Vector loadVector(std::istream& in, const std::string& name) {
    expectToken(in, "VECTOR"); std::string nm; int n; in >> nm >> n;
    if (nm != name) throw std::runtime_error("model file: expected vector '" + name + "', got '" + nm + "'");
    Vector v(n); for (int i = 0; i < n; ++i) in >> v[i]; return v;
}

// ----------------------------- generation -------------------------------------
//  Identical for every rung. Each model exposes:
//      .tokenToId, .vocabulary
//      Vector nextDistribution(const std::vector<int>& seq) const
//          -> probabilities for the token after `seq`, or an EMPTY vector if the
//             model cannot continue (used by the v2 foil on an unseen prefix).
//  greedy = take the most likely token; sample = draw from the distribution.
template <class Model>
std::string autocomplete(const Model& m, const std::string& prompt,
                         int maxNewTokens, bool sample, std::mt19937& rng) {
    std::vector<std::string> ptoks = tokenizePrompt(prompt);
    std::vector<int> seq;
    for (auto& w : ptoks) { auto it = m.tokenToId.find(w); if (it != m.tokenToId.end()) seq.push_back(it->second); }
    if (seq.empty()) { auto it = m.tokenToId.find("<bos>"); if (it != m.tokenToId.end()) seq.push_back(it->second); }

    for (int step = 0; step < maxNewTokens; ++step) {
        Vector p = m.nextDistribution(seq);
        if (p.empty()) break;                                   // model cannot continue
        int next = sample ? sampleFrom(p, rng) : argmax(p);
        if (m.vocabulary[next] == "<eos>") break;
        seq.push_back(next);
    }
    std::string out;
    for (int id : seq) {
        const std::string& w = m.vocabulary[id];
        if (w == "<bos>" || w == "<eos>") continue;
        if (!out.empty()) out += ' ';
        out += w;
    }
    return out;
}

// --------------------------- tiny CLI helpers ---------------------------------
inline int    intArg   (int argc, char** argv, const std::string& flag, int dflt) {
    for (int i = 1; i + 1 < argc; ++i) if (flag == argv[i]) return std::atoi(argv[i+1]);
    return dflt;
}
inline bool   flagArg  (int argc, char** argv, const std::string& flag) {
    for (int i = 1; i < argc; ++i) if (flag == argv[i]) return true;
    return false;
}
