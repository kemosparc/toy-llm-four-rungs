// ============================================================================
//  llm_v2_ngram_lookup.cpp     RUNG 2 of 4  —  the foil: memorize whole prefixes
// ----------------------------------------------------------------------------
//  The tempting wrong turn. Instead of one row per TOKEN, give one row per whole
//  PREFIX ("<bos> the capital of france is"). It then memorizes each context's
//  continuation perfectly — and is utterly blind to any prefix it never saw,
//  because that prefix simply has no row. It also explodes: the number of rows
//  grows with the text instead of staying at the vocabulary size.
//
//  Build:    g++ -std=c++17 -O2 llm_v2_ngram_lookup.cpp -o v2
//  Train:    ./v2 train capitals.txt v2.model         (or:  train --builtin v2.model)
//  Generate: ./v2 gen   v2.model "the capital of france is"   (seen  -> works)
//            ./v2 gen   v2.model "france is the capital"      (unseen-> blind, stops)
// ============================================================================
#include "toy_llm.hpp"

// ============================================================================
//  THE MODEL: index by the whole prefix.
//    E : [prefix][D]   one row per distinct context seen in training
//    W : [D][vocab]    prefix row -> next-token scores
// ============================================================================
struct NgramLookupModel {
    int embeddingSize = 0; double learningRate = 0.0;
    std::vector<std::string> vocabulary;            // OUTPUT vocab (single tokens)
    std::map<std::string,int> tokenToId;
    std::vector<std::string> prefixVocab;           // one entry per memorized context
    std::map<std::string,int> prefixToId;
    Matrix E, W;
    std::vector<std::pair<int,int>> examples;       // (prefixId, targetId)
    std::mt19937 rng;

    int vocabularySize() const { return (int)vocabulary.size(); }
    int prefixCount()    const { return (int)prefixVocab.size(); }

    NgramLookupModel() : rng(0) {}

    NgramLookupModel(const std::vector<std::string>& tokens, int D, double lr, unsigned seed)
        : embeddingSize(D), learningRate(lr), rng(seed) {
        // output vocabulary (the single tokens we can predict)
        std::vector<std::string> d = tokens;
        std::sort(d.begin(), d.end()); d.erase(std::unique(d.begin(), d.end()), d.end());
        vocabulary = d; for (int i = 0; i < vocabularySize(); ++i) tokenToId[vocabulary[i]] = i;
        // prefix vocabulary + examples: every running context gets its own row
        int sentenceStart = 0;
        for (int t = 1; t < (int)tokens.size(); ++t) {
            if (tokens[t-1] == "<bos>") sentenceStart = t-1;
            std::string key;
            for (int i = sentenceStart; i < t; ++i) { if (!key.empty()) key += " "; key += tokens[i]; }
            if (!prefixToId.count(key)) { prefixToId[key] = (int)prefixVocab.size(); prefixVocab.push_back(key); }
            examples.push_back({ prefixToId[key], tokenToId[tokens[t]] });
        }
        std::uniform_real_distribution<double> init(-0.5, 0.5);
        E.assign(prefixCount(), Vector(D));
        W.assign(D, Vector(vocabularySize()));
        for (auto& r : E) for (double& x : r) x = init(rng);
        for (auto& r : W) for (double& x : r) x = init(rng);
    }

    Vector predictFromPrefix(int prefixId) const {
        const Vector& x = E[prefixId];
        Vector logits(vocabularySize(), 0.0);
        for (int v = 0; v < vocabularySize(); ++v)
            for (int k = 0; k < embeddingSize; ++k) logits[v] += x[k]*W[k][v];
        return softmax(logits);
    }
    // Look the running prefix up by exact string. No row -> EMPTY -> blind.
    Vector nextDistribution(const std::vector<int>& seq) const {
        std::string key;
        for (size_t i = 0; i < seq.size(); ++i) { if (i) key += " "; key += vocabulary[seq[i]]; }
        auto it = prefixToId.find(key);
        if (it == prefixToId.end()) return {};
        return predictFromPrefix(it->second);
    }
    double trainOnExample(int prefixId, int targetId) {
        Vector x = E[prefixId];
        Vector p = predictFromPrefix(prefixId);
        double loss = crossEntropyLoss(p, targetId);
        Vector g = p; g[targetId] -= 1.0;
        Vector dx(embeddingSize, 0.0);
        for (int k = 0; k < embeddingSize; ++k)
            for (int v = 0; v < vocabularySize(); ++v) dx[k] += W[k][v]*g[v];
        for (int k = 0; k < embeddingSize; ++k)
            for (int v = 0; v < vocabularySize(); ++v) W[k][v] -= learningRate*x[k]*g[v];
        for (int k = 0; k < embeddingSize; ++k) E[prefixId][k] -= learningRate*dx[k];
        return loss;
    }

    // -------- persistence (prefixes hold spaces, so they are line-based) -------
    void saveTo(const std::string& path) const {
        std::ofstream out(path);
        out << "TOYLLM ngram_lookup\n" << "D " << embeddingSize << "\n" << "lr " << learningRate << "\n";
        saveVocab(out, vocabulary);
        out << "PREFIXES " << prefixVocab.size() << "\n";
        for (auto& p : prefixVocab) out << p << "\n";
        saveMatrix(out, "E", E);
        saveMatrix(out, "W", W);
    }
    static NgramLookupModel loadFrom(const std::string& path) {
        std::ifstream in(path);
        if (!in) throw std::runtime_error("cannot open model: " + path);
        expectToken(in, "TOYLLM"); std::string arch; in >> arch;
        NgramLookupModel m;
        expectToken(in, "D");  in >> m.embeddingSize;
        expectToken(in, "lr"); in >> m.learningRate;
        m.vocabulary = loadVocab(in);
        for (int i = 0; i < (int)m.vocabulary.size(); ++i) m.tokenToId[m.vocabulary[i]] = i;
        expectToken(in, "PREFIXES"); size_t np; in >> np;
        in.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        m.prefixVocab.resize(np);
        for (size_t i = 0; i < np; ++i) { std::getline(in, m.prefixVocab[i]); m.prefixToId[m.prefixVocab[i]] = (int)i; }
        m.E = loadMatrix(in, "E");
        m.W = loadMatrix(in, "W");
        return m;
    }
};

static void usage() {
    std::cout <<
      "usage:\n"
      "  v2 train <data.txt|--builtin> <model.out> [--epochs N]\n"
      "  v2 gen   <model.in> \"<prompt>\" [--new N] [--sample]\n";
}

int main(int argc, char** argv) {
    if (argc < 2) { usage(); return 1; }
    std::string cmd = argv[1];

    if (cmd == "train") {
        if (argc < 4) { usage(); return 1; }
        std::string dataPath = argv[2], modelPath = argv[3];
        int epochs = intArg(argc, argv, "--epochs", 2000);
        std::string text = (dataPath == "--builtin") ? CORPUS : readFile(dataPath);

        std::vector<std::string> tokens = tokenize(text);
        NgramLookupModel model(tokens, /*D=*/8, /*lr=*/0.1, /*seed=*/42);

        std::cout << "training v2 (memorize whole prefixes)\n"
                  << "  rows by TOKEN would be : " << model.vocabularySize() << " (fixed)\n"
                  << "  rows by PREFIX (actual): " << model.prefixCount()
                  << " (already more, and it grows with the text)\n";
        for (int epoch = 1; epoch <= epochs; ++epoch) {
            double total = 0.0;
            for (auto& ex : model.examples) total += model.trainOnExample(ex.first, ex.second);
            if (epoch == 1 || epoch % 500 == 0 || epoch == epochs)
                std::cout << "  epoch " << std::setw(5) << epoch << "   loss "
                          << std::fixed << std::setprecision(4) << total/model.examples.size() << "\n";
        }
        model.saveTo(modelPath);
        std::cout << "saved model -> " << modelPath << "\n";
        return 0;
    }

    if (cmd == "gen") {
        if (argc < 4) { usage(); return 1; }
        std::string modelPath = argv[2], prompt = argv[3];
        int maxNew = intArg(argc, argv, "--new", 12);
        bool sample = flagArg(argc, argv, "--sample");
        NgramLookupModel model = NgramLookupModel::loadFrom(modelPath);
        std::mt19937 rng(std::random_device{}());
        std::string out = autocomplete(model, prompt, maxNew, sample, rng);
        std::cout << out << "\n";
        // If nothing was added beyond the prompt, the prefix was never memorized.
        if (out == autocomplete(model, prompt, 0, false, rng))
            std::cout << "  (no row for this exact prefix - the lookup model is blind to it)\n";
        return 0;
    }

    usage(); return 1;
}
