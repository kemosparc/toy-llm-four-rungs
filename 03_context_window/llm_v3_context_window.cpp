// ============================================================================
//  llm_v3_context_window.cpp   RUNG 3 of 4  —  last N tokens, glued together
// ----------------------------------------------------------------------------
//  Same learning loop as the bigram, but the context is the last N tokens with
//  their embeddings concatenated into one long vector. Now the country word is
//  still inside the window when the model must answer, so it can finally finish
//  "the capital of france is ___" correctly.
//
//  Build:    g++ -std=c++17 -O2 llm_v3_context_window.cpp -o v3
//  Train:    ./v3 train capitals.txt v3.model         (or:  train --builtin v3.model)
//  Generate: ./v3 gen   v3.model "the capital of france is"
//            optional:  --new 12   --sample
// ============================================================================
#include "toy_llm.hpp"

// ============================================================================
//  THE MODEL: next token from the last N tokens.
//    tokenEmbeddings : [vocab][D]        one vector per token            (E)
//    outputWeights   : [N*D][vocab]      glued context -> next-token     (W)
// ============================================================================
struct ContextWindowModel {
    int contextWindow = 0, embeddingSize = 0; double learningRate = 0.0;
    std::vector<std::string> vocabulary;
    std::map<std::string,int> tokenToId;
    Matrix tokenEmbeddings, outputWeights;
    std::mt19937 rng;

    int vocabularySize() const { return (int)vocabulary.size(); }
    int idOf(const std::string& t) const { return tokenToId.at(t); }

    ContextWindowModel() : rng(0) {}

    ContextWindowModel(const std::vector<std::string>& tokens, int N, int D, double lr, unsigned seed)
        : contextWindow(N), embeddingSize(D), learningRate(lr), rng(seed) {
        std::vector<std::string> d = tokens;
        std::sort(d.begin(), d.end()); d.erase(std::unique(d.begin(), d.end()), d.end());
        vocabulary = d; for (int i = 0; i < vocabularySize(); ++i) tokenToId[vocabulary[i]] = i;
        std::uniform_real_distribution<double> init(-0.5, 0.5);
        tokenEmbeddings.assign(vocabularySize(), Vector(D));
        outputWeights.assign(N*D, Vector(vocabularySize()));
        for (auto& r : tokenEmbeddings) for (double& x : r) x = init(rng);
        for (auto& r : outputWeights)  for (double& x : r) x = init(rng);
    }

    std::vector<int> contextEndingAt(const std::vector<int>& ids, int t) const {
        std::vector<int> c; int bos = idOf("<bos>");
        for (int back = contextWindow; back >= 1; --back) { int j = t-back; c.push_back(j < 0 ? bos : ids[j]); }
        return c;
    }
    // Glue the window's embeddings into one long context vector.
    Vector buildContextVector(const std::vector<int>& ctx) const {
        Vector x(contextWindow * embeddingSize);
        for (int s = 0; s < contextWindow; ++s)
            for (int k = 0; k < embeddingSize; ++k)
                x[s*embeddingSize + k] = tokenEmbeddings[ctx[s]][k];
        return x;
    }
    Vector predict(const std::vector<int>& ctx) const {
        Vector x = buildContextVector(ctx);
        Vector logits(vocabularySize(), 0.0);
        for (int v = 0; v < vocabularySize(); ++v)
            for (int k = 0; k < contextWindow*embeddingSize; ++k) logits[v] += x[k]*outputWeights[k][v];
        return softmax(logits);
    }
    Vector nextDistribution(const std::vector<int>& seq) const {
        return predict(contextEndingAt(seq, (int)seq.size()));
    }
    double trainOnExample(const std::vector<int>& ctx, int targetId) {
        Vector x = buildContextVector(ctx);
        Vector p = predict(ctx);
        double loss = crossEntropyLoss(p, targetId);
        Vector g = p; g[targetId] -= 1.0;
        int W = contextWindow*embeddingSize;
        Vector dContext(W, 0.0);
        for (int k = 0; k < W; ++k) for (int v = 0; v < vocabularySize(); ++v) dContext[k] += outputWeights[k][v]*g[v];
        for (int k = 0; k < W; ++k) for (int v = 0; v < vocabularySize(); ++v) outputWeights[k][v] -= learningRate*x[k]*g[v];
        for (int s = 0; s < contextWindow; ++s)
            for (int k = 0; k < embeddingSize; ++k)
                tokenEmbeddings[ctx[s]][k] -= learningRate * dContext[s*embeddingSize + k];
        return loss;
    }

    // -------- persistence --------
    void saveTo(const std::string& path) const {
        std::ofstream out(path);
        out << "TOYLLM context_window\n" << "N " << contextWindow << "\n"
            << "D " << embeddingSize << "\n" << "lr " << learningRate << "\n";
        saveVocab(out, vocabulary);
        saveMatrix(out, "E", tokenEmbeddings);
        saveMatrix(out, "W", outputWeights);
    }
    static ContextWindowModel loadFrom(const std::string& path) {
        std::ifstream in(path);
        if (!in) throw std::runtime_error("cannot open model: " + path);
        expectToken(in, "TOYLLM"); std::string arch; in >> arch;
        ContextWindowModel m;
        expectToken(in, "N");  in >> m.contextWindow;
        expectToken(in, "D");  in >> m.embeddingSize;
        expectToken(in, "lr"); in >> m.learningRate;
        m.vocabulary = loadVocab(in);
        for (int i = 0; i < (int)m.vocabulary.size(); ++i) m.tokenToId[m.vocabulary[i]] = i;
        m.tokenEmbeddings = loadMatrix(in, "E");
        m.outputWeights   = loadMatrix(in, "W");
        return m;
    }
};

static void usage() {
    std::cout <<
      "usage:\n"
      "  v3 train <data.txt|--builtin> <model.out> [--epochs N]\n"
      "  v3 gen   <model.in> \"<prompt>\" [--new N] [--sample]\n";
}

int main(int argc, char** argv) {
    if (argc < 2) { usage(); return 1; }
    std::string cmd = argv[1];

    if (cmd == "train") {
        if (argc < 4) { usage(); return 1; }
        std::string dataPath = argv[2], modelPath = argv[3];
        int epochs = intArg(argc, argv, "--epochs", 3000);
        std::string text = (dataPath == "--builtin") ? CORPUS : readFile(dataPath);

        std::vector<std::string> tokens = tokenize(text);
        ContextWindowModel model(tokens, /*N=*/3, /*D=*/8, /*lr=*/0.05, /*seed=*/42);

        std::vector<int> ids; for (auto& t : tokens) ids.push_back(model.idOf(t));
        std::vector<std::pair<std::vector<int>,int>> examples;
        for (int t = 1; t < (int)ids.size(); ++t)
            examples.push_back({ model.contextEndingAt(ids, t), ids[t] });

        std::cout << "training v3 (window of " << 3 << ") on " << examples.size()
                  << " examples, vocab " << model.vocabularySize() << "\n";
        for (int epoch = 1; epoch <= epochs; ++epoch) {
            double total = 0.0;
            for (auto& ex : examples) total += model.trainOnExample(ex.first, ex.second);
            if (epoch == 1 || epoch % 1000 == 0 || epoch == epochs)
                std::cout << "  epoch " << std::setw(5) << epoch << "   loss "
                          << std::fixed << std::setprecision(4) << total/examples.size() << "\n";
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
        ContextWindowModel model = ContextWindowModel::loadFrom(modelPath);
        std::mt19937 rng(std::random_device{}());
        std::cout << autocomplete(model, prompt, maxNew, sample, rng) << "\n";
        return 0;
    }

    usage(); return 1;
}
