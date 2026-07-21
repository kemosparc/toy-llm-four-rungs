// ============================================================================
//  llm_v1_bigram.cpp          RUNG 1 of 4  —  one token of memory
// ----------------------------------------------------------------------------
//  Predict the next token from the ONE token before it. Simplest possible
//  neural language model: embed -> score -> softmax -> loss -> gradient.
//  Its wall: with one token of memory it cannot answer "the capital of france
//  is ___", because when it must answer it is only looking at "is".
//
//  Build:    g++ -std=c++17 -O2 llm_v1_bigram.cpp -o v1
//  Train:    ./v1 train capitals.txt v1.model         (or:  train --builtin v1.model)
//  Generate: ./v1 gen   v1.model "the capital of france is"
//            optional:  --new 12   --sample
// ============================================================================
#include "toy_llm.hpp"

// ============================================================================
//  THE MODEL: next token from ONE previous token.
//    tokenEmbeddings : [vocab][D]   one vector per token            (E)
//    outputWeights   : [D][vocab]   embedding -> next-token scores  (W)
// ============================================================================
struct BigramModel {
    int embeddingSize = 0; double learningRate = 0.0;
    std::vector<std::string> vocabulary;
    std::map<std::string,int> tokenToId;
    Matrix tokenEmbeddings, outputWeights;
    std::mt19937 rng;

    int vocabularySize() const { return (int)vocabulary.size(); }
    int idOf(const std::string& t) const { return tokenToId.at(t); }

    BigramModel() : rng(0) {}

    BigramModel(const std::vector<std::string>& tokens, int D, double lr, unsigned seed)
        : embeddingSize(D), learningRate(lr), rng(seed) {
        std::vector<std::string> d = tokens;
        std::sort(d.begin(), d.end()); d.erase(std::unique(d.begin(), d.end()), d.end());
        vocabulary = d; for (int i = 0; i < vocabularySize(); ++i) tokenToId[vocabulary[i]] = i;
        std::uniform_real_distribution<double> init(-0.5, 0.5);
        tokenEmbeddings.assign(vocabularySize(), Vector(D));
        outputWeights.assign(D, Vector(vocabularySize()));
        for (auto& r : tokenEmbeddings) for (double& x : r) x = init(rng);
        for (auto& r : outputWeights)  for (double& x : r) x = init(rng);
    }

    Vector predictFrom(int tokenId) const {
        Vector logits(vocabularySize(), 0.0);
        for (int v = 0; v < vocabularySize(); ++v)
            for (int k = 0; k < embeddingSize; ++k)
                logits[v] += tokenEmbeddings[tokenId][k] * outputWeights[k][v];
        return softmax(logits);
    }

    // a bigram's context is just the last token
    Vector nextDistribution(const std::vector<int>& seq) const {
        return predictFrom(seq.back());
    }

    double trainOnPair(int contextId, int targetId) {
        Vector x = tokenEmbeddings[contextId];
        Vector p = predictFrom(contextId);
        double loss = crossEntropyLoss(p, targetId);
        Vector g = p; g[targetId] -= 1.0;
        Vector dEmbedding(embeddingSize, 0.0);
        for (int k = 0; k < embeddingSize; ++k)
            for (int v = 0; v < vocabularySize(); ++v) dEmbedding[k] += outputWeights[k][v] * g[v];
        for (int k = 0; k < embeddingSize; ++k)
            for (int v = 0; v < vocabularySize(); ++v) outputWeights[k][v] -= learningRate * x[k] * g[v];
        for (int k = 0; k < embeddingSize; ++k)
            tokenEmbeddings[contextId][k] -= learningRate * dEmbedding[k];
        return loss;
    }

    // -------- persistence --------
    void saveTo(const std::string& path) const {
        std::ofstream out(path);
        out << "TOYLLM bigram\n" << "D " << embeddingSize << "\n" << "lr " << learningRate << "\n";
        saveVocab(out, vocabulary);
        saveMatrix(out, "E", tokenEmbeddings);
        saveMatrix(out, "W", outputWeights);
    }
    static BigramModel loadFrom(const std::string& path) {
        std::ifstream in(path);
        if (!in) throw std::runtime_error("cannot open model: " + path);
        expectToken(in, "TOYLLM"); std::string arch; in >> arch;
        BigramModel m;
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
      "  v1 train <data.txt|--builtin> <model.out> [--epochs N]\n"
      "  v1 gen   <model.in> \"<prompt>\" [--new N] [--sample]\n";
}

int main(int argc, char** argv) {
    if (argc < 2) { usage(); return 1; }
    std::string cmd = argv[1];

    if (cmd == "train") {
        if (argc < 4) { usage(); return 1; }
        std::string dataPath = argv[2], modelPath = argv[3];
        int epochs = intArg(argc, argv, "--epochs", 500);
        std::string text = (dataPath == "--builtin") ? CORPUS : readFile(dataPath);

        std::vector<std::string> tokens = tokenize(text);
        BigramModel model(tokens, /*D=*/8, /*lr=*/0.1, /*seed=*/42);

        std::vector<std::pair<int,int>> pairs;
        for (size_t i = 0; i + 1 < tokens.size(); ++i)
            pairs.push_back({ model.idOf(tokens[i]), model.idOf(tokens[i+1]) });

        std::cout << "training v1 (one token of memory) on " << pairs.size()
                  << " pairs, vocab " << model.vocabularySize() << "\n";
        for (int epoch = 1; epoch <= epochs; ++epoch) {
            double total = 0.0;
            for (auto& pr : pairs) total += model.trainOnPair(pr.first, pr.second);
            if (epoch == 1 || epoch % 100 == 0 || epoch == epochs)
                std::cout << "  epoch " << std::setw(5) << epoch << "   loss "
                          << std::fixed << std::setprecision(4) << total/pairs.size() << "\n";
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
        BigramModel model = BigramModel::loadFrom(modelPath);
        std::mt19937 rng(std::random_device{}());
        std::cout << autocomplete(model, prompt, maxNew, sample, rng) << "\n";
        return 0;
    }

    usage(); return 1;
}
