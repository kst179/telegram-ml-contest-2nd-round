#include "tglang.h"

#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <math.h>

#include "tokenizer.h"
#include "gru.h"
#include "svc.h"

typedef struct LangPredictThreadArgs {
    GRU* gru;
    int* tokens;
    int num_tokens;
    int prediction;
} LangPredictThreadArgs;

void* langPredictThread(void* args_) {
    LangPredictThreadArgs* args = args_;
    Matrix logits = gruGetLogits(args->gru, args->tokens, args->num_tokens);
    logits.data[0] = -INFINITY; // lang net never predicts "other"
    args->prediction = matVecArgmax(logits);
}

const float BIN_THRESHOLD = 0.666;

enum TglangLanguage tglang_detect_programming_language(const char *text) {
    static Tokenizer* tokenizer = NULL;
    static GRU* gru_bin = NULL;
    static GRU* gru_lang = NULL;
    static SVC* svc = NULL;

#ifdef EMBED_WEIGHTS
    if (tokenizer == NULL) { tokenizer = loadEmbedTokenizer(); }
    if (gru_bin == NULL) { gru_bin = gruLoadEmbed(0); }
    if (gru_lang == NULL) { gru_lang = gruLoadEmbed(1); }
    if (svc == NULL) { svc = loadEmbedSVC(); }
#else
    if (tokenizer == NULL) { tokenizer = createTokenizer("resources/tokenizer_vocab.bin"); }
    if (gru_lang == NULL) { gru_bin = gruCreate("resources/gru_binary.bin"); }
    if (gru_lang == NULL) { gru_lang = gruCreate("resources/gru_lang.bin"); }
    if (svc == NULL) { svc = createSVC("resources/svc_weights.bin"); }
#endif

    int num_tokens;
    int* tokens;

    tokenize(tokenizer, text, &num_tokens, &tokens);

    LangPredictThreadArgs args = { gru_lang, tokens, num_tokens, 0 };

    pthread_t lang_thread;
    pthread_create(&lang_thread, NULL, langPredictThread, (void*)&args);

    Matrix logits = gruGetLogits(gru_bin, tokens, num_tokens);
    
    float a = logits.data[0];
    float b = logits.data[1];
    float max = (a > b) ? a : b;
    float proba = exp(b - max) / (exp(a - max) + exp(b - max));

    int is_code = proba >= BIN_THRESHOLD;

    pthread_join(lang_thread, NULL);
    free(tokens);

    if (!is_code) {
        return TGLANG_LANGUAGE_OTHER;
    }

    return (enum TglangLanguage)args.prediction;
}
