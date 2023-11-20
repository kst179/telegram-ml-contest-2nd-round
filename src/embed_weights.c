#include "tokenizer.h"
#include "svc.h"
#include "lang_names.h"
#include "gru.h"

int main() {
    GRU* gru_bin = gruCreate("resources/gru_binary.bin");
    GRU* gru_lang = gruCreate("resources/gru_lang.bin");

    // SVC* svc = createSVC("resources/svc_weights.bin");
    Tokenizer* tokenizer = createTokenizer("resources/tokenizer_vocab.bin");

    gruSaveEmbed(gru_bin, "src/embed_gru_binary.h", 0);
    gruSaveEmbed(gru_lang, "src/embed_gru_lang.h", 1);
    // saveEmbedSVC(svc, "src/embed_svc.h");
    saveEmbedTokenizer(tokenizer, "src/embed_tokenizer.h");

    freeGRU(&gru_bin);
    freeGRU(&gru_lang);
    // freeSVC(&svc);
    freeTokenizer(&tokenizer);
    
    return 0;
}
