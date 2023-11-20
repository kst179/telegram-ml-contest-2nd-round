#include <time.h>
#include <stdio.h>
#include <stdlib.h>

#include <pthread.h>

#include "tokenizer.h"
#include "svc.h"
#include "gru.h"

#include "lang_names.h"
#include "tglang.h"

typedef struct Args {
    GRU* gru;
    int* tokens;
    int num_tokens;
} Args;

void* runPredictThread(void* args_) {
    Args* args = (Args*)args_;

    gruPredict(args->gru, args->tokens, args->num_tokens);
    return NULL;
}

int main() {
    FILE* file = fopen("input.txt", "r");

    fseek(file, 0, SEEK_END);
    int size = ftell(file);
    fseek(file, 0, SEEK_SET);

    char* text = malloc(sizeof(char) * (size + 1));
    fread(text, sizeof(char), size, file);
    text[size] = 0;
    fclose(file);

    Tokenizer* tokenizer = createTokenizer("resources/tokenizer_vocab.bin");
    GRU* gru = gruCreate("resources/gru_weights.bin");
    GRU* gru2 = gruCreate("resources/gru_weights.bin");
    SVC* svc = createSVC("resources/svc_weights.bin");

    clock_t start, end;
    double cpu_time_used;
    time_t wall_time_start, wall_time_end;

    int n_repeats = 100;

    int prediction;

    start = clock();
    time(&wall_time_start);

    pthread_t thread1, thread2;

    for (int i = 0; i < n_repeats; ++i) {
        // int num_tokens;
        // int* tokens;

        // tokenize(tokenizer, text, &num_tokens, &tokens);

        // Args args1 = {gru, tokens, num_tokens};
        // Args args2 = {gru2, tokens, num_tokens};

        // pthread_create(&thread1, NULL, runPredictThread, &args1);
        // pthread_create(&thread2, NULL, runPredictThread, &args2);

        // pthread_join(thread1, NULL);
        // pthread_join(thread2, NULL);

        // int prediction = gruPredict(gru, tokens, num_tokens);

        // Matrix gru_last_state = gruGetLastState(gru, tokens, num_tokens);
        // int prediction = predictSVC(svc, tokens, num_tokens, gru_last_state);
        
        // free(tokens);

        prediction = tglang_detect_programming_language(text);
    }

    end = clock();
    time(&wall_time_end);

    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    double wall_time_used = difftime(wall_time_end, wall_time_start);
    
    printf("Prediction: %s\n", lang_names[prediction]);
    printf("CPU time per run: %f ms\n", cpu_time_used  * 1000 / n_repeats);
    printf("CPU time per 4Kb: %f ms\n", cpu_time_used  * 1000 * 4096 / n_repeats / size);
    printf("Wall time: %f s\n", wall_time_used);
    printf("Wall time per run: %f ms\n", wall_time_used * 1000 / n_repeats);
    printf("Wall time per 4Kb: %f ms\n", wall_time_used * 1000 * 4096 / n_repeats / size);

    free(text);

    freeGRU(&gru);
    freeSVC(&svc);
    freeTokenizer(&tokenizer);

    return 0;
}
