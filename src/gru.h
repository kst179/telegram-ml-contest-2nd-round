#include <pthread.h>

#include "matrix.h"

typedef struct GRU {
    int num_threads_per_dir;
    int num_layers;
    int is_bidirectional;

    // Weights
    Matrix* embeddings;
    
    Matrix* weights_h;
    Matrix* bias_h;

    Matrix* weights_i;
    Matrix* bias_i;

    Matrix weights_classifier;
    Matrix bias_classifier;

    // Preallocated matrices for calculations
    Matrix last_hidden_state;
    Matrix logits;
    Matrix* rzn;
    Matrix* rzn_i;

    int hidden_dim;
    int embedding_size;
    int num_classes;
} GRU;

typedef struct GRUThreadArgs {
    pthread_barrier_t* dir_barrier;
    pthread_barrier_t* layer_barrier;
    GRU* gru;
    int* tokens;
    int num_tokens;
    int dir;
    int thread_idx;
    Matrix inputs;
    Matrix hidden_states;
} GRUThreadArgs;

GRU* gruCreate(const char* path);
Matrix gruGetLastState(GRU* gru, int* tokens, int num_tokens);
Matrix gruGetLogits(GRU* gru, int* tokens, int num_tokens);
int gruPredict(GRU* gru, int* tokens, int num_tokens);
void freeGRU(GRU** gru);

void gruRunLayer(
    GRU* gru, 
    int* tokens, 
    int num_tokens, 
    int layer, 
    int dir,
    int thread_idx,
    Matrix inputs,
    Matrix hidden_states,
    pthread_barrier_t* dir_barrier
);
void* gruRunLayersThread(void* pthread_args);

void gruSaveEmbed(GRU* gru, const char* path, int gru_id);
GRU* gruLoadEmbed(int gru_id);