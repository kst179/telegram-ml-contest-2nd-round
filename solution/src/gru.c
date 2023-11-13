#include <pthread.h>
#include <stdio.h>

#include "matrix.h"
#include "gru.h"

#ifdef EMBED_WEIGHTS
#include "embed_gru.h"
#endif


GRU* createGRU(const char* path) {
    int num_layers;
    int is_bidirectional;
    int num_threads_per_dir;

    int hidden_dim;
    int embedding_size;
    int num_classes;

    FILE* file = fopen(path, "rb");
    
    fread(&num_layers, sizeof(num_layers), 1, file);
    fread(&is_bidirectional, sizeof(is_bidirectional), 1, file);
    fread(&num_threads_per_dir, sizeof(num_threads_per_dir), 1, file);

    fread(&hidden_dim, sizeof(hidden_dim), 1, file);
    fread(&embedding_size, sizeof(embedding_size), 1, file);
    fread(&num_classes, sizeof(embedding_size), 1, file);

    GRU* gru = malloc(sizeof(GRU));

    gru->num_threads_per_dir = num_threads_per_dir;
    gru->num_layers = num_layers;
    gru->is_bidirectional = is_bidirectional;

    gru->hidden_dim = hidden_dim;
    gru->embedding_size = embedding_size;
    gru->num_classes = num_classes;

    int num_directions = is_bidirectional ? 2 : 1;

    gru->embeddings = malloc(sizeof(Matrix) * num_directions);
    
    for (int dir = 0; dir < num_directions; dir++) {
        gru->embeddings[dir] = matReadFromFile(embedding_size, hidden_dim * 3, file);
    }

    gru->weights_h = malloc(sizeof(Matrix) * num_layers * num_directions);
    gru->bias_h = malloc(sizeof(Matrix) * num_layers * num_directions);

    gru->weights_i = malloc(sizeof(Matrix) * (num_layers - 1) * num_directions);
    gru->bias_i = malloc(sizeof(Matrix) * (num_layers - 1) * num_directions);

    for (int layer = 0; layer < num_layers; ++layer) {
        Matrix weights, bias;

        for (int dir = 0; dir < num_directions; dir++) {
            int param_idx = layer * num_directions + dir;
            gru->weights_h[param_idx] = matReadFromFile(hidden_dim * 3, hidden_dim, file);
            gru->bias_h[param_idx] = matReadFromFile(1, hidden_dim * 3, file);
        
            if (layer != 0) {
                param_idx = (layer - 1) * num_directions + dir;

                gru->weights_i[param_idx] = 
                    matReadFromFile(hidden_dim * 3, hidden_dim * num_directions, file);

                gru->bias_i[param_idx] = 
                    matReadFromFile(1, hidden_dim * 3, file);
            }
        }
    }

    gru->weights_classifier = matReadFromFile(num_classes, hidden_dim * num_directions, file);
    gru->bias_classifier = matReadFromFile(1, num_classes, file);

    fclose(file);

    gru->last_hidden_state = matCreate(1, hidden_dim * num_directions);
    gru->logits = matCreate(1, num_classes);
    
    gru->rzn = malloc(sizeof(Matrix) * num_directions);
    gru->rzn_i = malloc(sizeof(Matrix) * num_directions);

    for (int dir = 0; dir < num_directions; dir++) {
        gru->rzn[dir] = matCreate(1, hidden_dim * 3);
        gru->rzn_i[dir] = matCreate(1, hidden_dim * 3);
    }

    return gru;
}

void gruRunLayer(
    GRU* gru, 
    int* tokens, 
    int num_tokens, 
    int layer, 
    int dir,
    int thread_idx,
    Matrix inputs,              // in:  [num_tokens, hidden_dim * num_directions]
    Matrix hidden_states,       // out: [num_tokens, hidden_dim]
    pthread_barrier_t* dir_barrier
) {
    const int num_directions = gru->is_bidirectional ? 2 : 1; 
    const int hid_dim = gru->hidden_dim;
    const int num_threads_per_dir = gru->num_threads_per_dir;

    int dims_per_thread = hid_dim / num_threads_per_dir;

    int param_idx = layer * num_directions + dir;

    int r1 = dims_per_thread * thread_idx;
    int r2 = dims_per_thread * (thread_idx + 1);

    Matrix weights_h = matSliceRows(gru->weights_h[param_idx], r1 * 3, r2 * 3);
    Matrix bias_h = matSliceCols(gru->bias_h[param_idx], r1 * 3, r2 * 3);

    Matrix rzn = matSliceCols(gru->rzn[dir], r1 * 3, r2 * 3);

    Matrix rz = matSliceCols(rzn, 0, dims_per_thread * 2);
    Matrix r = matSliceCols(rzn, 0, dims_per_thread);
    Matrix z = matSliceCols(rzn, dims_per_thread, dims_per_thread * 2);
    Matrix n = matSliceCols(rzn, dims_per_thread * 2, dims_per_thread * 3);

    Matrix weights_i;
    Matrix bias_i;
    Matrix rzn_i;
    Matrix rz_i;
    Matrix n_i;
    
    if (layer != 0) {
        param_idx = (layer - 1) * num_directions + dir;

        weights_i = matSliceRows(gru->weights_i[param_idx], r1 * 3, r2 * 3);
        bias_i = matSliceCols(gru->bias_i[param_idx], r1 * 3, r2 * 3);
        rzn_i = matSliceCols(gru->rzn_i[dir], r1 * 3, r2 * 3);

        rz_i = matSliceCols(rzn_i, 0, dims_per_thread * 2);
        n_i = matSliceCols(rzn_i, dims_per_thread * 2, dims_per_thread * 3);
    }

    Matrix prev_hidden_state;
    Matrix prev_hidden_state_chunk;
    Matrix out_hidden_state_chunk;

    int token_idx = 0;

    for (int i = 0; i < num_tokens; i++) {
        int prev_token_idx = token_idx;
        token_idx = dir ? num_tokens - i - 1 : i;

        // dir is one-directional hidden state from previous cell, and output for current one
        prev_hidden_state_chunk = out_hidden_state_chunk;
        out_hidden_state_chunk = matSubmatrix(hidden_states, token_idx, token_idx+1, r1, r2);

        if (layer == 0) {
            int token_id = tokens[token_idx];
            rzn_i = matSubmatrix(gru->embeddings[dir], token_id, token_id + 1, r1 * 3, r2 * 3);

            rz_i = matSliceCols(rzn_i, 0, dims_per_thread * 2);
            n_i = matSliceCols(rzn_i, dims_per_thread * 2, dims_per_thread * 3);
        } else {
            // bidirectional hidden state from previous layer (2 * hid_dim size)
            Matrix input = matSelectRow(inputs, token_idx);

            // rzn_i = bias_i + weights_i @ hidden_state (from previous layer)
            matCopy(bias_i, rzn_i);
            matVecProduct(weights_i, input, rzn_i);
        }

        // rzn = bias_h + weights_h @ hidden_state
        matCopy(bias_h, rzn);
        
        // first hidden state is 0 so skip mat product
        if (i != 0) {
            Matrix prev_hidden_state = matSelectRow(hidden_states, prev_token_idx);
            matVecProduct(weights_h, prev_hidden_state, rzn);
        }

        // rz = sigmoid(rz_i + rz)
        matSum(rz_i, rz, rz);
        matInplaceSigmoid(rz);

        // n = tanh(n_i + r * n)
        matHProduct(r, n, n);
        matSum(n_i, n, n);
        matInplaceTanh(n);
        
        // h = (1 - z) * n + z * h, h = 0 if first iter
        if (i == 0) {
            matSlerpZero(n, z, out_hidden_state_chunk);
        } else {
            matSlerp(n, prev_hidden_state_chunk, z, out_hidden_state_chunk);
        }

        // fprintf(stderr, "thread %d \tdir %d \tlayer %d\t i %d\n", thread_idx + num_threads_per_dir * dir, dir, layer, i);
        if (num_threads_per_dir > 1) {
            pthread_barrier_wait(dir_barrier);
        }
    }
}

void swap(Matrix *a, Matrix *b) {
    Matrix c = *a;
    *a = *b;
    *b = c;
}

void* gruRunLayersThread(void* pthread_args) {
    GRUThreadArgs* args = (GRUThreadArgs*)pthread_args;
    GRU* gru = args->gru;
    const int dir = args->dir;
    const int hid_dim = gru->hidden_dim;

    for (int layer = 0; layer < gru->num_layers; layer++) {
        Matrix dir_hidden_states = matSliceCols(args->hidden_states, 
                                                hid_dim * dir, hid_dim * dir + hid_dim); 

        gruRunLayer(
            gru, 
            args->tokens, 
            args->num_tokens, 
            layer, 
            dir, 
            args->thread_idx, 
            args->inputs, 
            dir_hidden_states, 
            args->dir_barrier
        );

        swap(&args->inputs, &args->hidden_states);
        pthread_barrier_wait(args->layer_barrier);
    }

    return NULL;
}

Matrix getLastStateGRU(GRU* gru, int* tokens, int num_tokens) {
    const int num_directions = gru->is_bidirectional ? 2 : 1;
    const int hid_dim = gru->hidden_dim;
    const int num_threads_per_dir = gru->num_threads_per_dir;
    const int num_threads = num_threads_per_dir * num_directions;

    // inputs are hidden states from previous layer, 
    // first layer uses embeddings instead of inputs
    Matrix inputs = matCreate(num_tokens, hid_dim * num_directions);

    // hidden_states is output hidden states for current layer (but last)
    // after each layer inputs and hidden_states are swapped
    Matrix hidden_states = matCreate(num_tokens, hid_dim * num_directions);
    
    // num_threads - 1 are to be spawned cause main thread also does calculation for one layer
    pthread_t* threads;
    GRUThreadArgs* thread_args;

    if (num_threads > 1) {
        threads = malloc(sizeof(pthread_t) * (num_threads - 1));
        thread_args = malloc(sizeof(GRUThreadArgs) * (num_threads - 1));
    }

    // threads in same direction are synced with dir barriers
    // to split hidden state write and read on next iteration
    pthread_barrier_t* dir_barriers = malloc(sizeof(pthread_barrier_t) * num_directions);
    // layer barrier syncs threads between different layers
    pthread_barrier_t layer_barrier;

    for (int dir = 0; dir < num_directions; dir++) {
        pthread_barrier_init(&dir_barriers[dir], NULL, num_threads_per_dir);
    }

    pthread_barrier_init(&layer_barrier, NULL, num_threads);

    for (int dir = 0; dir < num_directions; dir++) {
        for (int local_tread_idx = 0; local_tread_idx < num_threads_per_dir; local_tread_idx++) {
            int thread_idx = num_threads_per_dir * dir + local_tread_idx;
            if (thread_idx == num_threads - 1) {
                continue;
            }

            GRUThreadArgs* args = &thread_args[thread_idx];
            
            args->gru = gru;
            args->tokens = tokens;
            args->num_tokens = num_tokens;
            args->dir = dir;
            args->thread_idx = local_tread_idx;
            args->inputs = inputs;
            args->hidden_states = hidden_states;
            args->dir_barrier = &dir_barriers[dir];
            args->layer_barrier = &layer_barrier;

            pthread_create(&threads[thread_idx], NULL,
                            gruRunLayersThread, (void*)args);
        }
    }

    // last thread is the main one
    int local_thread_idx = num_threads_per_dir - 1;
    int dir = num_directions - 1;

    for (int layer = 0; layer < gru->num_layers; layer++) {
        Matrix dir_hidden_states = matSliceCols(hidden_states, 
                                                hid_dim * dir, hid_dim * dir + hid_dim); 

        gruRunLayer(gru, tokens, num_tokens, layer, dir, local_thread_idx, 
                    inputs, dir_hidden_states, &dir_barriers[dir]);

        // wait other threads to compute the hidden states of this layer
        // pthread_barrier_wait(&layer_barrier);

        // swap input and output hidden states        
        if (layer != gru->num_layers - 1) {
            swap(&inputs, &hidden_states);
        }

        // release other threads to start calculating next layer
        pthread_barrier_wait(&layer_barrier);
    }

    for (int thread_idx = 0; thread_idx < num_threads - 1; thread_idx++) {
        pthread_join(threads[thread_idx], NULL);
    }

    if (gru->is_bidirectional) {
        matCopy(matSubmatrix(hidden_states, num_tokens - 1, num_tokens, 0, hid_dim), 
                matSliceCols(gru->last_hidden_state, 0, hid_dim));

        matCopy(matSubmatrix(hidden_states, 0, 1, hid_dim, hid_dim * 2), 
                matSliceCols(gru->last_hidden_state, hid_dim, hid_dim * 2));
    } else {
        matCopy(matSelectRow(hidden_states, num_tokens - 1), gru->last_hidden_state);
    }

    for (int dir = 0; dir < num_directions; dir++) {
        pthread_barrier_destroy(&dir_barriers[dir]);
    }

    if (num_threads > 1) {
        free(threads);
        free(thread_args);
    }
    
    free(dir_barriers);

    matFree(inputs);
    matFree(hidden_states);

    return gru->last_hidden_state;
}

int predictGRU(GRU* gru, int* tokens, int num_tokens) {
    Matrix last_state = getLastStateGRU(gru, tokens, num_tokens);

    matCopy(gru->bias_classifier, gru->logits);
    matVecProduct(gru->weights_classifier, last_state, gru->logits);

    return matVecArgmax(gru->logits);
}

void freeGRU(GRU** gru) {
    matFree((*gru)->weights_classifier);
    matFree((*gru)->bias_classifier);

    matFree((*gru)->last_hidden_state);
    matFree((*gru)->logits);

    int num_directions = (*gru)->is_bidirectional ? 2 : 1;

    for (int dir = 0; dir < num_directions; dir++) {
        matFree((*gru)->embeddings[dir]);

        matFree((*gru)->rzn[dir]);
        matFree((*gru)->rzn_i[dir]);

        for (int layer = 0; layer < (*gru)->num_layers; layer++) {
            matFree((*gru)->weights_h[layer * num_directions + dir]);
            matFree((*gru)->bias_h[layer * num_directions + dir]);

            if (layer != 0) {
                matFree((*gru)->weights_i[(layer - 1) * num_directions + dir]);
                matFree((*gru)->bias_i[(layer - 1) * num_directions + dir]);
            }
        }
    }

    free((*gru)->embeddings);
    free((*gru)->rzn);
    free((*gru)->rzn_i);
    free((*gru)->weights_h);
    free((*gru)->bias_h);
    free((*gru)->weights_i);
    free((*gru)->bias_i);
    
    free(*gru);
    *gru = NULL;
}

void saveEmbedGRU(GRU* gru, const char* path) {
#if 0
    FILE* file = fopen(path, "w");

    GRU gru_copy = *gru;
    
    size_t offset = sizeof(GRU);
    offset = (offset + 31) / 32 * 32; // 32 bytes aligned

    gru_copy.embeddings.data = (float*)offset; 
    offset += matSizeBytes(gru->embeddings);
    
    gru_copy.weights_h.data = (float*)offset; 
    offset += matSizeBytes(gru->weights_h);
    
    gru_copy.bias_h.data = (float*)offset; 
    offset += matSizeBytes(gru->bias_h);
  
    gru_copy.weights_classifier.data = (float*)offset;
    offset += matSizeBytes(gru->weights_classifier);

    gru_copy.bias_classifier.data = (float*)offset;
    offset += matSizeBytes(gru->bias_classifier);
    
    gru_copy.hidden_state.data = (float*)offset;
    offset += matSizeBytes(gru->hidden_state);

    gru_copy.logits.data = (float*)offset;
    offset += matSizeBytes(gru->logits);
    
    gru_copy.rzn.data = (float*)offset;
    offset += matSizeBytes(gru->rzn);
    
    gru_copy.r = matSubmatrix(gru_copy.rzn, 0, 1, 0, gru->hidden_dim);
    gru_copy.z = matSubmatrix(gru_copy.rzn, 0, 1, gru->hidden_dim, gru->hidden_dim * 2);
    gru_copy.n = matSubmatrix(gru_copy.rzn, 0, 1, gru->hidden_dim * 2, gru->hidden_dim * 3);
    gru_copy.rz = matSubmatrix(gru_copy.rzn, 0, 1, 0, gru->hidden_dim * 2);

    offset = sizeof(GRU);
    offset = (offset + 31) / 32 * 32; // 32 bytes aligned

    fprintf(file, "#ifndef EMBEDDED_GRU\n#define EMBEDDED_GRU\n\n unsigned int GRU_DATA[] __attribute__((aligned(32))) = {\n");

    int j = 1;
    int k = 0;
    for(k = 0; k < sizeof(gru_copy) / 4; k++) {
        fprintf(file, "0x%08X,", ((unsigned int*)&gru_copy)[k]);
        if (j++ % 16 == 0) { fprintf(file, "\n"); }
    }
    for (; k < offset / 4; k++) {
        fprintf(file, "0x%08X,", 0u);
        if (j++ % 16 == 0) { fprintf(file, "\n"); }
    }
    
    for(int i = 0; i < matSize(gru->embeddings); i++) {
        fprintf(file, "0x%08X,", ((unsigned int*)gru->embeddings.data)[i]);
        if (j++ % 16 == 0) { fprintf(file, "\n"); }
    }

    for(int i = 0; i < matSize(gru->weights_h); i++) {
        fprintf(file, "0x%08X,", ((unsigned int*)gru->weights_h.data)[i]);
        if (j++ % 16 == 0) { fprintf(file, "\n"); }
    }

    for(int i = 0; i < matSize(gru->bias_h); i++) {
        fprintf(file, "0x%08X,", ((unsigned int*)gru->bias_h.data)[i]);
        if (j++ % 16 == 0) { fprintf(file, "\n"); }
    }

    for(int i = 0; i < matSize(gru->weights_classifier); i++) {
        fprintf(file, "0x%08X,", ((unsigned int*)gru->weights_classifier.data)[i]);
        if (j++ % 16 == 0) { fprintf(file, "\n"); }
    }

    for(int i = 0; i < matSize(gru->bias_classifier); i++) {
        fprintf(file, "0x%08X,", ((unsigned int*)gru->bias_classifier.data)[i]);
        if (j++ % 16 == 0) { fprintf(file, "\n"); }
    }

    for(int i = 0; i < matSize(gru->hidden_state); i++) {
        fprintf(file, "0x%08X,", 0u);
        if (j++ % 16 == 0) { fprintf(file, "\n"); }
    }

    for(int i = 0; i < matSize(gru->logits); i++) {
        fprintf(file, "0x%08X,", 0u);
        if (j++ % 16 == 0) { fprintf(file, "\n"); }
    }

    for(int i = 0; i < matSize(gru->rzn); i++) {
        fprintf(file, "0x%08X,", 0u);
        if (j++ % 16 == 0) { fprintf(file, "\n"); }
    }

    fprintf(file, "0x%08X\n};\n\n#endif\n", 0u);
    fclose(file);
#endif
}

GRU* loadEmbedGRU() {
#ifdef EMBEDDED_GRU

    GRU* gru = (GRU*)&GRU_DATA[0];
    gru->embeddings.data = (float*)((unsigned char*)GRU_DATA + (size_t)gru->embeddings.data);
    gru->weights_h.data = (float*)((unsigned char*)GRU_DATA + (size_t)gru->weights_h.data);
    gru->bias_h.data = (float*)((unsigned char*)GRU_DATA + (size_t)gru->bias_h.data);
    gru->weights_classifier.data = (float*)((unsigned char*)GRU_DATA + (size_t)gru->weights_classifier.data);
    gru->bias_classifier.data = (float*)((unsigned char*)GRU_DATA + (size_t)gru->bias_classifier.data);
    gru->hidden_state.data = (float*)((unsigned char*)GRU_DATA + (size_t)gru->hidden_state.data);
    gru->logits.data = (float*)((unsigned char*)GRU_DATA + (size_t)gru->logits.data);
    gru->rzn.data = (float*)((unsigned char*)GRU_DATA + (size_t)gru->rzn.data);

    gru->r = submatrix(gru->rzn, 0, 1, 0, gru->hidden_dim);
    gru->z = submatrix(gru->rzn, 0, 1, gru->hidden_dim, gru->hidden_dim * 2);
    gru->n = submatrix(gru->rzn, 0, 1, gru->hidden_dim * 2, gru->hidden_dim * 3);
    gru->rz = submatrix(gru->rzn, 0, 1, 0, gru->hidden_dim * 2);

    return gru;
    
#else

    return NULL;
    
#endif
}
