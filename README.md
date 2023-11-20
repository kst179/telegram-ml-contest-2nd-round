# Telegram Programming Language Detection ML Contest Round 2

This repo contains my solution of the second round of the [TG ML contest](https://contest.com/docs/ML-Competition-2023).
You can find my first round solution [here](https://github.com/kst179/telegram-ml-contest)

# Task
The task was to define the programming language only by the code snippet. The inference should be fast: the text of **4096 bytes** should be processed in less than **50 ms**.

# Dataset
I used the [same dataset](https://disk.yandex.ru/d/emA4ymSc2gNkZQ) as in the first round, however I've cleaned it a little bit: deleted empty files, or files with only space characters, delete some outlier files, which I found in FunC category. Also data from organizers were added to classify other/code categories. Dataset was splitted in 4 splits:
    * train_gru &mdash; was used for pretraining GRU.
    * train_svc &mdash; was used only for testing as I decided to not use svc in this round
    * test &mdash; for testing
    * extra &mdash; was added by organizers three days before deadline, was used for fine-tuning of binary classificaiton model

dataset also contains the origins of the data it can be one of "gh" and "tg" for data from github and telegram. 

# Solution
In this round I've used only tokenizer and two GRU networks for classification.

## Tokenizer
The tokenizer is the same as in previous round, however I've fixed some bugs (previous version could return unk tokens in some cases when some tokens that can represent the text exist). And retrained it on the novel data (but took only up to 500 files from github data to train each language, and to avoid crashing on memory out). The vocab size is `2**15` as in previous round.

## GRU
The first round learned us that you can classify the languages well, but your model will fail due to bad other/code detection, and on new data we can see that `other : code` distribution is roughly `8 : 1`. That's why I chose to focus on the code/other binary classification rather than learning single model to predict anything. Also I've realised that I have not used any parallel processing in the first round, so I add second GRU that predicts binary label and do it (almost) for free, because runs in a separate thread.

The GRU archeticture also was modernized, now I was using `3-layers bidirectional` GRUs with direction interconnection between layers (each next layer uses information from the hidden states from both directions). I've also increased the size of hidden state to `104`. The calculation of both directions is parallelized into two threads, so bidirectionality of the models also (almost) free. Two models run in parallel in about `40 ms` (per 4kB text) on my laptop, and can be efficient on any 4+ core processors with AVX2 support. I tried to parallelize the one direction layer calculation, but scince threads should be syncronized after each token, the overhead is too big to get some advantages (however this implementation can theoreticaly work faster on several thread in cases when the hidden space is bigger).

Moreover, I used the AVX implementation of exponent function from the [avx mathfun](https://github.com/reyoung/avx_mathfun/tree/master) library to speed up a little the exponent calculation in the sigmoid and tanh functions.

## Training
I've trained models in two setups: binary (for code/other) and multiclass (for specific languages). Both networks was trained on `train_gru` data, however the binary model was then finetuned on the `extra` data split. The target metric for the binary model was **AUC ROC** because the classes are unbalanced and I wanted to select optimal threshold later.

Another novelty was the data sampling algorithm, now the sampler from github code tries to mimic the snippet character number distribution based on the telegram data. For that I've learned parameters of log-logistic (aka fisk) distribution on telegram data, then sampled target size of the snippet and selected the number of lines from gh files so that the total size is close to the target one. This approach produces shorter snippets (90% are shorter than 200 symbols), so the model is forced to recognize language from small amount of information. However, scince the original data has long snippets in some files, I selected the distribution with heavy tail (and there is no some statistical reason or other intuition behind this choice, it just fitted well).

I pretrained both networks on rented `NVIDIA GeForce GTX 1080 x2` and `1080 Ti x2` for about **3 days** (70 hours), then finetuned the binary one on a laptop `GeForce RTX 3050` (which is faster but smaller in VRAM) for about **30 minutes** (until it overfitted on a small dataset). The pretraining parameters: optimizer: `adam`, `lr = 1e-4`, `epochs=100`, `batch_size=64` for binary setup and `batch_size=48` with `gradient_accumulation = 2` on multilanguages setup. The finetuning was made only on `extra` split, which was not seen on training, and model was optimized with `lr=1e-5` to avoid catastrofic forgetting of the original data and better fit to the real snippets from telegram, `batch_size = 8`, `gradient_accumulation = 4`. 

The threshold for the binary classification was selected to maximize the **F1** score on the test data, so the F1 score is about `88%` which gives accuracy about `98%` of the other/code recognition (but the constant model, which always returns "other" gives about `91%`, so accuracy is a bad metric here).


## Build

The C inference lib is located in the `solution` dir. To build it simply run following commands
```bash
mkdir build && cd build                   # create build dir
cmake .. -DCMAKE_BUILD_TYPE=Release \     # configure
         -DUSE_AVX_EXP=ON                 # if set, uses avx vectors for exponent calculation
make                                      # build
cd ..                                     # return to solution dir
```

Then you can test builded solution by creating file named `input.txt` (should be located in the `solution` dir) with the code snippet and running (also from the `solution` dir):
```bash
echo "print('Hello, world!')" > input.txt # fill the input.txt
./build/test                              # run test program
```

The simple test will run your snippet several times and return prediction and mean time per run.

To use the solution as shared lib you can copy and add `solution/build/libtglang.so` to your project, along with the files in `solution/resources/*.bin`. It is important to keep `resources` dir in the same place as your final binary file (or script if you load shared lib into interactive languages). However there is an option to embed weights in the lib. To do so build project once, then run `embed_weights` and recompile with `EMBED_WEIGHTS=ON` 
```bash
./build/embed_weights       # create embed_*.h files
cd build                    # return to build dir
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DUSE_AVX_EXP=ON \
         -DEMBED_WEIGHTS=ON # reconfigure to build with embedded weights
make                        # build with embedded weights
```
The compilation with weights embedding will take some time, however the resulting `libtglang.so` shared library will run anywhere without need of `resources` folder.

## Reproducing results

The results can be easily reproduced using scripts in the `training` dir:

* `train_tokenizer` &mdash; trains the tokenizer on data and saves it into binary file which can be used from C inference lib.
* `train_gru` &mdash; trains gru and saves it to the artifacts dir
* `transform_gru` &mdash; converts pytorch weights to binary one

Also some notebooks are provided, where I made data filtering, and split, gru testing, and search for optimal threshold. However the code is a little bit trashy there. 

Additional modules:
* `cgru` &mdash; api to call C gru inference from python
* `ctokenizer` &mdash; same api for tokenizer
* `gh_dataset` &mdash; contains dataset object to fetch data samples
* `gru_model` &mdash; describes the gru network
* `gru_trainer` &mdash; implements the trainer to fit gru
* `languages_list_old` &mdash; languages list from 1st round (enum class)
* `languages_list` &mdash; languages list from 2nd round
* `paths` &mdash; useful constant paths
* `test_final_model` &mdash; to test resulting model on my data (actually it was not used in this round, everything was tested in the notebooks)
