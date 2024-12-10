#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif

#include "unet-image.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>
#include <algorithm>
#include <fstream>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

struct unet_conv2d_layer {
    struct ggml_tensor * weights;
    struct ggml_tensor * biases;
    struct ggml_tensor * scales;
    struct ggml_tensor * beta;
    struct ggml_tensor * rolling_mean;
    struct ggml_tensor * rolling_variance;
    int padding = 1;
    int strike = 1;
    bool batch_normalize = true;
    bool load_next = false;
    bool skip_load = false;
    bool activate = true; 
   
    const char * name_conv = NULL;
    const char * name_bn = NULL;
};

struct unet_model {
    int width = 224;
    int height = 224;
    std::vector<unet_conv2d_layer> conv2d_layers;
    ggml_backend_t backend = NULL;
    ggml_backend_buffer_t buffer;
    struct ggml_context * ctx;
};

struct unet_params {
    float thresh          = 0.15f;
    std::string model     = "modelunet.gguf";
    std::vector<std::string> fname_inp;
    std::vector<std::string> fname_out;
    int threads;
};