#include "unet.h"

void unet_print_usage(int argc, char ** argv, const unet_params & params) {
    fprintf(stderr, "usage: %s [options]\n", argv[0]);
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h, --help            show this help message and exit\n");
    fprintf(stderr, "  -th T, --thresh T     detection threshold (default: %.2f)\n", params.thresh);
    fprintf(stderr, "  -m FNAME, --model FNAME\n");
    fprintf(stderr, "                        model path (default: %s)\n", params.model.c_str());
    fprintf(stderr, "  -i FNAME, --inp FNAME\n");
    // fprintf(stderr, "                        input file (default: %s)\n", params.fname_inp.c_str());
    fprintf(stderr, "  -o FNAME, --out FNAME\n");
    // fprintf(stderr, "                        output file (default: %s)\n", params.fname_out.c_str());
    fprintf(stderr, "\n");
}

bool unet_params_parse(int argc, char ** argv, unet_params & params) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-th" || arg == "--thresh") {
            params.thresh = std::stof(argv[++i]);
        } else if (arg == "-m" || arg == "--model") {
            params.model = argv[++i];
        } else if (arg == "-t" || arg == "--threads") {
            params.threads = std::stof(argv[++i]);
        } else if (arg == "-i" || arg == "--inp") {
            while (++i < argc && argv[i][0] != '-') {
                params.fname_inp.push_back(argv[i]);
            }
            --i;          
        } else if (arg == "-o" || arg == "--out") {
            while (++i < argc && argv[i][0] != '-') {
                params.fname_out.push_back(argv[i]);
            }
            --i; 
        } else if (arg == "-h" || arg == "--help") {
            unet_print_usage(argc, argv, params);
            exit(0);
        } else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            unet_print_usage(argc, argv, params);
            exit(0);
        }
    }
    return true;
}

static bool load_model(const std::string & fname, unet_model & model, int n_threads = 1) 
{
    // initialize the backend, use CPU or CUDA
#ifdef GGML_USE_CUDA
    fprintf(stderr, "%s: using CUDA backend\n", __func__);
    model.backend = ggml_backend_cuda_init(0); // init device 0
    if(!model.backend)
    {
        fprintf(stderr, "%s: ggml_backend_cuda_init() failed\n", __func__);
    }
#endif

     // if there aren't GPU Backends fallback to CPU backend
    if (!model.backend) {
        model.backend = ggml_backend_cpu_init();
    } 

    if (ggml_backend_is_cpu(model.backend)) {
        ggml_backend_cpu_set_n_threads(model.backend, n_threads);
    }

    // Read data from .gguf file: vesion, gguf magic number, tensor_count ... to gguf_ctx
    struct ggml_context *tmp_ctx = nullptr;
    struct gguf_init_params gguf_params = {
        /*no_alloc = */ false,
        /*.ctx     = */ &tmp_ctx,       
    };
    struct gguf_context * gguf_ctx = gguf_init_from_file(fname.c_str(), gguf_params);  
    if (!gguf_ctx)
    {
        fprintf(stderr, "%s: gguf_init_from_file() failed \n", __func__);
        return false;      
    }

    // Allocate `ggml_context` to store tensor data
    int num_tensors = gguf_get_n_tensors(gguf_ctx);    
    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead() * num_tensors, //multiplication, mem_size is a multiple of b
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    // initialize the pointer to point memory area allocate tensor (memory size, adress)
    model.ctx = ggml_init(params);
    // create tensors and save to main memory(RAM) zone of model.ctx
    for (int i = 0; i < num_tensors; i++) {   
        const char * name = gguf_get_tensor_name(gguf_ctx, i);  
        struct ggml_tensor * src = ggml_get_tensor(tmp_ctx, name); 
        if (i < 10) {
            printf("value of tensor src: %f\n", ggml_get_f32_1d(src, i));
        }     
        struct ggml_tensor * dst = ggml_dup_tensor(model.ctx, src);       
        ggml_set_name(dst, name);
    }
    model.buffer = ggml_backend_alloc_ctx_tensors(model.ctx, model.backend);
    // copy tensors from main memory to backend
    for (struct ggml_tensor * cur = ggml_get_first_tensor(model.ctx); cur != NULL; cur = ggml_get_next_tensor(model.ctx, cur)) {
        struct ggml_tensor * src = ggml_get_tensor(tmp_ctx, ggml_get_name(cur));
        size_t n_size = ggml_nbytes(src);
        ggml_backend_tensor_set(cur, ggml_get_data(src), 0, n_size);
    }
    gguf_free(gguf_ctx);

    // load tensor from ctx to vector conv2d_layers
    model.width  = 224;
    model.height = 224;
    model.conv2d_layers.resize(59);

    model.conv2d_layers[0].padding = 3;
    model.conv2d_layers[0].strike = 2;
    model.conv2d_layers[0].name_conv = "conv1_conv";
    model.conv2d_layers[0].name_bn = "conv1_bn";

    model.conv2d_layers[1].padding = 0;
    model.conv2d_layers[1].name_conv = "conv2_block1_1_conv";
    model.conv2d_layers[1].name_bn = "conv2_block1_1_bn";

    model.conv2d_layers[2].name_conv = "conv2_block1_2_conv";
    model.conv2d_layers[2].name_bn = "conv2_block1_2_bn";

    model.conv2d_layers[3].padding = 0;
    model.conv2d_layers[3].load_next = true;
    model.conv2d_layers[3].activate = false;
    model.conv2d_layers[3].name_conv = "conv2_block1_0_conv";
    model.conv2d_layers[3].name_bn = "conv2_block1_0_bn";

    model.conv2d_layers[4].padding = 0;
    model.conv2d_layers[4].skip_load = true;  
    model.conv2d_layers[4].activate = false;
    model.conv2d_layers[4].name_conv = "conv2_block1_3_conv";
    model.conv2d_layers[4].name_bn = "conv2_block1_3_bn";

    model.conv2d_layers[5].padding = 0;
    model.conv2d_layers[5].name_conv = "conv2_block2_1_conv";
    model.conv2d_layers[5].name_bn = "conv2_block2_1_bn";

    model.conv2d_layers[6].name_conv = "conv2_block2_2_conv";
    model.conv2d_layers[6].name_bn = "conv2_block2_2_bn";

    model.conv2d_layers[7].padding = 0;
    model.conv2d_layers[7].activate = false;
    model.conv2d_layers[7].name_conv = "conv2_block2_3_conv";
    model.conv2d_layers[7].name_bn = "conv2_block2_3_bn";

    model.conv2d_layers[8].padding = 0;
    model.conv2d_layers[8].name_conv = "conv2_block3_1_conv";
    model.conv2d_layers[8].name_bn = "conv2_block3_1_bn";

    model.conv2d_layers[9].name_conv = "conv2_block3_2_conv";
    model.conv2d_layers[9].name_bn = "conv2_block3_2_bn";

    model.conv2d_layers[10].padding = 0;
    model.conv2d_layers[10].activate = false;
    model.conv2d_layers[10].name_conv = "conv2_block3_3_conv";
    model.conv2d_layers[10].name_bn = "conv2_block3_3_bn";

    model.conv2d_layers[11].padding = 0;
    model.conv2d_layers[11].name_conv = "conv3_block1_1_conv";
    model.conv2d_layers[11].name_bn = "conv3_block1_1_bn";
    model.conv2d_layers[11].strike = 2;

    model.conv2d_layers[12].name_conv = "conv3_block1_2_conv";
    model.conv2d_layers[12].name_bn = "conv3_block1_2_bn";

    model.conv2d_layers[13].padding = 0;
    model.conv2d_layers[13].load_next = true;
    model.conv2d_layers[13].activate = false;
    model.conv2d_layers[13].name_conv = "conv3_block1_0_conv";
    model.conv2d_layers[13].name_bn = "conv3_block1_0_bn";
    model.conv2d_layers[13].strike = 2;

    model.conv2d_layers[14].padding = 0;
    model.conv2d_layers[14].skip_load = true; 
    model.conv2d_layers[14].activate = false;
    model.conv2d_layers[14].name_conv = "conv3_block1_3_conv";
    model.conv2d_layers[14].name_bn = "conv3_block1_3_bn";

    model.conv2d_layers[15].padding = 0;
    model.conv2d_layers[15].name_conv = "conv3_block2_1_conv";
    model.conv2d_layers[15].name_bn = "conv3_block2_1_bn";

    model.conv2d_layers[16].name_conv = "conv3_block2_2_conv";
    model.conv2d_layers[16].name_bn = "conv3_block2_2_bn";

    model.conv2d_layers[17].padding = 0;
    model.conv2d_layers[17].activate = false;
    model.conv2d_layers[17].name_conv = "conv3_block2_3_conv";
    model.conv2d_layers[17].name_bn = "conv3_block2_3_bn";

    model.conv2d_layers[18].padding = 0;
    model.conv2d_layers[18].name_conv = "conv3_block3_1_conv";
    model.conv2d_layers[18].name_bn = "conv3_block3_1_bn";

    model.conv2d_layers[19].name_conv = "conv3_block3_2_conv";
    model.conv2d_layers[19].name_bn = "conv3_block3_2_bn";

    model.conv2d_layers[20].padding = 0;
    model.conv2d_layers[20].activate = false;
    model.conv2d_layers[20].name_conv = "conv3_block3_3_conv";
    model.conv2d_layers[20].name_bn = "conv3_block3_3_bn";

    model.conv2d_layers[21].padding = 0;
    model.conv2d_layers[21].name_conv = "conv3_block4_1_conv";
    model.conv2d_layers[21].name_bn = "conv3_block4_1_bn";

    model.conv2d_layers[22].name_conv = "conv3_block4_2_conv";
    model.conv2d_layers[22].name_bn = "conv3_block4_2_bn";

    model.conv2d_layers[23].padding = 0;
    model.conv2d_layers[23].activate = false;
    model.conv2d_layers[23].name_conv = "conv3_block4_3_conv";
    model.conv2d_layers[23].name_bn = "conv3_block4_3_bn";

    model.conv2d_layers[24].padding = 0;
    model.conv2d_layers[24].name_conv = "conv4_block1_1_conv";
    model.conv2d_layers[24].name_bn = "conv4_block1_1_bn";
    model.conv2d_layers[24].strike = 2;

    model.conv2d_layers[25].name_conv = "conv4_block1_2_conv";
    model.conv2d_layers[25].name_bn = "conv4_block1_2_bn";

    model.conv2d_layers[26].padding = 0;
    model.conv2d_layers[26].load_next = true;
    model.conv2d_layers[26].activate = false;
    model.conv2d_layers[26].name_conv = "conv4_block1_0_conv";
    model.conv2d_layers[26].name_bn = "conv4_block1_0_bn";
    model.conv2d_layers[26].strike = 2;

    model.conv2d_layers[27].padding = 0;
    model.conv2d_layers[27].skip_load = true;
    model.conv2d_layers[27].activate = false;
    model.conv2d_layers[27].name_conv = "conv4_block1_3_conv";
    model.conv2d_layers[27].name_bn = "conv4_block1_3_bn";

    model.conv2d_layers[28].padding = 0;
    model.conv2d_layers[28].name_conv = "conv4_block2_1_conv";
    model.conv2d_layers[28].name_bn = "conv4_block2_1_bn";

    model.conv2d_layers[29].name_conv = "conv4_block2_2_conv";
    model.conv2d_layers[29].name_bn = "conv4_block2_2_bn";

    model.conv2d_layers[30].padding = 0;
    model.conv2d_layers[30].activate = false;
    model.conv2d_layers[30].name_conv = "conv4_block2_3_conv";
    model.conv2d_layers[30].name_bn = "conv4_block2_3_bn";

    model.conv2d_layers[31].padding = 0;
    model.conv2d_layers[31].name_conv = "conv4_block3_1_conv";
    model.conv2d_layers[31].name_bn = "conv4_block3_1_bn";

    model.conv2d_layers[32].name_conv = "conv4_block3_2_conv";
    model.conv2d_layers[32].name_bn = "conv4_block3_2_bn";

    model.conv2d_layers[33].padding = 0;
    model.conv2d_layers[33].activate = false;
    model.conv2d_layers[33].name_conv = "conv4_block3_3_conv";
    model.conv2d_layers[33].name_bn = "conv4_block3_3_bn";

    model.conv2d_layers[34].padding = 0;
    model.conv2d_layers[34].name_conv = "conv4_block4_1_conv";
    model.conv2d_layers[34].name_bn = "conv4_block4_1_bn";

    model.conv2d_layers[35].name_conv = "conv4_block4_2_conv";
    model.conv2d_layers[35].name_bn = "conv4_block4_2_bn";

    model.conv2d_layers[36].padding = 0;
    model.conv2d_layers[36].activate = false;
    model.conv2d_layers[36].name_conv = "conv4_block4_3_conv";
    model.conv2d_layers[36].name_bn = "conv4_block4_3_bn";

    model.conv2d_layers[37].padding = 0;
    model.conv2d_layers[37].name_conv = "conv4_block5_1_conv";
    model.conv2d_layers[37].name_bn = "conv4_block5_1_bn";

    model.conv2d_layers[38].name_conv = "conv4_block5_2_conv";
    model.conv2d_layers[38].name_bn = "conv4_block5_2_bn";

    model.conv2d_layers[39].padding = 0;
    model.conv2d_layers[39].activate = false;
    model.conv2d_layers[39].name_conv = "conv4_block5_3_conv";
    model.conv2d_layers[39].name_bn = "conv4_block5_3_bn";

    model.conv2d_layers[40].padding = 0;
    model.conv2d_layers[40].name_conv = "conv4_block6_1_conv";
    model.conv2d_layers[40].name_bn = "conv4_block6_1_bn";

    model.conv2d_layers[41].name_conv = "conv4_block6_2_conv";
    model.conv2d_layers[41].name_bn = "conv4_block6_2_bn";
    
    model.conv2d_layers[42].padding = 0;
    model.conv2d_layers[42].activate = false;
    model.conv2d_layers[42].name_conv = "conv4_block6_3_conv";
    model.conv2d_layers[42].name_bn = "conv4_block6_3_bn";

    model.conv2d_layers[43].padding = 0;
    model.conv2d_layers[43].name_conv = "conv5_block1_1_conv";
    model.conv2d_layers[43].name_bn = "conv5_block1_1_bn";
    model.conv2d_layers[43].strike = 2;

    model.conv2d_layers[44].name_conv = "conv5_block1_2_conv";
    model.conv2d_layers[44].name_bn = "conv5_block1_2_bn";

    model.conv2d_layers[45].padding = 0;
    model.conv2d_layers[45].load_next = true;
    model.conv2d_layers[45].activate = false;
    model.conv2d_layers[45].name_conv = "conv5_block1_0_conv";
    model.conv2d_layers[45].name_bn = "conv5_block1_0_bn";
    model.conv2d_layers[45].strike = 2;

    model.conv2d_layers[46].padding = 0;
    model.conv2d_layers[46].skip_load = true;
    model.conv2d_layers[46].activate = false;
    model.conv2d_layers[46].name_conv = "conv5_block1_3_conv";
    model.conv2d_layers[46].name_bn = "conv5_block1_3_bn";

    model.conv2d_layers[47].padding = 0;
    model.conv2d_layers[47].name_conv = "conv5_block2_1_conv";
    model.conv2d_layers[47].name_bn = "conv5_block2_1_bn";

    model.conv2d_layers[48].name_conv = "conv5_block2_2_conv";
    model.conv2d_layers[48].name_bn = "conv5_block2_2_bn";

    model.conv2d_layers[49].padding = 0;
    model.conv2d_layers[49].activate = false;
    model.conv2d_layers[49].name_conv = "conv5_block2_3_conv";
    model.conv2d_layers[49].name_bn = "conv5_block2_3_bn";

    model.conv2d_layers[50].padding = 0;
    model.conv2d_layers[50].name_conv = "conv5_block3_1_conv";
    model.conv2d_layers[50].name_bn = "conv5_block3_1_bn";

    model.conv2d_layers[51].name_conv = "conv5_block3_2_conv";
    model.conv2d_layers[51].name_bn = "conv5_block3_2_bn";

    model.conv2d_layers[52].padding = 0;
    model.conv2d_layers[52].activate = false;
    model.conv2d_layers[52].name_conv = "conv5_block3_3_conv";
    model.conv2d_layers[52].name_bn = "conv5_block3_3_bn";

    model.conv2d_layers[53].name_conv = "conv2d";
    model.conv2d_layers[53].name_bn = "batch_normalization";

    model.conv2d_layers[54].name_conv = "conv2d_1";
    model.conv2d_layers[54].name_bn = "batch_normalization_1";

    model.conv2d_layers[55].name_conv = "conv2d_2";
    model.conv2d_layers[55].name_bn = "batch_normalization_2";

    model.conv2d_layers[56].name_conv = "conv2d_3";
    model.conv2d_layers[56].name_bn = "batch_normalization_3";

    model.conv2d_layers[57].name_conv = "conv2d_4";
    model.conv2d_layers[57].name_bn = "batch_normalization_4";

    model.conv2d_layers[58].padding = 0;    
    model.conv2d_layers[58].batch_normalize = false;
    model.conv2d_layers[58].activate = false;
    model.conv2d_layers[58].name_conv = "conv2d_5"; 
  
    for (int i = 0; i < (int)model.conv2d_layers.size(); i++) {
        char name[256];
        if(model.conv2d_layers[i].skip_load) continue;

        else if(model.conv2d_layers[i].load_next) 
        {
            snprintf(name, sizeof(name), "%s/kernel:0", model.conv2d_layers[i].name_conv);
            model.conv2d_layers[i].weights = ggml_get_tensor(model.ctx, name);

            snprintf(name, sizeof(name), "%s/bias:0", model.conv2d_layers[i].name_conv);
            model.conv2d_layers[i].biases = ggml_get_tensor(model.ctx, name);       

            snprintf(name, sizeof(name), "%s/kernel:0", model.conv2d_layers[i+1].name_conv);
            model.conv2d_layers[i+1].weights = ggml_get_tensor(model.ctx, name);

            snprintf(name, sizeof(name), "%s/bias:0", model.conv2d_layers[i+1].name_conv);
            model.conv2d_layers[i+1].biases = ggml_get_tensor(model.ctx, name);

            if (model.conv2d_layers[i].batch_normalize) {
                snprintf(name, sizeof(name), "%s/gamma:0", model.conv2d_layers[i].name_bn);
                model.conv2d_layers[i].scales = ggml_get_tensor(model.ctx, name);

                snprintf(name, sizeof(name), "%s/beta:0", model.conv2d_layers[i].name_bn);
                model.conv2d_layers[i].beta = ggml_get_tensor(model.ctx, name);                     

                snprintf(name, sizeof(name), "%s/moving_mean:0", model.conv2d_layers[i].name_bn);
                model.conv2d_layers[i].rolling_mean = ggml_get_tensor(model.ctx, name);
             
                snprintf(name, sizeof(name), "%s/moving_variance:0", model.conv2d_layers[i].name_bn);
                model.conv2d_layers[i].rolling_variance = ggml_get_tensor(model.ctx, name);             
            }

            if (model.conv2d_layers[i+1].batch_normalize) {
                snprintf(name, sizeof(name), "%s/gamma:0", model.conv2d_layers[i+1].name_bn);
                model.conv2d_layers[i+1].scales = ggml_get_tensor(model.ctx, name);
          
                snprintf(name, sizeof(name), "%s/beta:0", model.conv2d_layers[i+1].name_bn);
                model.conv2d_layers[i+1].beta = ggml_get_tensor(model.ctx, name);
              
                snprintf(name, sizeof(name), "%s/moving_mean:0", model.conv2d_layers[i+1].name_bn);
                model.conv2d_layers[i+1].rolling_mean = ggml_get_tensor(model.ctx, name);
             
                snprintf(name, sizeof(name), "%s/moving_variance:0", model.conv2d_layers[i+1].name_bn);
                model.conv2d_layers[i+1].rolling_variance = ggml_get_tensor(model.ctx, name);              
            }
        }
        else
        {
            snprintf(name, sizeof(name), "%s/kernel:0", model.conv2d_layers[i].name_conv);
            model.conv2d_layers[i].weights = ggml_get_tensor(model.ctx, name);

            snprintf(name, sizeof(name), "%s/bias:0", model.conv2d_layers[i].name_conv);
            model.conv2d_layers[i].biases = ggml_get_tensor(model.ctx, name);       

            if (model.conv2d_layers[i].batch_normalize) {
                snprintf(name, sizeof(name), "%s/gamma:0", model.conv2d_layers[i].name_bn);
                model.conv2d_layers[i].scales = ggml_get_tensor(model.ctx, name);              

                snprintf(name, sizeof(name), "%s/beta:0", model.conv2d_layers[i].name_bn);
                model.conv2d_layers[i].beta = ggml_get_tensor(model.ctx, name);              

                snprintf(name, sizeof(name), "%s/moving_mean:0", model.conv2d_layers[i].name_bn);
                model.conv2d_layers[i].rolling_mean = ggml_get_tensor(model.ctx, name);          

                snprintf(name, sizeof(name), "%s/moving_variance:0", model.conv2d_layers[i].name_bn);
                model.conv2d_layers[i].rolling_variance = ggml_get_tensor(model.ctx, name);              
            }
        }      
        
    }     
    return true;
}

static void print_shape(int layer, const ggml_tensor * t)
{
    printf("Layer %2d output shape:  %3d x %3d x %4d x %3d\n", layer, (int)t->ne[0], (int)t->ne[1], (int)t->ne[2], (int)t->ne[3]);
}

static ggml_tensor * apply_conv2d_unet(ggml_context * ctx, ggml_tensor * input, const unet_conv2d_layer & layer)
{   
    struct ggml_tensor * result = ggml_conv_2d(ctx, layer.weights, input, layer.strike, layer.strike, layer.padding, layer.padding, 1, 1);
  
    result = ggml_add(ctx, result, ggml_repeat(ctx,layer.biases, result)); 
 
    if (layer.batch_normalize) {   
      
        result = ggml_sub(ctx, result, ggml_repeat(ctx,layer.rolling_mean, result));

        result = ggml_div(ctx, result, ggml_sqrt(ctx, ggml_repeat(ctx,layer.rolling_variance, result)));
   
        result = ggml_mul(ctx, result, ggml_repeat(ctx,layer.scales, result));

        result = ggml_add(ctx, result, ggml_repeat(ctx,layer.beta, result));
    }    
 
    if (layer.activate) {
        result = ggml_relu(ctx, result);
    }

    return result;
}

static struct ggml_cgraph * build_graph_unet(struct ggml_context * ctx_cgraph, const unet_model & model) {   
    struct ggml_cgraph * gf = ggml_new_graph(ctx_cgraph);   

    struct ggml_tensor * input = ggml_new_tensor_4d(ctx_cgraph, GGML_TYPE_F32, model.width, model.height, 3, 1); // 224x224x3x1
    print_shape(100, input);  
    ggml_set_name(input, "input");

    struct ggml_tensor * result = apply_conv2d_unet(ctx_cgraph, input, model.conv2d_layers[0]);  
    struct ggml_tensor * layer_0 = result; 
    print_shape(0, result);
    // result = ggml_pad(ctx_cgraph, result, 1, 1, 0, 0);    
    result = ggml_pool_2d(ctx_cgraph, result, GGML_OP_POOL_MAX, 3, 3, 2, 2, 1, 1);
    struct ggml_tensor * layer_3_connect = result;    
    result = apply_conv2d_unet(ctx_cgraph, result, model.conv2d_layers[1]);
    print_shape(1, result);
    result = apply_conv2d_unet(ctx_cgraph, result, model.conv2d_layers[2]);
    struct ggml_tensor * layer_4_connect = result;
    print_shape(2, result);
    result = apply_conv2d_unet(ctx_cgraph, layer_3_connect, model.conv2d_layers[3]);
    struct ggml_tensor * layer_3 = result;
    print_shape(3, result);
    result = apply_conv2d_unet(ctx_cgraph, layer_4_connect, model.conv2d_layers[4]);
    struct ggml_tensor * layer_4 = result;
    print_shape(4, result);

    result = ggml_add(ctx_cgraph, layer_3, layer_4);
    result = ggml_relu(ctx_cgraph, result); 
    struct ggml_tensor * layer_3_4 = result;   

    result = apply_conv2d_unet(ctx_cgraph, result, model.conv2d_layers[5]);
    print_shape(5, result);
    result = apply_conv2d_unet(ctx_cgraph, result, model.conv2d_layers[6]);
    print_shape(6, result);
    result = apply_conv2d_unet(ctx_cgraph, result, model.conv2d_layers[7]);
    print_shape(7, result);

    result = ggml_add(ctx_cgraph, layer_3_4, result);
    result = ggml_relu(ctx_cgraph, result); 
    struct ggml_tensor * layer_3_4_7 = result;

    result = apply_conv2d_unet(ctx_cgraph, result, model.conv2d_layers[8]);
    print_shape(8, result);
    result = apply_conv2d_unet(ctx_cgraph, result, model.conv2d_layers[9]);
    print_shape(9, result);
    result = apply_conv2d_unet(ctx_cgraph, result, model.conv2d_layers[10]);
    print_shape(10, result);

    result = ggml_add(ctx_cgraph, layer_3_4_7, result);
    result = ggml_relu(ctx_cgraph, result); 
    struct ggml_tensor * layer_3_4_7_10 = result;

    result = apply_conv2d_unet(ctx_cgraph, result, model.conv2d_layers[11]);
    print_shape(11, result);
    result = apply_conv2d_unet(ctx_cgraph, result, model.conv2d_layers[12]);
    struct ggml_tensor * layer_12 = result;
    print_shape(12, result);
    result = apply_conv2d_unet(ctx_cgraph, layer_3_4_7_10, model.conv2d_layers[13]);
    struct ggml_tensor * layer_13 = result;
    print_shape(13, result);
    result = apply_conv2d_unet(ctx_cgraph, layer_12, model.conv2d_layers[14]);
    print_shape(14, result);
    struct ggml_tensor * layer_14 = result;

    result = ggml_add(ctx_cgraph, layer_13, layer_14);
    result = ggml_relu(ctx_cgraph, result); 
    struct ggml_tensor * layer_13_14 = result;

    result = apply_conv2d_unet(ctx_cgraph, result, model.conv2d_layers[15]);
    print_shape(15, result);
    result = apply_conv2d_unet(ctx_cgraph, result, model.conv2d_layers[16]);
    print_shape(16, result);
    result = apply_conv2d_unet(ctx_cgraph, result, model.conv2d_layers[17]);
    print_shape(17, result);

    result = ggml_add(ctx_cgraph, layer_13_14, result);
    result = ggml_relu(ctx_cgraph, result); 
    struct ggml_tensor * layer_13_14_17 = result;

    result = apply_conv2d_unet(ctx_cgraph, result, model.conv2d_layers[18]);
    print_shape(18, result);
    result = apply_conv2d_unet(ctx_cgraph, result, model.conv2d_layers[19]);
    print_shape(19, result);
    result = apply_conv2d_unet(ctx_cgraph, result, model.conv2d_layers[20]);
    print_shape(20, result);

    result = ggml_add(ctx_cgraph, layer_13_14_17, result);
    result = ggml_relu(ctx_cgraph, result); 
    struct ggml_tensor * layer_13_14_17_20 = result;

    result = apply_conv2d_unet(ctx_cgraph, result, model.conv2d_layers[21]);
    print_shape(21, result);
    result = apply_conv2d_unet(ctx_cgraph, result, model.conv2d_layers[22]);
    print_shape(22, result);
    result = apply_conv2d_unet(ctx_cgraph, result, model.conv2d_layers[23]);
    print_shape(23, result);

    result = ggml_add(ctx_cgraph, layer_13_14_17_20, result);
    result = ggml_relu(ctx_cgraph, result); 
    struct ggml_tensor * layer_13_14_17_20_23 = result;

    result = apply_conv2d_unet(ctx_cgraph, result, model.conv2d_layers[24]);
    print_shape(24, result);
    result = apply_conv2d_unet(ctx_cgraph, result, model.conv2d_layers[25]);
    struct ggml_tensor * layer_25 = result;
    print_shape(25, result);
    result = apply_conv2d_unet(ctx_cgraph, layer_13_14_17_20_23, model.conv2d_layers[26]);
    print_shape(26, result);
    struct ggml_tensor * layer_26 = result;
    result = apply_conv2d_unet(ctx_cgraph, layer_25, model.conv2d_layers[27]);
    print_shape(27, result);
    struct ggml_tensor * layer_27 = result;

    result = ggml_add(ctx_cgraph, layer_26, layer_27);
    result = ggml_relu(ctx_cgraph, result); 
    struct ggml_tensor * layer_26_27 = result;

    result = apply_conv2d_unet(ctx_cgraph, result, model.conv2d_layers[28]);
    print_shape(28, result);
    result = apply_conv2d_unet(ctx_cgraph, result, model.conv2d_layers[29]);
    print_shape(29, result);
    result = apply_conv2d_unet(ctx_cgraph, result, model.conv2d_layers[30]);
    print_shape(30, result);

    result = ggml_add(ctx_cgraph, layer_26_27, result);
    result = ggml_relu(ctx_cgraph, result); 
    struct ggml_tensor * layer_26_27_30 = result;

    result = apply_conv2d_unet(ctx_cgraph, result, model.conv2d_layers[31]);
    print_shape(31, result);
    result = apply_conv2d_unet(ctx_cgraph, result, model.conv2d_layers[32]);
    print_shape(32, result);
    result = apply_conv2d_unet(ctx_cgraph, result, model.conv2d_layers[33]);
    print_shape(33, result);

    result = ggml_add(ctx_cgraph, layer_26_27_30, result);
    result = ggml_relu(ctx_cgraph, result); 
    struct ggml_tensor * layer_26_27_30_33 = result;

    result = apply_conv2d_unet(ctx_cgraph, result, model.conv2d_layers[34]);
    print_shape(34, result);
    result = apply_conv2d_unet(ctx_cgraph, result, model.conv2d_layers[35]);
    print_shape(35, result);
    result = apply_conv2d_unet(ctx_cgraph, result, model.conv2d_layers[36]);
    print_shape(36, result);

    result = ggml_add(ctx_cgraph, layer_26_27_30_33, result);
    result = ggml_relu(ctx_cgraph, result); 
    struct ggml_tensor * layer_26_27_30_33_36 = result;

    result = apply_conv2d_unet(ctx_cgraph, result, model.conv2d_layers[37]);
    print_shape(37, result);
    result = apply_conv2d_unet(ctx_cgraph, result, model.conv2d_layers[38]);
    print_shape(38, result);
    result = apply_conv2d_unet(ctx_cgraph, result, model.conv2d_layers[39]);
    print_shape(39, result);

    result = ggml_add(ctx_cgraph, layer_26_27_30_33_36, result);
    result = ggml_relu(ctx_cgraph, result); 
    struct ggml_tensor * layer_26_27_30_33_36_39 = result;

    result = apply_conv2d_unet(ctx_cgraph, result, model.conv2d_layers[40]);
    print_shape(40, result);
    result = apply_conv2d_unet(ctx_cgraph, result, model.conv2d_layers[41]);
    print_shape(41, result);
    result = apply_conv2d_unet(ctx_cgraph, result, model.conv2d_layers[42]);
    print_shape(42, result);

    result = ggml_add(ctx_cgraph, layer_26_27_30_33_36_39, result);
    result = ggml_relu(ctx_cgraph, result); 
    struct ggml_tensor * layer_26_27_30_33_36_39_42 = result;

    result = apply_conv2d_unet(ctx_cgraph, result, model.conv2d_layers[43]);
    print_shape(43, result);
    result = apply_conv2d_unet(ctx_cgraph, result, model.conv2d_layers[44]);
    struct ggml_tensor * layer_44 = result;
    print_shape(44, result);
    result = apply_conv2d_unet(ctx_cgraph, layer_26_27_30_33_36_39_42, model.conv2d_layers[45]);
    struct ggml_tensor * layer_45 = result;
    print_shape(45, result);
    result = apply_conv2d_unet(ctx_cgraph, layer_44, model.conv2d_layers[46]);
    struct ggml_tensor * layer_46 = result;
    print_shape(46, result);

    result = ggml_add(ctx_cgraph, layer_45, layer_46);
    result = ggml_relu(ctx_cgraph, result); 
    struct ggml_tensor * layer_45_46 = result;

    result = apply_conv2d_unet(ctx_cgraph, result, model.conv2d_layers[47]);
    print_shape(47, result);
    result = apply_conv2d_unet(ctx_cgraph, result, model.conv2d_layers[48]);
    print_shape(48, result);
    result = apply_conv2d_unet(ctx_cgraph, result, model.conv2d_layers[49]);
    print_shape(49, result);

    result = ggml_add(ctx_cgraph, layer_45_46, result);
    result = ggml_relu(ctx_cgraph, result); 
    struct ggml_tensor * layer_45_46_49 = result;

    result = apply_conv2d_unet(ctx_cgraph, result, model.conv2d_layers[50]);
    print_shape(50, result);
    result = apply_conv2d_unet(ctx_cgraph, result, model.conv2d_layers[51]);
    print_shape(51, result);
    result = apply_conv2d_unet(ctx_cgraph, result, model.conv2d_layers[52]);
    print_shape(52, result);

    result = ggml_add(ctx_cgraph, layer_45_46_49, result);
    result = ggml_relu(ctx_cgraph, result); 
    struct ggml_tensor * layer_45_46_49_52 = result;

    result = ggml_upscale(ctx_cgraph, result, 2);
                                                                        
    result = ggml_concat(ctx_cgraph, result, layer_26_27_30_33_36_39_42, 2);

    result = apply_conv2d_unet(ctx_cgraph, result, model.conv2d_layers[53]);
    print_shape(53, result);

    result = ggml_upscale(ctx_cgraph, result, 2);
   
    result = ggml_concat(ctx_cgraph, result, layer_13_14_17_20_23, 2);

    result = apply_conv2d_unet(ctx_cgraph, result, model.conv2d_layers[54]);
    print_shape(54, result);

    result = ggml_upscale(ctx_cgraph, result, 2);
    
    result = ggml_concat(ctx_cgraph, result, layer_3_4_7_10, 2);

    result = apply_conv2d_unet(ctx_cgraph, result, model.conv2d_layers[55]);
    print_shape(55, result);

    result = ggml_upscale(ctx_cgraph, result, 2);
    
    result = ggml_concat(ctx_cgraph, result, layer_0, 2);

    result = apply_conv2d_unet(ctx_cgraph, result, model.conv2d_layers[56]);
    print_shape(56, result);

    result = ggml_upscale(ctx_cgraph, result, 2);

    result = apply_conv2d_unet(ctx_cgraph, result, model.conv2d_layers[57]);
    print_shape(57, result);
    result = apply_conv2d_unet(ctx_cgraph, result, model.conv2d_layers[58]);
    result = ggml_sigmoid(ctx_cgraph, result);
    print_shape(58, result);
    struct ggml_tensor * layer_58 = result;

    ggml_set_output(layer_58);
    ggml_set_name(layer_58, "layer_58");
    print_shape(59, result);

    ggml_build_forward_expand(gf, layer_58);    
    return gf;
}

struct unet_layer {   
    std::vector<float> predictions;
    int w;
    int h;

    unet_layer(struct ggml_tensor * prev_layer)       
    {
        w = prev_layer->ne[0];
        h = prev_layer->ne[1];
        predictions.resize(ggml_nbytes(prev_layer)/sizeof(float));
        ggml_backend_tensor_get(prev_layer, predictions.data(), 0, ggml_nbytes(prev_layer));
    }
};

void detect_defect(unet_image & img, unet_image & dst, struct ggml_cgraph * gf, const unet_model & model, float thresh)
{   
    unet_image sized = letterbox_image_unet(img, model.width, model.height);
    struct ggml_tensor * input = ggml_graph_get_tensor(gf, "input");
    ggml_backend_tensor_set(input, sized.data.data(), 0, ggml_nbytes(input));

    if (ggml_backend_graph_compute(model.backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "%s: ggml_backend_graph_compute() failed\n", __func__);
        return;
    }

    struct ggml_tensor * layer_58 = ggml_graph_get_tensor(gf, "layer_58");
    unet_layer unet58{layer_58};   

    dst.w = model.width;
    dst.h = model.height;
    dst.c = 1;
    dst.data.resize(dst.w*dst.h*dst.c);

    if (unet58.predictions.size() != dst.w * dst.h * dst.c) {
        fprintf(stderr, "%s: Size of predictions does not match image dimensions.\n", __func__);
        return;
    }

    for (int k = 0; k < dst.c; ++k){
        for (int j = 0; j < dst.h; ++j){
            for (int i = 0; i < dst.w; ++i){                
                int index = i + dst.w*j + dst.w*dst.h*k;
                if (unet58.predictions[index] < thresh) {
                    dst.data[index] = 0.;                 
                }
                else {
                    dst.data[index] = 255.;
                }                
            }
        }
    }
}


int main(int argc, char ** argv) 
{
    ggml_time_init();
    unet_model model;

    unet_params params;
    if (!unet_params_parse(argc, argv, params)) 
    {
        return 1;
    }
  
    if (!load_model(params.model, model, params.threads)) 
    {
        fprintf(stderr, "%s: failed to load model from '%s'\n", __func__, params.model.c_str());
        return 1;
    }  
    // create a temporally context to build the graph
    struct ggml_init_params params0 = {
        /*.mem_size   =*/ ggml_tensor_overhead()*GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true, // the tensors will be allocated later by ggml_gallocr_alloc_graph()
    };
    struct ggml_context * ctx_cgraph = ggml_init(params0); // pointer to save adress of tensor
   
    struct ggml_cgraph * gf = build_graph_unet(ctx_cgraph, model);

    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
    ggml_gallocr_alloc_graph(allocr, gf);

    const int64_t t_start_ms = ggml_time_ms();
   
    for (size_t idx = 0; idx < params.fname_inp.size(); ++idx) {
        const std::string &input_file = params.fname_inp[idx];
        std::string output_file;
    
        if (idx < params.fname_out.size()) {
            output_file = params.fname_out[idx];
        } else {
            
            output_file = "defect prediction" + std::to_string(idx + 1) + ".jpg";
        }
      
        unet_image img(0, 0, 0);
        if (!load_unet_image(input_file.c_str(), img)) {
            fprintf(stderr, "%s: failed to load image from '%s'\n", __func__, input_file.c_str());
            return 1;
        }
       
        unet_image img_result(0, 0, 0);
        detect_defect(img, img_result, gf, model, params.thresh);
    
        if (!save_unet_image(img_result, output_file.c_str(), 80)) {
            fprintf(stderr, "%s: failed to save image to '%s'\n", __func__, output_file.c_str());
            return 1;
        }

        printf("Processed: %s -> %s\n", input_file.c_str(), output_file.c_str());
    }

    const int64_t t_detect_ms = ggml_time_ms() - t_start_ms;  
    printf("Detected objects saved in (time: %f sec.)\n",  t_detect_ms / 1000.0f);

    ggml_free(ctx_cgraph);
    ggml_gallocr_free(allocr);
    ggml_free(model.ctx);
    ggml_backend_buffer_free(model.buffer);
    ggml_backend_free(model.backend);
    return 0;
}



