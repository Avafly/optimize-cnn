#include "common.h"
#include "layers.h"

#include "3rdparty/fmt/base.h"

#include <omp.h>

#include <cstddef>

constexpr int MODEL_SIZE = 7018;
constexpr int BLOB_SIZE = 10730;
constexpr int IM2COL_BUF_SIZE = 10368;
constexpr int NUM_LAYER = 7;

static float Inputs[optcnn::NUM_IMAGES * optcnn::IMG_SIZE]
    __attribute__((aligned(optcnn::ALIGN)));
static float ModelParam[MODEL_SIZE]
    __attribute__((aligned(optcnn::ALIGN)));
static float Blobs[optcnn::MAX_THREADS * BLOB_SIZE]
    __attribute__((aligned(optcnn::ALIGN)));
static float Im2colBuf[optcnn::MAX_THREADS * IM2COL_BUF_SIZE]
    __attribute__((aligned(optcnn::ALIGN)));
static float Logits[optcnn::NUM_IMAGES * optcnn::NUM_CLASSES]
    __attribute__((aligned(optcnn::ALIGN)));
static float RefLogits[optcnn::NUM_IMAGES * optcnn::NUM_CLASSES]
    __attribute__((aligned(optcnn::ALIGN)));

static optcnn::Layer Layers[NUM_LAYER];

static void build_model()
{
    // this is a simple demo where layers are defined directly in the code
    // the model could be defined by a config file in a real use case
    Layers[0] = {optcnn::LAYER_CONV, ModelParam, ModelParam + 72, 3, 8, 1, 0.0f, 0, 0};
    Layers[1] = {optcnn::LAYER_RELU, nullptr, nullptr, 0, 0, 0, 0.1f, 0, 0};
    Layers[2] = {optcnn::LAYER_MAXPOOL, nullptr, nullptr, 2, 0, 0, 0.0f, 0, 0};
    Layers[3] = {optcnn::LAYER_CONV, ModelParam + 80, ModelParam + 1232, 3, 16, 0, 0.0f, 0, 0};
    Layers[4] = {optcnn::LAYER_RELU, nullptr, nullptr, 0, 0, 0, 0.1f, 0, 0};
    Layers[5] = {optcnn::LAYER_MAXPOOL, nullptr, nullptr, 2, 0, 0, 0.0f, 0, 0};
    Layers[6] = {optcnn::LAYER_FC, ModelParam + 1248, ModelParam + 7008, 0, 0, 0, 0.0f, 576, 10};
}

static void run(float *in, const std::size_t index, float *blob, float *data_col)
{
    float *bottom = in, *top = blob;
    int in_c = optcnn::IMG_C, in_h = optcnn::IMG_H, in_w = optcnn::IMG_W;
    int out_c = 0, out_h = 0, out_w = 0, out_size = 0;
    bool first = true;

    for (int i = 0; i < NUM_LAYER; ++i)
    {
        const optcnn::Layer &layer = Layers[i];
        switch (layer.type)
        {
            case optcnn::LAYER_CONV:
                if (!first)
                {
                    bottom = top;
                    top += out_size;
                }
                first = false;
                out_c = layer.filters;
                out_h = in_h - layer.kernel_size + 2 * layer.padding + 1;
                out_w = in_w - layer.kernel_size + 2 * layer.padding + 1;
                optcnn::conv_layer(bottom, top, data_col, in_c, in_h, in_w, out_c, out_h, out_w,
                                   layer.weight, layer.bias, layer.kernel_size, layer.padding);
                in_c = out_c, in_h = out_h, in_w = out_w;
                out_size = out_c * out_h * out_w;
                break;
            case optcnn::LAYER_RELU:
                optcnn::relu(top, top, out_size, layer.alpha);
                break;
            case optcnn::LAYER_MAXPOOL:
                bottom = top;
                top += out_size;
                out_h = in_h / layer.kernel_size;
                out_w = in_w / layer.kernel_size;
                optcnn::max_pool(bottom, top, in_c, in_h, in_w, out_h, out_w);
                in_h = out_h, in_w = out_w;
                out_size = in_c * out_h * out_w;
                break;
            case optcnn::LAYER_FC:
                bottom = top;
                top += out_size;
                optcnn::fc_layer(bottom, top, layer.weight, layer.bias, layer.out_feat,
                                 layer.in_feat);
                out_size = layer.out_feat;
                break;
        }
    }

    float *logits = Logits + index * optcnn::NUM_CLASSES;
    for (int i = 0; i < optcnn::NUM_CLASSES; ++i)
        logits[i] = top[i];
}

int main(int argc, char *argv[])
{
    // get args
    optcnn::Args args = optcnn::parse_args(argc, argv);

    fmt::print("Data dir: {}\n", args.data_dir);
    fmt::print("Threads:  {}\n", args.threads);
    fmt::print("Runs:     {}\n", args.runs);

    // load data
    optcnn::load_data(args.images_path(), Inputs, sizeof(Inputs));
    optcnn::load_data(args.ref_path(), RefLogits, sizeof(RefLogits));
    // load model
    optcnn::load_data(args.path("opt/model.bin"), ModelParam, MODEL_SIZE * sizeof(float));

    build_model();

    // run inference
    omp_set_num_threads(1);
    openblas_set_num_threads(1);
    const int threads = args.threads < optcnn::MAX_THREADS ? args.threads : optcnn::MAX_THREADS;
    const optcnn::Timing t = optcnn::run_timed(args.runs, [&] {
        for (int base = 0; base < optcnn::NUM_IMAGES; base += optcnn::BATCH)
        {
            const int count = optcnn::NUM_IMAGES - base < optcnn::BATCH ? optcnn::NUM_IMAGES - base
                                                                        : optcnn::BATCH;
            #pragma omp parallel for num_threads(threads) schedule(static)
            for (int j = 0; j < count; ++j)
            {
                const int tid = omp_get_thread_num();
                const std::size_t i = static_cast<std::size_t>(base) + j;
                run(Inputs + i * optcnn::IMG_SIZE, i, Blobs + tid * BLOB_SIZE,
                    Im2colBuf + tid * IM2COL_BUF_SIZE);
            }
        }
    });

    fmt::print("inference: {} imgs over {} runs: best {:.1f}  mean {:.1f}  worst {:.1f} ms  "
               "(spread {:.1f}%, {:.1f} img/ms)\n",
               optcnn::NUM_IMAGES, args.runs, t.best, t.mean, t.worst,
               100.0 * (t.worst - t.best) / t.best, optcnn::NUM_IMAGES / t.best);

    optcnn::verify(Logits, RefLogits, optcnn::NUM_IMAGES, 1e-4f);

    return 0;
}
