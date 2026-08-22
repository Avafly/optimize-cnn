#include "common.h"

#include "3rdparty/fmt/base.h"

#include <cblas.h>
#include <omp.h>

#include <cstddef>

constexpr int MODEL_SIZE = 7018;
constexpr int BLOB_SIZE = 10730;
constexpr int IM2COL_BUF_SIZE = 10368;

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

static void im2col(const float *data_im, float *data_col, const int in_c, const int in_h,
                   const int in_w, const int out_h, const int out_w, const int kernel_size,
                   const int padding)
{
    const int kk = kernel_size * kernel_size;
    const int n = out_h * out_w;
    for (int c = 0; c < in_c; ++c)
    {
        for (int kh = 0; kh < kernel_size; ++kh)
        {
            for (int kw = 0; kw < kernel_size; ++kw)
            {
                float *row = data_col + (c * kk + kh * kernel_size + kw) * n;
                for (int oh = 0; oh < out_h; ++oh)
                {
                    const int ih = oh + kh - padding;
                    for (int ow = 0; ow < out_w; ++ow)
                    {
                        const int iw = ow + kw - padding;
                        *row++ = (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w)
                                     ? data_im[(c * in_h + ih) * in_w + iw]
                                     : 0.0f;
                    }
                }
            }
        }
    }
}

static void conv_layer(const float *bottom, float *top, float *data_col, const int in_c,
                       const int in_h, const int in_w, const int out_c, const int out_h,
                       const int out_w, const float *weight, const float *bias,
                       const int kernel_size, const int padding)
{
    im2col(bottom, data_col, in_c, in_h, in_w, out_h, out_w, kernel_size, padding);

    const int m = out_c;
    const int n = out_h * out_w;
    const int k = in_c * kernel_size * kernel_size;

    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
            top[i * n + j] = bias[i];

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0f, weight, k, data_col, n,
                1.0f, top, n);
}

static void relu(const float *bottom, float *top, const int count, const float alpha)
{
    for (int i = 0; i < count; ++i)
        top[i] = bottom[i] > 0.0f ? bottom[i] : bottom[i] * alpha;
}

static void max_pool(const float *bottom, float *top, const int channels, const int in_h,
                     const int in_w, const int out_h, const int out_w)
{
    for (int c = 0; c < channels; ++c)
    {
        for (int oh = 0; oh < out_h; ++oh)
        {
            for (int ow = 0; ow < out_w; ++ow)
            {
                const float *p = bottom + (c * in_h + oh * 2) * in_w + ow * 2;
                float v = p[0];
                v = p[1] > v ? p[1] : v;
                v = p[in_w] > v ? p[in_w] : v;
                v = p[in_w + 1] > v ? p[in_w + 1] : v;
                top[(c * out_h + oh) * out_w + ow] = v;
            }
        }
    }
}

static void fc_layer(const float *bottom, float *top, const float *weight, const float *bias,
                     const int out_feat, const int in_feat)
{
    for (int i = 0; i < out_feat; ++i)
        top[i] = bias[i];
    cblas_sgemv(CblasRowMajor, CblasNoTrans, out_feat, in_feat, 1.0f, weight, in_feat, bottom, 1,
                1.0f, top, 1);
}

static void run(float *in, const std::size_t index, float *blob, float *data_col,
                const float *model_param)
{
    float *bottom = in, *top = blob;
    int kernel_size, padding;
    int in_c, in_h, in_w, out_c, out_h, out_w, in_feat, out_feat;

    // parse model
    const float *conv1w = model_param;
    const float *conv1b = model_param + 72;
    const float *conv2w = model_param + 80;
    const float *conv2b = model_param + 1232;
    const float *fcw = model_param + 1248;
    const float *fcb = model_param + 7008;

    // conv1
    kernel_size = 3, padding = 1;
    in_c = 1, in_h = optcnn::IMG_H, in_w = optcnn::IMG_W, out_c = 8;
    out_h = in_h - kernel_size + 2 * padding + 1;
    out_w = in_w - kernel_size + 2 * padding + 1;
    conv_layer(bottom, top, data_col, in_c, in_h, in_w, out_c, out_h, out_w, conv1w, conv1b,
               kernel_size, padding);

    // relu
    relu(top, top, out_c * out_h * out_w, 0.1f);

    // max pooling
    bottom = top;
    kernel_size = 2;
    top += out_c * out_h * out_w;
    in_c = out_c, in_h = out_h, in_w = out_w, in_c = out_c;
    out_h = in_h / kernel_size;
    out_w = in_w / kernel_size;
    max_pool(bottom, top, in_c, in_h, in_w, out_h, out_w);

    // conv2
    bottom = top;
    top += out_c * out_h * out_w;
    kernel_size = 3, padding = 0;
    in_c = out_c, in_h = out_h, in_w = out_w, out_c = 16;
    out_h = in_h - kernel_size + 1;
    out_w = in_w - kernel_size + 1;
    conv_layer(bottom, top, data_col, in_c, in_h, in_w, out_c, out_h, out_w, conv2w, conv2b,
               kernel_size, padding);

    // relu
    relu(top, top, out_c * out_h * out_w, 0.1f);

    // max pooling
    bottom = top;
    kernel_size = 2;
    top += out_c * out_h * out_w;
    in_c = out_c, in_h = out_h, in_w = out_w, in_c = out_c;
    out_h = in_h / kernel_size;
    out_w = in_w / kernel_size;
    max_pool(bottom, top, in_c, in_h, in_w, out_h, out_w);

    // fc
    bottom = top;
    top += out_c * out_h * out_w;
    in_feat = out_c * out_h * out_w, out_feat = optcnn::NUM_CLASSES;
    fc_layer(bottom, top, fcw, fcb, out_feat, in_feat);

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
                    Im2colBuf + tid * IM2COL_BUF_SIZE, ModelParam);
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
