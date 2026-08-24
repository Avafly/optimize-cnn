#include "common.h"

#include "3rdparty/fmt/base.h"

#include <arm_neon.h>
#include <omp.h>

#include <cstddef>
#include <cstring>

constexpr int MODEL_SIZE = 7018;
constexpr int BLOB_SIZE = 10730;
constexpr int PAD_W = 30;
constexpr int PAD_SIZE = PAD_W * PAD_W;

static float Inputs[optcnn::NUM_IMAGES * optcnn::IMG_SIZE]
    __attribute__((aligned(optcnn::ALIGN)));
static float ModelParam[MODEL_SIZE]
    __attribute__((aligned(optcnn::ALIGN)));
static float Blobs[optcnn::MAX_THREADS * BLOB_SIZE]
    __attribute__((aligned(optcnn::ALIGN)));
static float Pad[optcnn::MAX_THREADS * PAD_SIZE]
    __attribute__((aligned(optcnn::ALIGN)));
static float Logits[optcnn::NUM_IMAGES * optcnn::NUM_CLASSES]
    __attribute__((aligned(optcnn::ALIGN)));
static float RefLogits[optcnn::NUM_IMAGES * optcnn::NUM_CLASSES]
    __attribute__((aligned(optcnn::ALIGN)));

static float Conv1WP[9 * 8]
    __attribute__((aligned(optcnn::ALIGN)));
static float Conv2WP[72 * 16]
    __attribute__((aligned(optcnn::ALIGN)));
static float FcWP[576 * 10]
    __attribute__((aligned(optcnn::ALIGN)));

static void pack_weights(const float *conv1w, const float *conv2w, const float *fcw)
{
    for (int oc = 0; oc < 8; ++oc)
        for (int t = 0; t < 9; ++t)
            Conv1WP[t * 8 + oc] = conv1w[oc * 9 + t];
    for (int oc = 0; oc < 16; ++oc)
        for (int t = 0; t < 72; ++t)
            Conv2WP[t * 16 + oc] = conv2w[oc * 72 + t];
    for (int kb = 0; kb < 144; ++kb)
        for (int oc = 0; oc < 10; ++oc)
            for (int l = 0; l < 4; ++l)
                FcWP[(kb * 10 + oc) * 4 + l] = fcw[oc * 576 + kb * 4 + l];
}

// copy data into the middle of a zero-bordered buffer to skip boundary check
static void pack_input(const float *in, float *pad)
{
    for (int h = 0; h < optcnn::IMG_H; ++h)
        std::memcpy(pad + (h + 1) * PAD_W + 1, in + h * optcnn::IMG_W,
                    optcnn::IMG_W * sizeof(float));
}

static void conv_layer(const float *bottom, float *top, const float *packed_weight,
                       const float *bias, const int in_c, const int in_h, const int in_w,
                       const int out_c, const int out_h, const int out_w, const int ks)
{
    const int n = out_h * out_w;
    for (int og = 0; og < out_c; og += 4)
    {
        for (int oh = 0; oh < out_h; ++oh)
        {
            for (int ob = 0; ob < out_w; ob += 4)
            {
                float32x4_t a0 = vdupq_n_f32(bias[og + 0]), a1 = vdupq_n_f32(bias[og + 1]),
                            a2 = vdupq_n_f32(bias[og + 2]), a3 = vdupq_n_f32(bias[og + 3]);
                for (int ic = 0; ic < in_c; ++ic)
                    for (int kh = 0; kh < ks; ++kh)
                        for (int kw = 0; kw < ks; ++kw)
                        {
                            const int t = (ic * ks + kh) * ks + kw;
                            const float32x4_t iv =
                                vld1q_f32(bottom + (ic * in_h + oh + kh) * in_w + ob + kw);
                            const float32x4_t w = vld1q_f32(packed_weight + t * out_c + og);
                            a0 = vfmaq_laneq_f32(a0, iv, w, 0);
                            a1 = vfmaq_laneq_f32(a1, iv, w, 1);
                            a2 = vfmaq_laneq_f32(a2, iv, w, 2);
                            a3 = vfmaq_laneq_f32(a3, iv, w, 3);
                        }
                float *d = top + (og * out_h + oh) * out_w + ob;
                vst1q_f32(d + 0 * n, a0);
                vst1q_f32(d + 1 * n, a1);
                vst1q_f32(d + 2 * n, a2);
                vst1q_f32(d + 3 * n, a3);
            }
        }
    }
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

static void fc_layer(const float *bottom, float *top, const float *weight, const float *bias)
{
    float32x4_t a0 = vdupq_n_f32(0.0f), a1 = vdupq_n_f32(0.0f), a2 = vdupq_n_f32(0.0f),
                a3 = vdupq_n_f32(0.0f), a4 = vdupq_n_f32(0.0f), a5 = vdupq_n_f32(0.0f),
                a6 = vdupq_n_f32(0.0f), a7 = vdupq_n_f32(0.0f), a8 = vdupq_n_f32(0.0f),
                a9 = vdupq_n_f32(0.0f);
    for (int k = 0; k < 576; k += 4)
    {
        const float32x4_t iv = vld1q_f32(bottom + k);
        const float *w = weight + k * 10;
        a0 = vfmaq_f32(a0, iv, vld1q_f32(w + 0));
        a1 = vfmaq_f32(a1, iv, vld1q_f32(w + 4));
        a2 = vfmaq_f32(a2, iv, vld1q_f32(w + 8));
        a3 = vfmaq_f32(a3, iv, vld1q_f32(w + 12));
        a4 = vfmaq_f32(a4, iv, vld1q_f32(w + 16));
        a5 = vfmaq_f32(a5, iv, vld1q_f32(w + 20));
        a6 = vfmaq_f32(a6, iv, vld1q_f32(w + 24));
        a7 = vfmaq_f32(a7, iv, vld1q_f32(w + 28));
        a8 = vfmaq_f32(a8, iv, vld1q_f32(w + 32));
        a9 = vfmaq_f32(a9, iv, vld1q_f32(w + 36));
    }
    top[0] = bias[0] + vaddvq_f32(a0);
    top[1] = bias[1] + vaddvq_f32(a1);
    top[2] = bias[2] + vaddvq_f32(a2);
    top[3] = bias[3] + vaddvq_f32(a3);
    top[4] = bias[4] + vaddvq_f32(a4);
    top[5] = bias[5] + vaddvq_f32(a5);
    top[6] = bias[6] + vaddvq_f32(a6);
    top[7] = bias[7] + vaddvq_f32(a7);
    top[8] = bias[8] + vaddvq_f32(a8);
    top[9] = bias[9] + vaddvq_f32(a9);
}

static void run(float *in, const std::size_t index, float *blob, float *pad,
                const float *model_param)
{
    const float *conv1b = model_param + 72;
    const float *conv2b = model_param + 1232;
    const float *fcb = model_param + 7008;

    float *conv1_out = blob;
    float *pool1_out = conv1_out + 8 * 784;
    float *conv2_out = pool1_out + 8 * 196;
    float *pool2_out = conv2_out + 16 * 144;
    float *fc_out = pool2_out + 16 * 36;

    pack_input(in, pad);
    conv_layer(pad, conv1_out, Conv1WP, conv1b, 1, PAD_W, PAD_W, 8, 28, 28, 3);
    relu(conv1_out, conv1_out, 8 * 28 * 28, 0.1f);
    max_pool(conv1_out, pool1_out, 8, 28, 28, 14, 14);

    conv_layer(pool1_out, conv2_out, Conv2WP, conv2b, 8, 14, 14, 16, 12, 12, 3);
    relu(conv2_out, conv2_out, 16 * 12 * 12, 0.1f);
    max_pool(conv2_out, pool2_out, 16, 12, 12, 6, 6);

    fc_layer(pool2_out, fc_out, FcWP, fcb);

    float *logits = Logits + index * optcnn::NUM_CLASSES;
    for (int i = 0; i < optcnn::NUM_CLASSES; ++i)
        logits[i] = fc_out[i];
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
    pack_weights(ModelParam + 0, ModelParam + 80, ModelParam + 1248);
    std::memset(Pad, 0, sizeof(Pad));

    // run inference
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
                run(Inputs + i * optcnn::IMG_SIZE, i, Blobs + tid * BLOB_SIZE, Pad + tid * PAD_SIZE,
                    ModelParam);
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
