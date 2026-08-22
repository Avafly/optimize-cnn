#ifndef LAYERS_H_
#define LAYERS_H_

#include <cblas.h>

namespace optcnn
{

enum LayerType
{
    LAYER_CONV,
    LAYER_RELU,
    LAYER_MAXPOOL,
    LAYER_FC
};

struct Layer
{
    LayerType type;
    const float *weight;
    const float *bias;
    // conv
    int kernel_size;
    int filters;
    int padding;
    // relu
    float alpha;
    // fc
    int in_feat;
    int out_feat;
};

inline void im2col(const float *data_im, float *data_col, const int in_c, const int in_h,
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

inline void conv_layer(const float *bottom, float *top, float *data_col, const int in_c,
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

inline void relu(const float *bottom, float *top, const int count, const float alpha)
{
    for (int i = 0; i < count; ++i)
        top[i] = bottom[i] > 0.0f ? bottom[i] : bottom[i] * alpha;
}

inline void max_pool(const float *bottom, float *top, const int channels, const int in_h,
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

inline void fc_layer(const float *bottom, float *top, const float *weight, const float *bias,
                     const int out_feat, const int in_feat)
{
    for (int i = 0; i < out_feat; ++i)
        top[i] = bias[i];
    cblas_sgemv(CblasRowMajor, CblasNoTrans, out_feat, in_feat, 1.0f, weight, in_feat, bottom, 1,
                1.0f, top, 1);
}

} // namespace optcnn

#endif // LAYERS_H_
