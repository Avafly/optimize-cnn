#include "common.h"

#include "3rdparty/fmt/base.h"

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>

static float Inputs[optcnn::NUM_IMAGES * optcnn::IMG_SIZE]
    __attribute__((aligned(optcnn::ALIGN)));
static float Logits[optcnn::NUM_IMAGES * optcnn::NUM_CLASSES]
    __attribute__((aligned(optcnn::ALIGN)));
static float RefLogits[optcnn::NUM_IMAGES * optcnn::NUM_CLASSES]
    __attribute__((aligned(optcnn::ALIGN)));

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

    // init onnxruntime
    Ort::Env env(ORT_LOGGING_LEVEL_ERROR, "cnn_ort");
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(args.threads);
    opts.SetGraphOptimizationLevel(ORT_ENABLE_ALL);

    Ort::Session session{nullptr};
    try
    {
        session = Ort::Session(env, args.path("onnxruntime/model.onnx").c_str(), opts);
    }
    catch (const Ort::Exception &e)
    {
        fmt::print(stderr, "error: cannot load model: {}\n", e.what());
        return 1;
    }

    Ort::AllocatorWithDefaultOptions alloc;
    const Ort::AllocatedStringPtr in_name = session.GetInputNameAllocated(0, alloc);
    const Ort::AllocatedStringPtr out_name = session.GetOutputNameAllocated(0, alloc);
    const char *in_names[] = {in_name.get()};
    const char *out_names[] = {out_name.get()};
    const Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // run inference
    constexpr int batch = 8;
    const optcnn::Timing t = optcnn::run_timed(args.runs, [&] {
        for (int base = 0; base < optcnn::NUM_IMAGES; base += optcnn::BATCH)
        {
            const int count = std::min(optcnn::BATCH, optcnn::NUM_IMAGES - base);
            for (int off = 0; off < count; off += batch)
            {
                const std::size_t b = std::min(batch, count - off);
                const std::size_t start = base + off;
                const int64_t shape[] = {static_cast<int>(b), optcnn::IMG_C, optcnn::IMG_H,
                                         optcnn::IMG_W};
                Ort::Value in =
                    Ort::Value::CreateTensor<float>(mem, Inputs + start * optcnn::IMG_SIZE,
                                                    b * optcnn::IMG_SIZE, shape, 4);
                auto outs = session.Run(Ort::RunOptions{nullptr}, in_names, &in, 1, out_names, 1);
                std::memcpy(Logits + start * optcnn::NUM_CLASSES, outs[0].GetTensorData<float>(),
                            sizeof(float) * b * optcnn::NUM_CLASSES);
            }
        }
    });

    fmt::print("inference: {} imgs over {} runs: best {:.1f}  mean {:.1f}  worst {:.1f} ms  "
               "(spread {:.1f}%, {:.1f} img/ms)\n",
               optcnn::NUM_IMAGES, args.runs, t.best, t.mean, t.worst,
               100.0 * (t.worst - t.best) / t.best, optcnn::NUM_IMAGES / t.best);

    optcnn::verify(Logits, RefLogits, optcnn::NUM_IMAGES, -1.0f);

    return 0;
}
