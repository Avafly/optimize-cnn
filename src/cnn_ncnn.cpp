#include "common.h"

#include "3rdparty/fmt/base.h"

#include <net.h>
#include <omp.h>

#include <cstddef>
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

    // init ncnn
    ncnn::Net net;
    net.opt.num_threads = 1;
    net.opt.use_packing_layout = false;
    net.opt.use_fp16_packed = false;
    net.opt.use_fp16_storage = false;
    net.opt.use_fp16_arithmetic = false;
    if (net.load_param(args.path("ncnn/model.ncnn.param").c_str()) ||
        net.load_model(args.path("ncnn/model.ncnn.bin").c_str()))
    {
        fmt::print(stderr, "error: cannot load model\n");
        return 1;
    }

    static ncnn::UnlockedPoolAllocator blob_pool[optcnn::MAX_THREADS];
    static ncnn::UnlockedPoolAllocator workspace_pool[optcnn::MAX_THREADS];

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
                ncnn::Mat in(optcnn::IMG_W, optcnn::IMG_H, Inputs + i * optcnn::IMG_SIZE);
                ncnn::Mat out;
                ncnn::Extractor ex = net.create_extractor();
                ex.set_blob_allocator(&blob_pool[tid]);
                ex.set_workspace_allocator(&workspace_pool[tid]);
                if (ex.input("in0", in) != 0 || ex.extract("out0", out) != 0)
                    continue;
                std::memcpy(Logits + i * optcnn::NUM_CLASSES, out.data,
                            sizeof(float) * optcnn::NUM_CLASSES);
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
