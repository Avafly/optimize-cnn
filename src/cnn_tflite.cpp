#include "common.h"

#include "3rdparty/fmt/base.h"

#include <omp.h>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model_builder.h>

#include <cstddef>
#include <cstring>
#include <memory>

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

    // init tflite
    const std::string model_path = args.path("tflite/model.tflite");
    std::unique_ptr<tflite::FlatBufferModel> model =
        tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
    if (model == nullptr)
    {
        fmt::print(stderr, "error: cannot load {}\n", model_path);
        return 1;
    }

    const int threads = args.threads < optcnn::MAX_THREADS ? args.threads : optcnn::MAX_THREADS;
    std::unique_ptr<tflite::Interpreter> interp[optcnn::MAX_THREADS];
    tflite::ops::builtin::BuiltinOpResolver resolver;
    for (int i = 0; i < threads; ++i)
    {
        if (tflite::InterpreterBuilder(*model, resolver)(&interp[i]) != kTfLiteOk ||
            interp[i]->SetNumThreads(1) != kTfLiteOk || interp[i]->AllocateTensors() != kTfLiteOk)
        {
            fmt::print(stderr, "error: cannot build interpreter\n");
            return 1;
        }
    }

    // run inference
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
                tflite::Interpreter *it = interp[tid].get();
                std::memcpy(it->typed_input_tensor<float>(0), Inputs + i * optcnn::IMG_SIZE,
                            sizeof(float) * optcnn::IMG_SIZE);
                it->Invoke();
                std::memcpy(Logits + i * optcnn::NUM_CLASSES, it->typed_output_tensor<float>(0),
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
