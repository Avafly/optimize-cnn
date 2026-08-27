# optimize-cnn

We sometimes need to run the AI models on devices with very limited resources. Many inference frameworks, e.g., [ORT](https://github.com/microsoft/onnxruntime) and [ncnn](https://github.com/Tencent/ncnn), perform well for larger models but fall short with small models: their optimizations for large models do not always benefit smaller ones (small/flat GEMMs, small convolutions with few channels, ...), and performance is traded away for generality.

This repo shows a tiny CNN implementation for recognizing fashion mnist. Simple, powerful enough, while extremely fast. In most real projects the hardware specs are known and the model architecture is fixed, which leaves a lot of room to optimize, and a tiny model means hand writing it is not much work. **As long as the model is small, the implementation is worth considering.**

Here I optimize a tiny CNN step by step, each version within a few dozen lines of changes.

## Task

Inference follows a typical real-time pattern: one batch of data arrives at a time, and the next one only after the current one is done. Take pill defect inspection as an example: one blister pack enters, its pills are checked for defects, and only then does the next pack enter.

Convolution weights are OIHW, images are BCHW. Training and data export are both in [train.py](https://github.com/Avafly/optimize-cnn/blob/main/python/train.py). The exported data is available in [releases](https://github.com/Avafly/optimize-cnn/releases).

The target hardware is a RPi 4B (4x Cortex-A72, 4GB RAM).

## Benchmark

### Speed

Each inference run recognizes 70,000 images. After one warmup it runs `--runs` times, and the fastest one is taken as the score.

| Implementation |  Elapsed time to run 70,000 images |
| :------------: | :--------------------------------: |
|    cnn_ort     |              2440 ms               |
|    cnn_ncnn    |              2405 ms               |
| **cnn_struct** |            **2116 ms**             |
| **cnn_const**  |            **1736 ms**             |
|  **cnn_neon**  |            **1045 ms**             |
|  **cnn_fuse**  |             **783 ms**             |

## Optimization

### cnn_ort & cnn_ncnn

Both framework implementations were tuned rather than left at their defaults. In [ncnn](https://github.com/Avafly/optimize-cnn/blob/main/src/cnn_ncnn.cpp), each thread gets its own allocator instead of sharing the default pool, and fp16 and pack are disabled (tested faster). In [ORT](https://github.com/Avafly/optimize-cnn/blob/main/src/cnn_ort.cpp), the best-performing input tensor shape was chosen and graph optimization enabled. Both land around 2400 ms at 4 threads. That is the baseline.

### cnn_struct

`layers.h` defines the layer implementations, and [cnn_struct](https://github.com/Avafly/optimize-cnn/blob/main/src/cnn_struct.cpp) builds the model from a config. Convolution uses the classic im2col + GEMM, everything else is a naive implementation. Its speed is 2116 ms.

### cnn_const

Turning key variables into constants lets the compiler optimize more aggressively. [cnn_const](https://github.com/Avafly/optimize-cnn/blob/main/src/cnn_const.cpp) hardcodes the model architecture, and its speed reaches 1736 ms.

### cnn_neon

The model architecture gives the MAC count and a theoretical ceiling. Measuring each layer's actual cost against that ceiling shows which part has the largest gap (most worth optimizing). The theoretical and actual cost of each layer is roughly:

| Layer |             Actual              | Theoretical |
| :---: | :-----------------------------: | :---------: |
| conv1 | im2col (291 ms) + gemm (355 ms) |   137 ms    |
| relu1 |             104 ms              |      —      |
| pool1 |              68 ms              |      —      |
| conv2 | im2col (154 ms) + gemm (669 ms) |   403 ms    |
| relu2 |              37 ms              |      —      |
| pool2 |              19 ms              |      —      |
|  fc   |              97 ms              |    14 ms    |

*The theoretical column can be computed. A 128-bit NEON `fmla` does 4 fp32 MACs and the A72 issues one per cycle. Conv1 needs 56448 MACs per image, so it cannot beat 56448/4 x 70000 / 4 threads / 1.8 GHz = 137 ms.*

In a normal model the data copying of im2col takes far less time than the computation, but in this tiny model it takes a third of the time. So the convolution weights are packed and the [GEMM hand-written](https://github.com/Avafly/optimize-gemm) to replace OpenBLAS, which avoids im2col. The FC layer is done the same way: pack the parameters, then implement it with NEON. This removes im2col and the OpenBLAS dependency. After the optimization [cnn_neon](https://github.com/Avafly/optimize-cnn/blob/main/src/cnn_neon.cpp) reaches 1045 ms.

### cnn_fuse

Max pooling is moved ahead of the activation since `leaky(max(x)) == max(leaky(x))`. The activation then runs on 4x fewer elements.

Then convolution, 2x2 max pooling and the activation become a single kernel: two conv output rows are kept in registers and pooled there, so the conv output never reaches memory. According to the measurement, the activation has to be applied while the pooled value is still in a register: left as a separate pass over the pooled buffer it costs 220 ms, against 14 ms for the same pass after the unfused pool. [cnn_fuse](https://github.com/Avafly/optimize-cnn/blob/main/src/cnn_fuse.cpp) reaches 783 ms.

## References

https://github.com/Avafly/optimize-gemm

https://github.com/BVLC/caffe

https://github.com/OpenMathLib/OpenBLAS
