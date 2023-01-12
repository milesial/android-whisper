#include "include/onnxruntime_cxx_api.h"
#include "include/nnapi_provider_factory.h"
#include <chrono>
#include <iostream>

using namespace std::chrono;


int main(int argc, char *argv[]) {
    Ort::Env env = Ort::Env{ORT_LOGGING_LEVEL_VERBOSE, "Default"};
    Ort::SessionOptions so;
    uint32_t nnapi_flags = NNAPI_FLAG_USE_FP16 | NNAPI_FLAG_CPU_DISABLED;
    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Nnapi(so, nnapi_flags));
    Ort::Session session(env, "/data/local/tmp/linear.onnx", so);

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    std::vector<float> input(3000 * 64);
    fill(input.begin(), input.end(), 1.0);

    float output[3000 * 64];
    long shape[] = {3000, 64};
    auto input_tensor = Ort::Value::CreateTensor<float>(memory_info, input.data(), 3000 * 64, shape, 2);
    auto output_tensor = Ort::Value::CreateTensor<float>(memory_info, output, 3000 * 64, shape, 2);

    Ort::RunOptions run_options;
    const char *input_names[] = {"onnx::Gemm_0"};
    const char *output_names[] = {"3"};

    auto start = high_resolution_clock::now();
    session.Run(run_options, input_names, &input_tensor, 1, output_names, &output_tensor, 1);

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    std::cout << duration.count() << std::endl;
    std::cout << "Finished " << output[0] << std::endl;
}