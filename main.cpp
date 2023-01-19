#include "include/onnxruntime_cxx_api.h"
#include "include/nnapi_provider_factory.h"
#include <chrono>
#include <iostream>

using namespace std::chrono;


int main(int argc, char *argv[]) {
    long audio_shape[] = {16000 * 30};
    long spectogram_shape[] = {1, 80, 3000};
    long features_shape[] = {1, 1500, 1280};

    Ort::RunOptions run_options;
    const char *input_names[] = {"input"};
    const char *output_names[] = {"output"};

    Ort::Env env = Ort::Env{ORT_LOGGING_LEVEL_VERBOSE, "Default"};
    Ort::SessionOptions so;
    //uint32_t nnapi_flags = NNAPI_FLAG_USE_FP16;
    //Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Nnapi(so, nnapi_flags));
    //bool enable_cpu_mem_arena = true;
    //Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_ACL(so, enable_cpu_mem_arena));

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

    /* spectrogram */
    Ort::Session spectogram_session(env, "/data/local/tmp/model/spectrogram.onnx", so);
    std::vector<float> audio(audio_shape[0]);
    std::vector<float> spectogram(spectogram_shape[0] * spectogram_shape[1] * spectogram_shape[2]);
    fill(audio.begin(), audio.end(), 1.0);

    auto audio_tensor = Ort::Value::CreateTensor<float>(memory_info, audio.data(), audio.size(), audio_shape, 1);
    auto spectogram_tensor = Ort::Value::CreateTensor<float>(memory_info, spectogram.data(), spectogram.size(), spectogram_shape, 3);

    auto start = high_resolution_clock::now();
    spectogram_session.Run(run_options, input_names, &audio_tensor, 1, output_names, &spectogram_tensor, 1);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    std::cout << duration.count() << std::endl;


    /* encoder */
    Ort::Session encoder_session(env, "/data/local/tmp/model/encoder.onnx", so);
    std::vector<float> features(features_shape[0] * features_shape[1] * features_shape[2]);

    auto features_tensor = Ort::Value::CreateTensor<float>(memory_info, features.data(), features.size(), features_shape, 3);

    start = high_resolution_clock::now();
    encoder_session.Run(run_options, input_names, &spectogram_tensor, 1, output_names, &features_tensor, 1);
    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);
    std::cout << duration.count() << std::endl;

    std::cout << "Finished " << features[0] << std::endl;
}