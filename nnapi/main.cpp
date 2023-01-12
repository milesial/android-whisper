#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <functional>
#include <unordered_map>

#include <android/NeuralNetworks.h>
#include <android/sharedmem.h>
#include <sys/system_properties.h>
#include <sys/mman.h>
#include <random>
#include <cmath>
#include <unistd.h>

using namespace std;

#define CHECK_NNAPI_ERROR(status)                                       \
    if (status != ANEURALNETWORKS_NO_ERROR)                             \
    {                                                                   \
        std::cerr << status << ", line: " << __LINE__ << std::endl;     \
        exit(1);                                                        \
    }

static int operand_idx = 0;
static std::unordered_map <uint32_t, std::vector<uint32_t>> operand_dims;

uint32_t addTensorOperand(ANeuralNetworksModel *model, std::vector <uint32_t> dims, const void *srcbuffer = nullptr) {
    ANeuralNetworksOperandType operandType;
    operandType.type = ANEURALNETWORKS_TENSOR_FLOAT32;
    operandType.dimensionCount = static_cast<uint32_t>(dims.size());
    operandType.dimensions = dims.data();
    operandType.scale = 0.0f;
    operandType.zeroPoint = 0;
    CHECK_NNAPI_ERROR(ANeuralNetworksModel_addOperand(model, &operandType));

    if (srcbuffer != nullptr) {
        const size_t bytes = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<uint32_t>()) * sizeof(float);
        CHECK_NNAPI_ERROR(ANeuralNetworksModel_setOperandValue(model, operand_idx, srcbuffer, bytes));
    }
    operand_dims[operand_idx] = dims;
    return operand_idx++;
}

uint32_t addScalarOperand(ANeuralNetworksModel *model, int32_t value = -999) {
    ANeuralNetworksOperandType operandType;
    operandType.type = ANEURALNETWORKS_INT32;
    operandType.dimensionCount = 0;
    operandType.dimensions = NULL;
    operandType.scale = 0.0f;
    operandType.zeroPoint = 0;
    CHECK_NNAPI_ERROR(ANeuralNetworksModel_addOperand(model, &operandType));
    if (value != -999)
        CHECK_NNAPI_ERROR(ANeuralNetworksModel_setOperandValue(model, operand_idx, &value, sizeof(value)));

    return operand_idx++;
}

uint32_t addFullyConnected(ANeuralNetworksModel *model, uint32_t input, uint32_t weight,
                           int32_t activation = ANEURALNETWORKS_FUSED_NONE, int32_t bias = -1) {
    uint32_t act = addScalarOperand(model, activation);
    uint32_t output = addTensorOperand(model, {operand_dims[input][0], operand_dims[weight][0]});

    if (bias < 0) {
        std::vector<float> zeros(operand_dims[weight][0]);
        fill(zeros.begin(), zeros.end(), 0.0);
        bias = addTensorOperand(model, {operand_dims[weight][0]}, zeros.data());
    }
    ANeuralNetworksModel_addOperation(model, ANEURALNETWORKS_FULLY_CONNECTED, 4,
                                      (uint32_t[]) {input, weight, (uint32_t) bias, act},
                                      1, (uint32_t[]) {output});
    return output;
}

uint32_t addAdd(ANeuralNetworksModel *model, uint32_t input1, uint32_t input2) {
    uint32_t act = addScalarOperand(model, ANEURALNETWORKS_FUSED_NONE);
    uint32_t output = addTensorOperand(model, operand_dims[input1]);
    ANeuralNetworksModel_addOperation(model, ANEURALNETWORKS_ADD, 3, (uint32_t[]) {input1, input2, act}, 1,
                                      (uint32_t[]) {output});
    return output;
}

int main(int argc, char *argv[]) {
    uint32_t numDevices;
    int type;
    CHECK_NNAPI_ERROR(ANeuralNetworks_getDeviceCount(&numDevices));
    ANeuralNetworksDevice *cpu_device;
    for (int i = 0; i < numDevices; i++) {
        CHECK_NNAPI_ERROR(ANeuralNetworks_getDevice(i, &cpu_device));
        CHECK_NNAPI_ERROR(ANeuralNetworksDevice_getType(cpu_device, &type));
        if (type == 2)
            break;
    }

    ANeuralNetworksModel *model;
    CHECK_NNAPI_ERROR(ANeuralNetworksModel_create(&model));

    uint32_t input = addTensorOperand(model, {3000, 32});
    std::vector<float> weight_buffer(32 * 40);
    fill(weight_buffer.begin(), weight_buffer.end(), 1.0);

    uint32_t weight = addTensorOperand(model, {40, 32}, weight_buffer.data());
    uint32_t output = addFullyConnected(model, input, weight);

    ANeuralNetworksModel_identifyInputsAndOutputs(model, 1, &input, 1, &output);
    CHECK_NNAPI_ERROR(ANeuralNetworksModel_relaxComputationFloat32toFloat16(model, false));
    CHECK_NNAPI_ERROR(ANeuralNetworksModel_finish(model));


    ANeuralNetworksCompilation *compilation;

    CHECK_NNAPI_ERROR(ANeuralNetworksCompilation_create(model, &compilation));
    //CHECK_NNAPI_ERROR(ANeuralNetworksCompilation_createForDevices(model, &cpu_device, 1, &compilation));

    //CHECK_NNAPI_ERROR(ANeuralNetworksCompilation_setPreference(compilation, ANEURALNETWORKS_PREFER_LOW_POWER));
    uint8_t token[ANEURALNETWORKS_BYTE_SIZE_OF_CACHE_TOKEN] = {0x47};
    CHECK_NNAPI_ERROR(ANeuralNetworksCompilation_setCaching(compilation, "/data/local/tmp/", token));
    CHECK_NNAPI_ERROR(ANeuralNetworksCompilation_finish(compilation));


    ANeuralNetworksExecution *run = NULL;
    CHECK_NNAPI_ERROR(ANeuralNetworksExecution_create(compilation, &run));

    std::vector<float> input_buffer(3000 * 32);
    fill(input_buffer.begin(), input_buffer.end(), 2.0);
    CHECK_NNAPI_ERROR(
            ANeuralNetworksExecution_setInput(run, 0, NULL, input_buffer.data(), input_buffer.size() * sizeof(float)));

    std::vector<float> output_buffer(3000 * 40);
    CHECK_NNAPI_ERROR(ANeuralNetworksExecution_setOutput(run, 0, NULL, output_buffer.data(), output_buffer.size() * sizeof(float)));
    //CHECK_NNAPI_ERROR(ANeuralNetworksExecution_compute(run));

    ANeuralNetworksEvent *run_end = NULL;
    ANeuralNetworksExecution_startCompute(run, &run_end);

    ANeuralNetworksEvent_wait(run_end);
    ANeuralNetworksEvent_free(run_end);
    ANeuralNetworksExecution_free(run);

    cout << output_buffer[32] << endl;

    ANeuralNetworksCompilation_free(compilation);
    ANeuralNetworksModel_free(model);
    return 0;
}