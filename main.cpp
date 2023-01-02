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

using namespace std;

#define CHECK_NNAPI_ERROR(status)                                       \
    if (status != ANEURALNETWORKS_NO_ERROR)                             \
    {                                                                   \
        std::cerr << status << ", line: " << __LINE__ << std::endl;     \
        exit(1);                                                        \
    }


int main(int argc, char *argv[]) {
    uint32_t numDevices;
    CHECK_NNAPI_ERROR(ANeuralNetworks_getDeviceCount(&numDevices));

    cout << "Found " << numDevices << " devices" << endl;

    ANeuralNetworksDevice *device = nullptr;
    const char *buffer = nullptr;
    long feature;
    int type;
    for (int i = 0; i < numDevices; i++) {
        CHECK_NNAPI_ERROR(ANeuralNetworks_getDevice(i, &device));
        CHECK_NNAPI_ERROR(ANeuralNetworksDevice_getName(device, &buffer));
        string name(buffer);
        CHECK_NNAPI_ERROR(ANeuralNetworksDevice_getType(device, &type));
        CHECK_NNAPI_ERROR(ANeuralNetworksDevice_getFeatureLevel(device, &feature));
        CHECK_NNAPI_ERROR(ANeuralNetworksDevice_getVersion(device, &buffer));
        string version(buffer);

        string
        type_str = (const string[]) {
                "unknown",
                "other",
                "cpu",
                "gpu",
                "accelerator"
        }[type];

        if (feature <= 31)
            feature -= 26;
        else if (feature > 1000000)
            feature -= 1000000;

        cout << name << " (type: " << type_str << ", feature: " << feature << ", version: " << version << ")" << endl;;
    }


    ANeuralNetworksModel *model;
    CHECK_NNAPI_ERROR(ANeuralNetworksModel_create(&model));

    // CHECK_NNAPI_ERROR(ANeuralNetworksModel_relaxComputationFloat32toFloat16(model, true));
    // CHECK_NNAPI_ERROR(ANeuralNetworksModel_finish(model));

    ANeuralNetworksModel_free(model);
    return 0;
}