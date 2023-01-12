#!/usr/bin/env sh

mkdir -p build && cd build

cmake -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake \
      -DANDROID_ABI=arm64-v8a \
      -DANDROID_PLATFORM=android-29 ..

make

path="/data/local/tmp/"
adb push ./nnapi-whisper-nnapi $path
adb shell $path/nnapi-whisper-nnapi

