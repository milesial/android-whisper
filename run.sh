#!/usr/bin/env sh

adb push ./build/nnapi-whisper /data/local/tmp
adb shell /data/local/tmp/nnapi-whisper
