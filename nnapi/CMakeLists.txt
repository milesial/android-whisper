cmake_minimum_required(VERSION 3.4.1)

project(nnapi-whisper-nnapi)

set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_STANDARD 17)

find_library(libneuralnetworks neuralnetworks)
find_library(libandroid android)

add_executable(${PROJECT_NAME} main.cpp)

target_link_libraries(${PROJECT_NAME}
                      ${libneuralnetworks}
                      ${libandroid})


