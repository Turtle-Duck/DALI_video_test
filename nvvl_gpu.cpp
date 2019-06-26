// Opencv
#include <opencv2/opencv.hpp>

// NVVL
#include <cuda.h>
#include "VideoLoader.h"
#include "cuda/utils.h"

#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <string>
#include <sys/types.h>
#include <dirent.h>
#include <fstream>
#include <random>
#include <time.h>


using namespace std;
using namespace cv;
using PictureSequence = NVVL::PictureSequence;

int main(int argc, char** argv) {

    if (argc < 2) {
        std::cout << "usage: " << argv[0] << " <video file>\n";
        return -1;
    }

    constexpr auto sequence_count = uint16_t{4};
    constexpr auto crop_width = int16_t{128};
    constexpr auto crop_height = int16_t{128};
    constexpr auto device_id = 0;
    constexpr auto batch_size = uint16_t{4};

    auto loader = NVVL::VideoLoader{device_id};
    auto filename = argv[1];
    auto frame_count = loader.frame_count(filename);
    std::cout << "Looks like there are " << frame_count << " frames" << std::endl;
    loader.read_sequence(filename, 0, frame_count); // (filename, frame_num, sequence_length);
    auto seq = PictureSequence{sequence_count};
    auto pixels = PictureSequence::Layer<float>{};

    float* data = nullptr;
    size_t pitch = 0;
    cudaMallocPitch(&data, &pitch,
                    crop_width * sizeof(float),
                    crop_height * sequence_count * 3);
    pixels.data = data;
    pixels.desc.count = sequence_count;
    pixels.desc.channels = 3;
    pixels.desc.width = crop_width;
    pixels.desc.crop_x = 0;
    pixels.desc.height = crop_height;
    pixels.desc.crop_y = 0;
    pixels.desc.horiz_flip = false;
    pixels.desc.normalized = false;
    pixels.desc.color_space = ColorSpace_RGB;
    pixels.desc.stride.c = 1;
    pixels.desc.stride.x = 3;
    pixels.desc.stride.y = pitch / sizeof(float) * 3;
    pixels.desc.stride.n = pixels.desc.stride.y * crop_height;

    // read from gpu
    seq.set_layer("data", pixels);
    loader.receive_frames_sync(seq);
    std::cout << "Receive down." << std::endl;
    constexpr auto sample_count = 128 * 128 * 3;
    auto frame_nums = seq.get_meta<int>("frame_num");
    std::cout << "Got a sequence of size: " << seq.count() << std::endl;

    // print the first 4 frames:
    for (int i = 0; i < seq.count(); ++i) {
        auto pixels_temp = seq.get_layer<float>("data", i);
        size_t data_stride = pixels_temp.desc.stride.y;
        auto data_temp = pixels_temp.data;
        float tmp[sample_count];
        if (cudaMemcpy(tmp, data_temp, sample_count * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
            throw std::runtime_error("Couldn't copy frame data to cpu");
        }
        std::cout << "Frame " << i << ":" << std::endl;
        for (int j = 0; j < 10; j++) {
            std::cout << "j = " << j << std::endl;
            for (int q = 0; q < 4; q++){
                for (int c = 0; c < 3; c++){
                    std::cout << tmp[j*128*3+q*3+c] << " ";
                }
            }
            std::cout << std::endl;
        }
    }

    return 0;
}



