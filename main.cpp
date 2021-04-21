//
// Created by dianh on 2021/04/16.
//
#include <bits/stdc++.h>

#include "swin_transformer.h"
#include "linear.h"
#include "feed_forward.h"
#include "cyclic_shift.h"
#include "functions.h"
#include "tensor.h"
#include "patch_merging.h"
#include "window_attention.h"

using namespace std;
typedef float data_t;
//auto *swin_t = new shift_window_transformer::SwinTransformer<data_t>(
//        96, std::array<int, 4>{2, 2, 6, 2},
//        std::array<int, 4>{3, 6, 12, 24});
//auto *swin_s = new shift_window_transformer::SwinTransformer<data_t>(
//        96, std::array<int, 4>{2, 2, 18, 2},
//        std::array<int, 4>{3, 6, 12, 24});
//auto *swin_b = new shift_window_transformer::SwinTransformer<data_t>(
//        96, std::array<int, 4>{2, 2, 18, 2},
//        std::array<int, 4>{4, 8, 16, 32});
//auto *swin_l = new shift_window_transformer::SwinTransformer<data_t>(
//        96, std::array<int, 4>{2, 2, 18, 2},
//        std::array<int, 4>{6, 12, 24, 48});

int main() {
/*
 *  Linear Test
 */
    try {
        std::cout << "Linear Test: ";
        int in_feature = 50;
        int out_feature = 100;
        int batch = 10;
        auto *layer = new shift_window_transformer::Linear<data_t>(in_feature, out_feature);
        shift_window_transformer::Tensor<data_t> input = shift_window_transformer::Tensor<data_t>(in_feature * batch,
                                                                                                  0);
        input.shape = {10, 50};
        shift_window_transformer::Tensor<data_t> output{};
        layer->forward(input, output);
        std::cout << "Ok" << std::endl;
        delete layer;
    } catch (...) {
        std::cout << "Error" << std::endl;
    }


/*
 * Feed Forward Test
 */
    try {
        std::cout << "Feed Forward Test: ";
        int dim = 50;
        int hidden_dim = 10;
        int batch = 10;
        auto *layer = new shift_window_transformer::FeedForward<data_t>(dim, hidden_dim);
        shift_window_transformer::Tensor<data_t> input = shift_window_transformer::Tensor<data_t>(batch * dim, 0);
        input.shape = {10, 50};
        shift_window_transformer::Tensor<data_t> output{};
        layer->forward(input, output);
        std::cout << "Ok" << std::endl;
        delete layer;
    } catch (...) {
        std::cout << "Error" << std::endl;
    }
/*
 * Cyclic Shift Test
 */
    try {
        std::cout << "Cyclic Shift Test: ";

        auto *layer = new shift_window_transformer::CyclicShift<data_t>(5);
        shift_window_transformer::Tensor<data_t> input(3 * 224 * 224, 0);
        input.shape = {3, 224, 224};
        shift_window_transformer::Tensor<data_t> output{};
        layer->forward(input, output);
        std::cout << "Ok" << std::endl;
        delete layer;
    } catch (...) {
        std::cout << "Error" << std::endl;
    }
/*
 * Test get_relative_distances
 */
    try {
        auto *tensor = shift_window_transformer::get_relative_distances<data_t>(4);
        delete tensor;
    } catch (...) {

    }

    /*
     * Test Patch Merging
     */
    try {
        std::cout << "Test Patch Merging: ";
        int in_channels = 3;
        int out_channels = 96;
        int downscaling_size = 4;
        auto *layer = new shift_window_transformer::PatchMerging<data_t>(in_channels, out_channels, downscaling_size);
        shift_window_transformer::Tensor<data_t> input(3 * 224 * 224, 0);
        input.shape = {3, 224, 224};
        shift_window_transformer::Tensor<data_t> output{};
        layer->forward(input, output);
        delete layer;
        std::cout << "Ok" << std::endl;
    } catch (...) {
        std::cout << "Error" << std::endl;
    }
    /*
     * Test window attention
     *
     */
    try {
        std::cout << "Test Window Attention: ";
        int dim = 96;
        int heads = 12;
        int headDim = 96;
        bool shifted = false;
        int windowSize = 7;
        bool relativePosEmbedding = true;
        auto *attention = new shift_window_transformer::WindowAttention<data_t>(dim, heads, headDim, shifted,
                                                                                windowSize, relativePosEmbedding);
        delete attention;
        std::cout << "Ok" << std::endl;
    } catch (...) {
        std::cout << "Error" << std::endl;
    }
    /*
     * Test Swin Transformer
     */
//    try {
        std::cout << "Test Tiny Swin Transformer: ";
        auto *swin_tiny = new shift_window_transformer::SwinTransformer<data_t>(
                96, std::array<int, 4>{2, 2, 6, 2},
                std::array<int, 4>{3, 6, 12, 24});
        shift_window_transformer::Tensor<data_t> input(3 * 224 * 224, 0);
        input.shape = {3, 224, 224};
        shift_window_transformer::Tensor<data_t> output{};
        swin_tiny->forward(input, output);
        delete swin_tiny;
        std::cout << "Ok" << std::endl;
//    } catch (...) {
//        std::cout << "Error" << std::endl;
//    }
    return 0;
}

