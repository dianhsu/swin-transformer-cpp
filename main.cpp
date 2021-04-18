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
using namespace std;
typedef float data_t;
auto *swin_t = new shift_window_transformer::SwinTransformer<data_t>(
        96, std::array<int, 4>{2, 2, 6, 2},
        std::array<int, 4>{3, 6, 12, 24});
auto *swin_s = new shift_window_transformer::SwinTransformer<data_t>(
        96, std::array<int, 4>{2, 2, 18, 2},
        std::array<int, 4>{3, 6, 12, 24});
auto *swin_b = new shift_window_transformer::SwinTransformer<data_t>(
        96, std::array<int, 4>{2, 2, 18, 2},
        std::array<int, 4>{4, 8, 16, 32});
auto *swin_l = new shift_window_transformer::SwinTransformer<data_t>(
        96, std::array<int, 4>{2, 2, 18, 2},
        std::array<int, 4>{6, 12, 24, 48});

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
        shift_window_transformer::Tensor<data_t> input = shift_window_transformer::Tensor<data_t>(in_feature * batch, 0);
        shift_window_transformer::Tensor<data_t> output{};
        layer->forward(input, output);
        std::cout << "Ok" << std::endl;
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
        shift_window_transformer::Tensor<data_t> output{};
        layer->forward(input, output);
        std::cout << "Ok" << std::endl;
    } catch (...) {
        std::cout << "Error" << std::endl;
    }
/*
 * Cyclic Shift Test
 */
    try {
        std::cout << "Cyclic Shift Test: ";
        std::array<int, 3> arr{3, 224, 224};
        auto *layer = new shift_window_transformer::CyclicShift<data_t>(arr, 5);
        shift_window_transformer::Tensor<data_t> input(3 * 224 * 224, 0);
        shift_window_transformer::Tensor<data_t> output{};
        layer->forward(input, output);
        std::cout << "Ok" << std::endl;
    } catch (...) {
        std::cout << "Error" << std::endl;
    }
/*
 * Test get_relative_distances
 */
    try {
        auto *tensor = shift_window_transformer::get_relative_distances(4);
        delete tensor;
    } catch (...) {

    }
    return 0;
}

