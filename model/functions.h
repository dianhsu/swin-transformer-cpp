//
// Created by dianh on 2021/04/16.
//

#ifndef SWIN_TRANSFORMER_CPP_FUNCTIONS_H
#define SWIN_TRANSFORMER_CPP_FUNCTIONS_H

#include <cmath>

namespace shift_window_transformer {
    template<typename T>
    T GELU(T x) {
        return x * (1 + tanh(sqrt(2 / M_PI) * (x + 0.044715 * x * x * x)));
        // approximate function provide by https://arxiv.org/pdf/1606.08415.pdf
    }

    std::vector<std::pair<int, int>> *get_relative_distances(int window_size) {
        auto *ret = new std::vector<std::pair<int, int>>{};
        std::vector<std::pair<int, int>> tmp{};
        for (int i = 0; i < window_size; ++i) {
            for (int j = 0; j < window_size; ++j) {
                tmp.emplace_back(i, j);
            }
        }
        for (int i1 = 0; i1 < window_size; ++i1) {
            for (int j1 = 0; j1 < window_size; ++j1) {
                for (int i2 = 0; i2 < window_size; ++i2) {
                    for (int j2 = 0; j2 < window_size; ++j2) {
                        ret->emplace_back(tmp[i2 * window_size + j2].first - tmp[i1 * window_size + j1].first,
                                          tmp[i2 * window_size + j2].second - tmp[i1 * window_size + j1].second);
                    }
                }
            }
        }
        return ret;
    }

    template<typename T>
    std::vector<T> *create_mask(int window_size, int displacement, bool upper_lower, bool left_right) {
        auto *ret = new std::vector<T>(window_size * window_size * window_size * window_size, 0);
        // TODO: not understand what it means.
        return ret;
    }
}
#endif //SWIN_TRANSFORMER_CPP_FUNCTIONS_H
