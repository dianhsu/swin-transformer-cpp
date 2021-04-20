//
// Created by dianh on 2021/04/16.
//

#ifndef SWIN_TRANSFORMER_CPP_FUNCTIONS_H
#define SWIN_TRANSFORMER_CPP_FUNCTIONS_H

#include <algorithm>
#include <cmath>

namespace shift_window_transformer {
    template<typename T>
    T GELU(T x) {
        return x * (1 + tanh(sqrt(2 / M_PI) * (x + 0.044715 * x * x * x)));
        // approximate function provide by https://arxiv.org/pdf/1606.08415.pdf
    }

    template<typename T>
    Tensor <T> *get_relative_distances(int window_size) {
        auto *ret = new Tensor<T>{};
        std::vector<std::pair<T, T>> tmp{};
        for (int i = 0; i < window_size; ++i) {
            for (int j = 0; j < window_size; ++j) {
                tmp.emplace_back((T) i, (T) j);
            }
        }
        for (int i1 = 0; i1 < window_size; ++i1) {
            for (int j1 = 0; j1 < window_size; ++j1) {
                for (int i2 = 0; i2 < window_size; ++i2) {
                    for (int j2 = 0; j2 < window_size; ++j2) {
                        ret->push_back(tmp[i2 * window_size + j2].first - tmp[i1 * window_size + j1].first);
                        ret->push_back(tmp[i2 * window_size + j2].second - tmp[i1 * window_size + j1].second);
                    }
                }
            }
        }
        ret->shape = {
                window_size * window_size,
                window_size * window_size,
                2
        };
        return ret;
    }

    template<typename T>
    Tensor <T> *create_mask(int window_size, int displacement, bool upper_lower, bool left_right) {
        int size_v2 = window_size * window_size;
        auto *ret = new Tensor<T>(size_v2 * size_v2, 0);
        ret->shape = {size_v2, size_v2};
        if (upper_lower) {
            for (int pos = 0; pos < size_v2 * size_v2; ++pos) {
                int d1 = pos / size_v2;
                int d2 = pos % size_v2;
                if ((d1 >= -displacement * window_size and d2 < -displacement * window_size) or
                    (d2 >= -displacement * window_size and d1 < -displacement * window_size)) {
                    (*ret)[pos] = -FP_INFINITE;
                }
            }
        }
        if (left_right) {
            for (int pos = 0; pos < size_v2 * size_v2; ++pos) {
                int tmp = pos;
                int d4 = tmp % window_size;
                tmp /= size_v2;
                int d2 = tmp % window_size;
                if ((d2 >= -displacement and d4 < -displacement) or
                    (d2 < -displacement and d4 >= -displacement)) {
                    (*ret)[pos] = -FP_INFINITE;
                }
            }
        }
        return ret;
    }
}
#endif //SWIN_TRANSFORMER_CPP_FUNCTIONS_H
