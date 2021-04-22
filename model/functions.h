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
        return 0.5 * x * (1 + tanh(sqrt(2.0 / M_PI) * (x + 0.044715 * x * x * x)));
        // approximate function provide by https://arxiv.org/pdf/1606.08415.pdf
    }

    template<typename T>
    Tensor <T> *get_relative_distances(int windowSize) {
        auto *ret = new Tensor<T>{};
        std::vector<std::pair<T, T>> tmp{};
        for (int i = 0; i < windowSize; ++i) {
            for (int j = 0; j < windowSize; ++j) {
                tmp.emplace_back((T) i, (T) j);
            }
        }
        for (int i1 = 0; i1 < windowSize; ++i1) {
            for (int j1 = 0; j1 < windowSize; ++j1) {
                for (int i2 = 0; i2 < windowSize; ++i2) {
                    for (int j2 = 0; j2 < windowSize; ++j2) {
                        ret->push_back(tmp[i2 * windowSize + j2].first - tmp[i1 * windowSize + j1].first);
                        ret->push_back(tmp[i2 * windowSize + j2].second - tmp[i1 * windowSize + j1].second);
                    }
                }
            }
        }
        ret->shape = {
                windowSize * windowSize,
                windowSize * windowSize,
                2
        };
        return ret;
    }

    template<typename T>
    Tensor <T> *create_mask(int windowSize, int displacement, bool upperLower, bool leftRight) {
        int sizeV2 = windowSize * windowSize;
        auto *ret = new Tensor<T>(sizeV2 * sizeV2, 0);
        ret->shape = {sizeV2, sizeV2};
        if (upperLower) {
            for (int pos = 0; pos < sizeV2 * sizeV2; ++pos) {
                int d1 = pos / sizeV2;
                int d2 = pos % sizeV2;
                if ((d1 >= sizeV2 - displacement * windowSize and d2 < sizeV2 - displacement * windowSize) or
                    (d1 < sizeV2 - displacement * windowSize and d2 >= sizeV2 - displacement * windowSize)) {
                    (*ret)[pos] = -INFINITY;
                }
            }
        }
        if (leftRight) {
            for (int pos = 0; pos < sizeV2 * sizeV2; ++pos) {
                int tmp = pos;
                int d4 = tmp % windowSize;
                tmp /= sizeV2;
                int d2 = tmp % windowSize;
                if ((d2 >= windowSize - displacement and d4 < windowSize - displacement) or
                    (d2 < windowSize - displacement and d4 >= windowSize - displacement)) {
                    (*ret)[pos] = -INFINITY;
                }
            }
        }
        return ret;
    }
}
#endif //SWIN_TRANSFORMER_CPP_FUNCTIONS_H
