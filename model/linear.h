//
// Created by dianh on 2021/04/16.
//

#ifndef SWIN_TRANSFORMER_CPP_LINEAR_H
#define SWIN_TRANSFORMER_CPP_LINEAR_H

#include "tensor.h"
#include "layer.h"

namespace shift_window_transformer {
    template<typename T>
    class Linear : public Layer<T> {
    public:


        Linear(int in_feature, int out_feature, bool use_bias) : in_feature(in_feature),
                                                                 out_feature(out_feature),
                                                                 use_bias(use_bias) {
            weights = new std::vector<T>(in_feature * out_feature, 0);
            if (use_bias)
                bias = new std::vector<T>(out_feature, 0);
        }

        long long parameterCount() {
            if (use_bias)
                return weights->size() + bias->size();
            return weights->size();
        }

        Linear(int i_feature, int o_feature) : Linear(i_feature, o_feature, true) {
        }

        ~Linear() {
            if (weights != nullptr) {
                delete weights;
                weights = nullptr;
            }
            if (bias != nullptr) {
                delete bias;
                bias = nullptr;
            }
        }

/**
 *
 * @param input DIM x INPUT_FEATURE
 * @param output DIM X OUTPUT_FEATURE
 */
        void forward(const Tensor <T> &input, Tensor <T> &output) {
            assert(input.shape.size() and input.shape.back() == in_feature);
            output.clear();
            output.shape.clear();
            output.shape.insert(output.shape.end(), input.shape.begin(), input.shape.end());
            output.shape.back() = out_feature;
            for (auto pos = 0; pos < input.size(); pos += in_feature) {
                std::vector<T> tmp{};
                if (use_bias) {
                    for (auto item: *(this->bias)) {
                        tmp.push_back(item);
                    }
                } else {
                    for (int i = 0; i < out_feature; ++i) {
                        tmp.push_back(0);
                    }
                }
                for (int i = 0; i < in_feature; ++i) {
                    for (int j = 0; j < out_feature; ++j) {
                        tmp[j] += input[i + pos] * (*(this->weights))[i * out_feature + j];
                    }
                }
                output.insert(output.end(), tmp.begin(), tmp.end());
            }
            assert(output.size() == input.size() / in_feature * out_feature);
        }

    private:
        std::vector<T> *weights = nullptr, *bias = nullptr;
        int in_feature{}, out_feature{};
        bool use_bias{};
    };
}
#endif //SWIN_TRANSFORMER_CPP_LINEAR_H
