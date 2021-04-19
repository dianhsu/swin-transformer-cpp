//
// Created by dianh on 2021/04/16.
//

#ifndef SWIN_TRANSFORMER_CPP_LINEAR_H
#define SWIN_TRANSFORMER_CPP_LINEAR_H

#include "tensor.h"
#include "layer.h"

namespace shift_window_transformer {
    template<typename T>
    class Linear : virtual public Layer<T> {
    public:
        Linear(int in_feature, int out_feature) : in_feature(in_feature), out_feature(out_feature), use_bias(true) {
            weights.resize(in_feature * out_feature);
            bias.resize(out_feature);
        }

        Linear(int in_feature, int out_feature, bool use_bias) : in_feature(in_feature),
                                                                 out_feature(out_feature),
                                                                 use_bias(use_bias) {
            weights.resize(in_feature * out_feature);
            if (use_bias)
                bias.resize(out_feature);
        }

/**
 *
 * @param input DIM x INPUT_FEATURE
 * @param output DIM X OUTPUT_FEATURE
 */
        void forward(const Tensor <T> &input, Tensor <T> &output) {
            output.clear();
            output.shape.clear();
            output.shape.insert(output.shape.end(), input.shape.begin(), input.shape.end());
            output.shape.back() = out_feature;
            for (auto pos = 0; pos < input.size(); pos += in_feature) {
                std::vector<T> tmp;
                if (use_bias) {
                    tmp = std::vector<T>(this->bias);
                } else {
                    tmp = std::vector<T>(out_feature, 0);
                }
                for (int i = 0; i < in_feature; ++i) {
                    for (int j = 0; j < out_feature; ++j) {
                        tmp[j] += input[i + pos] * this->weights[i * out_feature + j];
                    }
                }
                output.insert(output.end(), tmp.begin(), tmp.end());
            }
        }

    private:
        std::vector<T> weights, bias;
        int in_feature, out_feature;
        bool use_bias;
    };
}
#endif //SWIN_TRANSFORMER_CPP_LINEAR_H
