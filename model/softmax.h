//
// Created by dianh on 2021/04/21.
//

#ifndef SWIN_TRANSFORMER_CPP_SOFTMAX_H
#define SWIN_TRANSFORMER_CPP_SOFTMAX_H

#include "layer.h"
#include "tensor.h"

namespace shift_window_transformer {
    template<typename T>
    class SoftMax : virtual public Layer<T> {
    public:
        void forward(const Tensor <T> &input, Tensor <T> &output) {
            int dim = input.shape.back();
            output.clear();
            output.shape = input.shape;
            for (int i = 0; i < input.size(); i += dim) {
                T sum = 0;
                for (int j = 0; j < dim; ++j) {
                    sum += exp(input[i + j]);
                }
                for (int j = 0; j < dim; ++j) {
                    output.push_back(exp(input[i + j]) / sum);
                }
            }
        }

        long long parameterCount() {
            return 0;
        }
    };
}
#endif //SWIN_TRANSFORMER_CPP_SOFTMAX_H
