//
// Created by dianh on 2021/04/16.
//

#ifndef SWIN_TRANSFORMER_CPP_RESIDUAL_H
#define SWIN_TRANSFORMER_CPP_RESIDUAL_H

#include "layer.h"
#include "tensor.h"
#include "layer_norm.h"

namespace shift_window_transformer {
    template<typename T>
    class Residual : virtual public Layer<T> {
    public:
        Residual() : fn(nullptr) {
        }

        explicit Residual(Layer<T> *fn) : fn(fn) {
        }

        long long parameterCount() {
            if (fn != nullptr) {
                return fn->parameterCount();
            }
            return 0;
        }

        void forward(const Tensor<T> &input, Tensor<T> &output) {
            assert(fn != nullptr);
            fn->forward(input, output);
            for (int i = 0; i < output.size(); ++i) {
                output[i] += input[i];
            }
        }

    private:
        Layer<T> *fn;
    };
}
#endif //SWIN_TRANSFORMER_CPP_RESIDUAL_H
