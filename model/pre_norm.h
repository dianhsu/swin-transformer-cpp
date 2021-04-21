//
// Created by dianh on 2021/04/16.
//

#ifndef SWIN_TRANSFORMER_CPP_PRE_NORM_H
#define SWIN_TRANSFORMER_CPP_PRE_NORM_H

#include "layer.h"
#include "layer_norm.h"

namespace shift_window_transformer {
    template<typename T>
    class PreNorm : virtual public Layer<T> {
    public:
        PreNorm(Layer <T> *fn, int dim) : fn(fn), layerNorm(dim) {
        }

        long long parameterCount() {
            long long ret = 0;
            if (fn != nullptr) {
                ret += fn->parameterCount();
            }
            ret += layerNorm.parameterCount();
            return ret;
        }

        void forward(const Tensor <T> &input, Tensor <T> &output) {
            Tensor<T> tmp{};
            layerNorm.forward(input, tmp);
            fn->forward(tmp, output);
        }

    private:
        Layer <T> *fn = nullptr;
        LayerNorm <T> layerNorm;
    };
}
#endif //SWIN_TRANSFORMER_CPP_PRE_NORM_H
