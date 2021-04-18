//
// Created by dianh on 2021/04/16.
//

#ifndef SWIN_TRANSFORMER_CPP_PRE_NORM_H
#define SWIN_TRANSFORMER_CPP_PRE_NORM_H

#include "layer.h"
#include "layer_norm.h"

namespace shift_window_transformer {
    template<typename T>
    class PreNorm : virtual public Layer {
    public:
        PreNorm(Layer<T> *fn, int dim) : fn(fn) {
            layerNorm = LayerNorm<T>(dim);
        }

        void forward(const Tensor<T> &input, Tensor<T> &output) {
            Tensor<T> tmp{};
            layerNorm.forward(input, tmp);
            fn->forward(tmp, output);
        }

    private:
        Layer<T> *fn = nullptr;
        LayerNorm<T> layerNorm;
    };
}
#endif //SWIN_TRANSFORMER_CPP_PRE_NORM_H
