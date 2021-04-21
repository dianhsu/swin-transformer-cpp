//
// Created by dianh on 2021/04/16.
//

#ifndef SWIN_TRANSFORMER_CPP_LAYER_H
#define SWIN_TRANSFORMER_CPP_LAYER_H

#include <bits/stdc++.h>
#include "tensor.h"

namespace shift_window_transformer {
    template<typename T>
    class Layer {
    public:
        virtual void forward(const Tensor<T> &input, Tensor<T> &output) {

        }

        virtual long long parameterCount() {
            return 0;
        }
    };

}
#endif //SWIN_TRANSFORMER_CPP_LAYER_H
