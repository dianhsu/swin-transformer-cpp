//
// Created by dianh on 2021/04/16.
//

#ifndef SWIN_TRANSFORMER_CPP_PRE_NORM_H
#define SWIN_TRANSFORMER_CPP_PRE_NORM_H

#include "layer.h"

namespace shift_window_transformer {
    template<typename T>
    class PreNorm : virtual public Layer {
    public:
        void forward(const Tensor<T> &input, Tensor<T> &output){

        }
    };
}
#endif //SWIN_TRANSFORMER_CPP_PRE_NORM_H
