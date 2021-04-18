//
// Created by dianh on 2021/04/16.
//

#ifndef SWIN_TRANSFORMER_CPP_CYCLIC_SHIFT_H
#define SWIN_TRANSFORMER_CPP_CYCLIC_SHIFT_H

#include "tensor.h"
#include "layer.h"

namespace shift_window_transformer {
    template<typename T>
    class CyclicShift : virtual public Layer<T> {
    public:
        CyclicShift(const std::array<int, 3> &dims, int displacement) : displacement(displacement), dim(dims) {

        }

        void forward(const Tensor<T> &input, Tensor<T> &output) {
            int feature_size = dim[1] * dim[2];
            output.shape.clear();
            output.shape.insert(output.shape.begin(), input.shape.begin(), input.shape.end());
            for (int feature = 0; feature < input.size(); feature += feature_size) {
                std::vector<T> tmp(feature_size);
                for (int i = 0; i < dim[1]; ++i) {
                    for (int j = 0; j < dim[2]; ++j) {
                        int nex_i = (i + displacement + dim[1]) % dim[1];
                        int nex_j = (j + displacement + dim[2]) % dim[2];
                        tmp[nex_i * dim[2] + nex_j] = input[feature + i * dim[2] + j];
                    }
                }
                output.insert(output.end(), tmp.begin(), tmp.end());
            }
        }

    private:
        int displacement;
        std::array<int, 3> dim;
    };
}
#endif //SWIN_TRANSFORMER_CPP_CYCLIC_SHIFT_H
