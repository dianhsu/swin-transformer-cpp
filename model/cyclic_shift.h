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
        explicit CyclicShift(int displacement) : displacement(displacement) {
        }
        long long parameterCount(){
            return 0;
        }
        void forward(const Tensor <T> &input, Tensor <T> &output) {
            assert(input.shape.size() >= 3);
            int feature_size = input.shape[1] * input.shape[2];
            output.shape.clear();
            output.shape.insert(output.shape.begin(), input.shape.begin(), input.shape.end());
            for (int feature = 0; feature < input.size(); feature += feature_size) {
                std::vector<T> tmp(feature_size);
                for (int i = 0; i < input.shape[1]; ++i) {
                    for (int j = 0; j < input.shape[2]; ++j) {
                        int nex_i = (i + displacement + input.shape[1]) % input.shape[1];
                        int nex_j = (j + displacement + input.shape[2]) % input.shape[2];
                        tmp[nex_i * input.shape[2] + nex_j] = input[feature + i * input.shape[2] + j];
                    }
                }
                output.insert(output.end(), tmp.begin(), tmp.end());
            }
        }

    private:
        int displacement;
    };
}
#endif //SWIN_TRANSFORMER_CPP_CYCLIC_SHIFT_H
