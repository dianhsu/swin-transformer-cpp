//
// Created by dianh on 2021/04/16.
//

#ifndef SWIN_TRANSFORMER_CPP_TENSOR_H
#define SWIN_TRANSFORMER_CPP_TENSOR_H

#include <bits/stdc++.h>

namespace shift_window_transformer {
    template<typename T>
    class Tensor {
    public:
        Tensor() {
            data = nullptr;
        }

        Tensor(const std::array<int, 3> &data_size) : data_size(data_size) {
            if (this->data != nullptr) {
                delete[] data;
                this->data = nullptr;
            }
            this->data = new T[data_size[0] * data_size[1] * data_size[2]];
        }

        ~Tensor() {
            if (this->data != nullptr) {
                delete[] data;
            }
        }

        T &operator[](int index) {
            assert(index < data_size[0] * data_size[1] * data_size[2] && this->data != nullptr);
            return data[index];
        }

        T &at(int d, int x, int y) {
            assert(d < data_size[0] && x < data_size[1] && y < data_size[2] && this->data != nullptr);
            return data[d * data_size[1] * data_size[2] + x * data_size[2] + y];
        }
        int size(){
            return data_size[0] * data_size[1] * data_size[2];
        }

    private:
        std::array<int, 3> data_size;
        T *data;
    };
}
#endif //SWIN_TRANSFORMER_CPP_TENSOR_H
