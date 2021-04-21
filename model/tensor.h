//
// Created by dianh on 2021/04/16.
//

#ifndef SWIN_TRANSFORMER_CPP_TENSOR_H
#define SWIN_TRANSFORMER_CPP_TENSOR_H

#include <bits/stdc++.h>

namespace shift_window_transformer {
    typedef std::vector<int> TensorShape;

    template<typename T>
    class Tensor : public std::vector<T> {
    public:
        Tensor() : std::vector<T>() {
        }

        Tensor(int size, T default_data) : std::vector<T>(size, default_data) {
            shape.clear();
            shape.push_back(size);
        }

        TensorShape shape;

        friend std::ostream &operator<<(std::ostream &os, const Tensor<T> &vec) {
            int cnt = 1;
            os << "[" << vec.size() << "]";
            os << "[";
            for (auto i = 0; i < vec.shape.size(); ++i) {
                if (i)
                    os << ", ";
                os << vec.shape[i];
                cnt *= vec.shape[i];
            }
            os << "]";
            assert(cnt == vec.size());
            return os;
        }
    };
}

#endif //SWIN_TRANSFORMER_CPP_TENSOR_H
