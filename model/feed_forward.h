//
// Created by dianh on 2021/04/16.
//

#ifndef SWIN_TRANSFORMER_CPP_FEED_FORWARD_H
#define SWIN_TRANSFORMER_CPP_FEED_FORWARD_H

#include "layer.h"
#include "tensor.h"
#include "linear.h"
#include "functions.h"

namespace shift_window_transformer {
    template<typename T>
    class FeedForward : virtual public Layer<T> {
    public:
        FeedForward(int dim, int hidden_dim) : dim(dim), hidden_dim(hidden_dim) {
            linear1 = new Linear<T>(dim, hidden_dim);
            linear2 = new Linear<T>(hidden_dim, dim);
        }

        ~FeedForward() {
            if (linear1 != nullptr) {
                delete linear1;
                linear1 = nullptr;
            }
            if (linear2 != nullptr) {
                delete linear2;
                linear2 = nullptr;
            }
        }

        void forward(const Tensor <T> &input, Tensor <T> &output) {
            Tensor<T> tmp1{};
            linear1->forward(input, tmp1);
            Tensor<T> tmp2{};
            tmp2.shape.insert(tmp2.shape.end(), tmp1.shape.begin(), tmp1.shape.end());
            for (auto item: tmp1) {
                tmp2.push_back(GELU(item));
            }
            linear2->forward(tmp2, output);
        }

        long long parameterCount() {
            return linear1->parameterCount() + linear2->parameterCount();
        }

    private:
        Linear <T> *linear1 = nullptr;
        Linear <T> *linear2 = nullptr;
        int dim;
        int hidden_dim;
    };
}
#endif //SWIN_TRANSFORMER_CPP_FEED_FORWARD_H
