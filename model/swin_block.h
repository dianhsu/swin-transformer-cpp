//
// Created by dianh on 2021/04/16.
//

#ifndef SWIN_TRANSFORMER_CPP_SWIN_BLOCK_H
#define SWIN_TRANSFORMER_CPP_SWIN_BLOCK_H

#include "layer.h"
#include "residual.h"
#include "pre_norm.h"
#include "window_attention.h"
#include "feed_forward.h"

namespace shift_window_transformer {
    template<typename T>
    class SwinBlock : virtual public Layer<T> {
    public:
        SwinBlock(int dim, int heads, int headDim, int mlpDim, bool shifted, int windowSize,
                  bool relativePosEmbedding) {
            windowAttention = new WindowAttention<T>(dim, heads, headDim, shifted, windowSize, relativePosEmbedding);
            feedForward = new FeedForward<T>(dim, mlpDim);
            preNorm1 = new PreNorm<T>(windowAttention, dim);
            preNorm2 = new PreNorm<T>(feedForward, dim);
            residual1 = new Residual<T>(preNorm1);
            residual2 = new Residual<T>(preNorm2);
        }

        long long parameterCount() {
            long long ret = 0;
            if (windowAttention) {
                ret += windowAttention->parameterCount();
            }
            if (feedForward) {
                ret += feedForward->parameterCount();
            }
            if (preNorm1) {
                ret += preNorm1->parameterCount();
            }
            if (preNorm2) {
                ret += preNorm2->parameterCount();
            }
            if (residual1) {
                ret += residual1->parameterCount();
            }
            if (residual2) {
                ret += residual2->parameterCount();
            }
            return ret;
        }

        ~SwinBlock() {
            if (windowAttention != nullptr) {
                delete windowAttention;
                windowAttention = nullptr;
            }
            if (feedForward != nullptr) {
                delete feedForward;
                feedForward = nullptr;
            }
            if (preNorm1 != nullptr) {
                delete preNorm1;
                preNorm1 = nullptr;
            }
            if (preNorm2 != nullptr) {
                delete preNorm2;
                preNorm2 = nullptr;
            }
            if (residual1 != nullptr) {
                delete residual1;
                residual1 = nullptr;
            }
            if (residual2 != nullptr) {
                delete residual2;
                residual2 = nullptr;
            }
        }

        void forward(const Tensor<T> &input, Tensor<T> &output) {
            Tensor<T> tmp{};
            residual1->forward(input, tmp);
            residual2->forward(tmp, output);
        }

    private:
        Residual<T> *residual1 = nullptr, *residual2 = nullptr;
        PreNorm<T> *preNorm1 = nullptr, *preNorm2 = nullptr;
        WindowAttention<T> *windowAttention = nullptr;
        FeedForward<T> *feedForward = nullptr;
    };
}
#endif //SWIN_TRANSFORMER_CPP_SWIN_BLOCK_H
