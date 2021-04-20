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

        ~SwinBlock() {
            if (windowAttention) delete windowAttention;
            if (feedForward) delete feedForward;
            if (preNorm1) delete preNorm1;
            if (preNorm2) delete preNorm2;
            if (residual1) delete residual1;
            if (residual2) delete residual2;
        }

        void forward(const Tensor<T> &input, Tensor<T> &output) {
            Tensor<T> tmp{};
            residual1->forward(input, tmp);
            residual2->forward(tmp, output);
        }

    private:
        Residual<T> *residual1, *residual2;
        PreNorm<T> *preNorm1, *preNorm2;
        WindowAttention<T> *windowAttention;
        FeedForward<T> *feedForward;
    };
}
#endif //SWIN_TRANSFORMER_CPP_SWIN_BLOCK_H
