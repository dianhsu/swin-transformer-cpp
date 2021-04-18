//
// Created by dianh on 2021/04/16.
//

#ifndef SWIN_TRANSFORMER_CPP_WINDOW_ATTENTION_H
#define SWIN_TRANSFORMER_CPP_WINDOW_ATTENTION_H

#include <cmath>
#include "layer.h"
#include "tensor.h"
#include "cyclic_shift.h"
#include "functions.h"
#include "linear.h"

namespace shift_window_transformer {
    template<typename T>
    class WindowAttention : virtual public Layer<T> {
    public:
        WindowAttention(int dim, int heads, int headDim, bool shifted, int windowSize, bool relativePosEmbedding)
                : shifted(shifted),
                  scale(1 / sqrt(head_dim)),
                  innerDim(heads * headDim),
                  windowSize(windowSize),
                  relative_pos_embedding(relativePosEmbedding) {
            if (shifted) {
                int displacement = window_size / 2;
                cyclicShift = CyclicShift<T>(-displacement);
                cyclicBackShift = CyclicShift<T>(displacement);

            }
        }

    private:
        CyclicShift<T> cyclicShift;
        CyclicShift<T> cyclicBackShift;
        int windowSize;
        bool shifted;
        int innerDim;
        int heads;
        T scale;
        bool relativePosEmbedding;
    };
}
#endif //SWIN_TRANSFORMER_CPP_WINDOW_ATTENTION_H
