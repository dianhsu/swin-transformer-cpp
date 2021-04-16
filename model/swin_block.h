//
// Created by dianh on 2021/04/16.
//

#ifndef SWIN_TRANSFORMER_CPP_SWIN_BLOCK_H
#define SWIN_TRANSFORMER_CPP_SWIN_BLOCK_H

#include "layer.h"

namespace shift_window_transformer {
    template<typename T>
    class SwinBlock : virtual public Layer {
    public:
        SwinBlock(int dim, int heads, int head_dim, int mlp_dim, bool shifted, int window_size,
                  bool relative_pos_embedding) : dim(dim), heads(heads), head_dim(head_dim), mlp_dim(mlp_dim),
                                                 shifted(shifted), window_size(window_size),
                                                 relative_pos_embedding(relative_pos_embedding) {}

    private:
        int dim;
        int heads;
        int head_dim;
        int mlp_dim;
        bool shifted;
        int window_size;
        bool relative_pos_embedding;
    };
}
#endif //SWIN_TRANSFORMER_CPP_SWIN_BLOCK_H
