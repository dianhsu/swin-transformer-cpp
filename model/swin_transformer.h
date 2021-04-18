//
// Created by dianh on 2021/04/16.
//

#ifndef SWIN_TRANSFORMER_CPP_SWIN_TRANSFORMER_H
#define SWIN_TRANSFORMER_CPP_SWIN_TRANSFORMER_H

#include <bits/stdc++.h>

#include "layer.h"

namespace shift_window_transformer {
    template<typename T>
    class SwinTransformer : virtual public Layer<T> {
    public:
        SwinTransformer(int hiddenDim, const std::array<int, 4> &layers, const std::array<int, 4> &heads) : layers(
                layers), heads(heads), hidden_dim(hiddenDim), downscaling_factors(std::array<int, 4>{4, 2, 2, 2}) {
            head_dim = 32;
            window_size = 7;
            input_channels = 3;
            num_classes = 1000;
            relative_pos_embedding = true;
        }

        SwinTransformer(const std::array<int, 4> &layers, const std::array<int, 4> &heads, int hiddenDim,
                        int inputChannels,
                        int numClasses, int headDim, int windowSize, const std::array<int, 4> &downscalingFactors,
                        bool relativePosEmbedding) : layers(layers), heads(heads), hidden_dim(hiddenDim),
                                                     input_channels(inputChannels), num_classes(numClasses),
                                                     head_dim(headDim), window_size(windowSize),
                                                     downscaling_factors(downscalingFactors),
                                                     relative_pos_embedding(relativePosEmbedding) {}

        void forward(const Tensor<T> &input, Tensor<T> &output) {

        }

    private:
        std::array<int, 4> layers;
        std::array<int, 4> heads;
        int hidden_dim;
        int input_channels;
        int num_classes;
        int head_dim;
        int window_size;
        std::array<int, 4> downscaling_factors;
        bool relative_pos_embedding;
    };
}

#endif //SWIN_TRANSFORMER_CPP_SWIN_TRANSFORMER_H
