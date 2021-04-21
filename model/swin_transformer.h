//
// Created by dianh on 2021/04/16.
//

#ifndef SWIN_TRANSFORMER_CPP_SWIN_TRANSFORMER_H
#define SWIN_TRANSFORMER_CPP_SWIN_TRANSFORMER_H

#include <bits/stdc++.h>

#include "layer.h"
#include "stage_module.h"
#include "layer_norm.h"
#include "linear.h"

namespace shift_window_transformer {
    template<typename T>
    class SwinTransformer : virtual public Layer<T> {
    public:
        SwinTransformer(const std::array<int, 4> &layers, const std::array<int, 4> &heads, int hiddenDim,
                        int inputChannels,
                        int numClasses, int headDim, int windowSize, const std::array<int, 4> &downscalingFactors,
                        bool relativePosEmbedding) :
                stage1(inputChannels, hiddenDim, layers[0], downscalingFactors[0], heads[0], headDim,
                       windowSize, relativePosEmbedding),
                stage2(hiddenDim, hiddenDim * 2, layers[1], downscalingFactors[1], heads[1], headDim,
                       windowSize, relativePosEmbedding),
                stage3(hiddenDim * 2, hiddenDim * 4, layers[2], downscalingFactors[2], heads[2], headDim,
                       windowSize, relativePosEmbedding),
                stage4(hiddenDim * 4, hiddenDim * 8, layers[3], downscalingFactors[3], heads[3], headDim,
                       windowSize, relativePosEmbedding),
                layerNorm(hiddenDim * 8),
                linear(hiddenDim * 8, numClasses) {
        }

        SwinTransformer(int hiddenDim, const std::array<int, 4> &layers, const std::array<int, 4> &heads)
                : SwinTransformer(layers, heads, hiddenDim, 3, 1000, 32, 7, std::array<int, 4>{4, 2, 2, 2}, true) {
        }


        void forward(const Tensor<T> &input, Tensor<T> &output) {
            Tensor<T> tmp1{};
            Tensor<T> tmp2{};
            Tensor<T> tmp3{};
            Tensor<T> tmp4{};
            Tensor<T> tmp5{};
            Tensor<T> tmp6{};
            stage1.forward(input, tmp1);
            stage2.forward(tmp1, tmp2);
            stage3.forward(tmp2, tmp3);
            stage4.forward(tmp3, tmp4);
            tmp5.shape = {tmp4.shape[0]};
            int batch = tmp4.shape[1] * tmp4.shape[2];
            for (int i = 0; i < tmp4.size(); i += batch) {
                T sum = 0;
                for (int j = 0; j < batch; ++j) {
                    sum += tmp4[i + j];
                }
                tmp5.push_back(sum / batch);
            }
            layerNorm.forward(tmp5, tmp6);
            linear.forward(tmp6, output);
        }

        long long parameterCount() {
            return stage1.parameterCount() + stage2.parameterCount() + stage3.parameterCount() +
                   stage4.parameterCount() + layerNorm.parameterCount() + linear.parameterCount();
        }

    private:
        StageModule<T> stage1;
        StageModule<T> stage2;
        StageModule<T> stage3;
        StageModule<T> stage4;
        LayerNorm<T> layerNorm;
        Linear<T> linear;
    };
}

#endif //SWIN_TRANSFORMER_CPP_SWIN_TRANSFORMER_H
