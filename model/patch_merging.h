//
// Created by dianh on 2021/04/20.
//

#ifndef SWIN_TRANSFORMER_CPP_PATCH_MERGING_H
#define SWIN_TRANSFORMER_CPP_PATCH_MERGING_H

#include "layer.h"
#include "tensor.h"
#include "linear.h"

namespace shift_window_transformer {
    template<typename T>
    class PatchMerging : public Layer<T> {
    public:
        PatchMerging(int in_channels, int out_channels, int downscaling_factor) : layer(
                in_channels * downscaling_factor * downscaling_factor, out_channels) {
            this->in_channels = in_channels;
            this->out_channels = out_channels;
            this->downscaling_factor = downscaling_factor;
            //layer = Linear<T>(in_channels * downscaling_factor * downscaling_factor, out_channels);
        }

        /**
         *
         * @param input C, H, W
         * @param output H, W, C
         */
        void forward(const Tensor<T> &input, Tensor<T> &output) {
            auto &vec = input.shape;
            int ch_sum = vec[0], h = vec[1], w = vec[2];
            int new_h = h / downscaling_factor, new_w = w / downscaling_factor;
            Tensor<T> tmp{};
            tmp.resize(input.size());
            tmp.shape = {new_h, new_w,
                         (vec[0] * downscaling_factor * downscaling_factor)};
            /*
             * Unfold and reshape
             */
            for (int ch = 0; ch < ch_sum; ++ch) {
                for (int i = 0; i < new_h; ++i) {
                    for (int j = 0; j < new_w; ++j) {
                        for (int x = 0; x < downscaling_factor; ++x) {
                            for (int y = 0; y < downscaling_factor; ++y) {
                                int new_x = ch * downscaling_factor * downscaling_factor + x * downscaling_factor + y;
                                int new_y = i * new_w + j;
                                int tmp_pos = new_x * new_w * new_h + new_y;
                                // Input: C, H, W
                                // Tmp: H*W/(downscaling_size^2), C * downscaling_size^2
                                tmp[i * ch_sum * downscaling_factor * downscaling_factor * new_w + j * ch_sum +
                                    downscaling_factor * downscaling_factor +
                                    ch * downscaling_factor * downscaling_factor + x * downscaling_factor + y] = input[
                                        ch * h * w + (i * downscaling_factor + x) * w + j * downscaling_factor + y];
                            }
                        }
                    }
                }
            }
            layer.forward(tmp, output);
        }

        long long parameterCount() {
            return layer.parameterCount();
        }

    private:
        int in_channels;
        int out_channels;
        int downscaling_factor;
        Linear<T> layer;
    };
}
#endif //SWIN_TRANSFORMER_CPP_PATCH_MERGING_H
