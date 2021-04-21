//
// Created by dianh on 2021/04/16.
//

#ifndef SWIN_TRANSFORMER_CPP_STAGE_MODULE_H
#define SWIN_TRANSFORMER_CPP_STAGE_MODULE_H

#include <vector>
#include "tensor.h"
#include "layer.h"
#include "patch_merging.h"
#include "swin_block.h"

namespace shift_window_transformer {
    template<typename T>
    class StageModule : public Layer<T> {
    public:
        StageModule(int in_channels, int hidden_dimension, int layers_cnt, int downscaling_factor, int num_heads,
                    int head_dim, int window_size, bool relative_pos_embedding) : patch_partition(in_channels,
                                                                                                  hidden_dimension,
                                                                                                  downscaling_factor) {
            assert(layers_cnt % 2 == 0);

            for (int i = 0; i < layers_cnt; i += 2) {
                auto ptr1 = new SwinBlock<T>(hidden_dimension, num_heads, head_dim, hidden_dimension * 4, false,
                                             window_size,
                                             relative_pos_embedding);
                auto ptr2 = new SwinBlock<T>(hidden_dimension, num_heads, head_dim, hidden_dimension * 4, true,
                                             window_size,
                                             relative_pos_embedding);
                layers.push_back(ptr1);
                layers.push_back(ptr2);
            }
        }

        ~StageModule() {
            for (int i = 0; i < layers.size(); ++i) {
                delete layers[i];
            }
        }

        void forward(const Tensor<T> &input, Tensor<T> &output) {
            Tensor<T> tmp{};
            patch_partition.forward(input, tmp);
            for (auto blockPtr: layers) {
                Tensor<T> tmp_loop{};
                blockPtr->forward(tmp, tmp_loop);
                tmp = tmp_loop;
            }
            output.shape = {tmp.shape[2], tmp.shape[0], tmp.shape[1]};
            output.resize(tmp.size());
            for (int i = 0; i < tmp.shape[0]; ++i) {
                for (int j = 0; j < tmp.shape[1]; ++j) {
                    for (int k = 0; k < tmp.shape[2]; ++k) {
                        output[k * tmp.shape[0] * tmp.shape[1] + i * tmp.shape[1] + j] = tmp[
                                i * tmp.shape[1] * tmp.shape[2] + j * tmp.shape[2] + k];
                    }
                }
            }
        }

        long long parameterCount() {
            long long ret = patch_partition.parameterCount();
            for (int i = 0; i < layers.size(); ++i) {
                ret += layers[i]->parameterCount();
            }
            return ret;
        }

    private:
        PatchMerging<T> patch_partition;
        std::vector<SwinBlock<T> *> layers;

    };
}


#endif //SWIN_TRANSFORMER_CPP_STAGE_MODULE_H
