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
                  scale(1 / sqrt(headDim)),
                  innerDim(heads * headDim),
                  windowSize(windowSize),
                  relativePosEmbedding(relativePosEmbedding), to_qkv(dim, heads * headDim * 3, false),
                  to_out(heads * headDim, dim) {
            if (shifted) {
                int displacement = windowSize / 2;
                cyclicShift = new CyclicShift<T>(-displacement);
                cyclicBackShift = new CyclicShift<T>(displacement);
                upperLowerMask = create_mask<T>(windowSize, displacement, true, false);
                lowerRightMask = create_mask<T>(windowSize, displacement, false, true);
            }
            std::random_device rd{};
            std::mt19937 gen{rd()};
            std::normal_distribution<> d{0, 1};
            if (relativePosEmbedding) {
                relativeIndices = get_relative_distances<T>(windowSize);
                for (auto &item: *relativeIndices) {
                    item = item + windowSize - 1;
                }
                posEmbedding = new Tensor<T>();
                int cnt = (2 * windowSize - 1) * (2 * windowSize - 1);
                while (cnt-- > 0) {
                    posEmbedding->push_back((T) d(gen));
                }
                posEmbedding->shape.push_back(2 * windowSize - 1);
                posEmbedding->shape.push_back(2 * windowSize - 1);
            } else {
                posEmbedding = new Tensor<T>();
                int cnt = windowSize * windowSize * windowSize * windowSize;
                while (cnt-- > 0) {
                    posEmbedding->push_back((T) d(gen));
                }
                posEmbedding->shape.push_back(windowSize * windowSize);
                posEmbedding->shape.push_back(windowSize * windowSize);
            }
        }

        void forward(const Tensor<T> &input, Tensor<T> &output) {

            output.resize(input.size());
            output.shape = input.shape;

            Tensor<T> tmp{};
            if (shifted) {
                cyclicShift->forward(input, tmp);
            } else {
                tmp.insert(tmp.end(), input.begin(), input.end());
                tmp.shape.insert(tmp.shape.end(), input.shape.begin(), input.shape.end());
            }
            int n_h = tmp.shape[1];
            int n_w = tmp.shape[2];
            int h = heads;
            Tensor<T> qkv{};
            to_qkv.forward(tmp, qkv);
            // TODO: TO FINISH THIS MODULE
        }

        ~WindowAttention() {
            if (upperLowerMask != nullptr) {
                delete upperLowerMask;
                upperLowerMask = nullptr;
            }


            if (lowerRightMask != nullptr) {
                delete lowerRightMask;
                lowerRightMask = nullptr;
            }
            if (relativeIndices != nullptr) {
                delete relativeIndices;
                relativeIndices = nullptr;
            }
            if (posEmbedding != nullptr) {
                delete posEmbedding;
                posEmbedding = nullptr;
            }
            if (cyclicShift != nullptr) {
                delete cyclicShift;
                cyclicShift = nullptr;
            }
            if (cyclicBackShift != nullptr) {
                delete cyclicBackShift;
                cyclicBackShift = nullptr;
            }
        }

    private:
        CyclicShift<T> *cyclicShift = nullptr;
        CyclicShift<T> *cyclicBackShift = nullptr;
        Linear<T> to_qkv;
        Linear<T> to_out;
        Tensor<T> *upperLowerMask = nullptr;
        Tensor<T> *lowerRightMask = nullptr;
        Tensor<T> *relativeIndices = nullptr;
        Tensor<T> *posEmbedding = nullptr;
        int windowSize;
        bool shifted;
        int innerDim;
        int heads;
        T scale;
        bool relativePosEmbedding;
    };
}
#endif //SWIN_TRANSFORMER_CPP_WINDOW_ATTENTION_H
