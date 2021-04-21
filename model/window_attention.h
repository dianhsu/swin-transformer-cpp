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
#include "softmax.h"

namespace shift_window_transformer {
    template<typename T>
    class WindowAttention : virtual public Layer<T> {
    public:
        WindowAttention(int dim, int heads, int headDim, bool shifted, int windowSize, bool relativePosEmbedding)
                : shifted(shifted),
                  scale(1 / sqrt(headDim)),
                  innerDim(heads * headDim),
                  heads(heads),
                  headDim(headDim),
                  windowSize(windowSize),
                  relativePosEmbedding(relativePosEmbedding) {
            qLinear = new Linear<T>(dim, heads * headDim, false);
            kLinear = new Linear<T>(dim, heads * headDim, false);
            vLinear = new Linear<T>(dim, heads * headDim, false);
            outLinear = new Linear<T>(heads * headDim, dim);
            if (shifted) {
                int displacement = windowSize / 2;
                cyclicShift = new CyclicShift<T>(-displacement);
                cyclicBackShift = new CyclicShift<T>(displacement);
                upperLowerMask = create_mask<T>(windowSize, displacement, true, false);
                leftRightMask = create_mask<T>(windowSize, displacement, false, true);
            }
            std::random_device rd{};
            std::mt19937 gen{rd()};
            std::normal_distribution<> d{0, 1};
            if (relativePosEmbedding) {
                relativeIndices = get_relative_distances<int>(windowSize);
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

        void reArrange(const Tensor<T> &input, Tensor<T> &output) {
            assert(input.shape.size() == 3);
            int n_h = input.shape[0], n_w = input.shape[1];
            int nw_h = input.shape[0] / windowSize, nw_w = input.shape[1] / windowSize;
            int dim = input.shape[2] / heads;
            output.shape = {heads, nw_h * nw_w, windowSize * windowSize, dim};
            output.resize(input.size());
            int lock_j = input.shape[2];
            int lock_i = lock_j * input.shape[1];
            int key_c = dim;
            int key_b = key_c * windowSize * windowSize;
            int key_a = key_c * input.shape[0] * input.shape[1];
            for (int i = 0; i < n_h; ++i) {
                for (int j = 0; j < n_w; ++j) {
                    for (int k = 0; k < input.shape[2]; ++k) {
                        int a = k / dim;
                        int b = i / windowSize * (input.shape[0] / windowSize) + j / windowSize;
                        int c = i % windowSize * windowSize + j % windowSize;
                        int d = k % dim;
                        output[a * key_a + b * key_b + c * key_c + d] = input[i * lock_i + j * lock_j + k];
                    }
                }
            }
        }


        void forward(const Tensor<T> &input, Tensor<T> &output) {
            Tensor<T> tmp{};
            if (shifted) {
                cyclicShift->forward(input, tmp);
            } else {
                tmp.insert(tmp.end(), input.begin(), input.end());
                tmp.shape.insert(tmp.shape.end(), input.shape.begin(), input.shape.end());
            }
            Tensor<T> qTmp{};
            Tensor<T> kTmp{};
            Tensor<T> vTmp{};
            qLinear->forward(tmp, qTmp);
            kLinear->forward(tmp, kTmp);
            kLinear->forward(tmp, vTmp);
            Tensor<T> q{}, k{}, v{};
            int nw_h = qTmp.shape[0] / windowSize, nw_w = qTmp.shape[1] / windowSize;
            reArrange(qTmp, q);
            reArrange(kTmp, k);
            reArrange(vTmp, v);
            Tensor<T> dots{};
            int matX = q.shape[q.shape.size() - 2];
            int matY = q.shape[q.shape.size() - 1];
            dots.shape = q.shape;
            dots.shape[dots.shape.size() - 1] = dots.shape[dots.shape.size() - 2];
            for (int pos = 0; pos < q.size(); pos += matX * matY) {
                for (int pi = 0; pi < matX; ++pi) {
                    for (int pj = 0; pj < matX; ++pj) {
                        T val = 0;
                        for (int pk = 0; pk < matY; ++pk) {
                            val += q[pos + pi * matY + pk] * k[pos + pj * matY + pk];
                        }
                        dots.push_back(val);
                    }
                }
            }

            if (relativePosEmbedding) {
                assert(posEmbedding->size() == (2 * windowSize - 1) * (2 * windowSize - 1));
                for (int pos = 0; pos < dots.size(); pos += matX * matX) {
                    for (int i = 0; i < matX; ++i) {
                        for (int j = 0; j < matX; ++j) {
                            dots[pos + i * matX + j] += (*posEmbedding)[
                                    (*relativeIndices)[i * matX * 2 + j * 2] * (2 * windowSize - 1) +
                                    (*relativeIndices)[i * matX * 2 + j * 2 + 1]];
                        }
                    }
                }
            } else {
                assert(posEmbedding->size() == matX * matX);
                for (int pos = 0; pos < dots.size(); pos += matX * matX) {
                    for (int i = 0; i < matX; ++i) {
                        for (int j = 0; j < matX; ++j) {
                            dots[pos + i * matX + j] += (*posEmbedding)[i * matX + j];
                        }
                    }
                }
            }

            if (shifted) {
                int batch = dots.size() / dots.shape[0];
                for (int pos = 0; pos < dots.size(); pos += batch) {
                    for (int pi = dots.shape[1] - nw_w; pi < dots.shape[1]; ++pi) {
                        for (int pj = 0; pj < dots.shape[2]; ++pj) {
                            for (int pk = 0; pk < dots.shape[3]; ++pk) {
                                dots[pos + pi * dots.shape[2] * dots.shape[3] + pj * dots.shape[3] +
                                     pk] += (*upperLowerMask)[pj * dots.shape[3] + pk];
                            }
                        }
                    }
                    for (int pi = nw_w - 1; pi < dots.shape[1]; pi += nw_w) {
                        for (int pj = 0; pj < dots.shape[2]; ++pj) {
                            for (int pk = 0; pk < dots.shape[3]; ++pk) {
                                dots[pos + pi * dots.shape[2] * dots.shape[3] + pj * dots.shape[3] +
                                     pk] += (*leftRightMask)[pj * dots.shape[3] + pk];
                            }
                        }
                    }
                }
            }
            Tensor<T> attn{};
            softMax.forward(dots, attn);
            Tensor<T> tmp2{};
            tmp2.shape = v.shape;
            for (int pos = 0; pos < dots.shape[0] * dots.shape[1]; ++pos) {
                for (int pi = 0; pi < dots.shape[2]; ++pi) {
                    for (int pj = 0; pj < v.shape[3]; ++pj) {
                        T val = 0;
                        for (int pk = 0; pk < dots.shape[3]; ++pk) {
                            val += dots[pos * dots.shape[2] * dots.shape[3] + pi * dots.shape[3] + pk] *
                                   v[pos * v.shape[2] * v.shape[3] + pk * v.shape[3] + pj];
                        }
                        tmp2.push_back(val);
                    }
                }
            }
            Tensor<T> tmp3{};
            tmp3.resize(tmp2.size());
            tmp3.shape = qTmp.shape;
            for (int i = 0; i < tmp2.size(); ++i) {
                int d = i % headDim;
                int w_w = i / headDim % windowSize;
                int w_h = i / headDim / windowSize % windowSize;
                int mw_w = i / headDim / windowSize / windowSize % nw_w;
                int mw_h = i / headDim / windowSize / windowSize / nw_w % nw_h;
                int h = i / headDim / windowSize / windowSize / nw_w / nw_h;
                int new_pos = d + h * headDim +
                              w_w * headDim * heads +
                              w_h * windowSize * headDim * heads +
                              mw_w * windowSize * windowSize * headDim * heads +
                              mw_h * nw_w * windowSize * windowSize * headDim * heads;
                tmp3[new_pos] = tmp2[i];
            }
            Tensor<T> tmp4{};
            outLinear->forward(tmp3, tmp4);

            if (shifted) {
                cyclicBackShift->forward(tmp4, output);
            } else {
                output.clear();
                output.shape.clear();
                output.insert(output.end(), tmp4.begin(), tmp4.end());
                output.shape.insert(output.shape.end(), tmp4.shape.begin(), tmp4.shape.end());
            }
        }

        ~WindowAttention() {
            if (qLinear != nullptr) {
                delete qLinear;
                qLinear = nullptr;
            }
            if (kLinear != nullptr) {
                delete kLinear;
                kLinear = nullptr;
            }
            if (vLinear != nullptr) {
                delete vLinear;
                vLinear = nullptr;
            }
            if (outLinear != nullptr) {
                delete outLinear;
                outLinear = nullptr;
            }
            if (upperLowerMask != nullptr) {
                delete upperLowerMask;
                upperLowerMask = nullptr;
            }
            if (leftRightMask != nullptr) {
                delete leftRightMask;
                leftRightMask = nullptr;
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

        long long parameterCount() {
            long long ret = 0;
            if (cyclicShift != nullptr) ret += cyclicShift->parameterCount();
            if (cyclicBackShift != nullptr) ret += cyclicBackShift->parameterCount();
            if (qLinear != nullptr) ret += qLinear->parameterCount();
            if (kLinear != nullptr) ret += kLinear->parameterCount();
            if (vLinear != nullptr) ret += vLinear->parameterCount();
            if (outLinear != nullptr) ret += outLinear->parameterCount();
            if (upperLowerMask != nullptr) ret += upperLowerMask->size();
            if (leftRightMask != nullptr) ret += leftRightMask->size();
            if (relativeIndices != nullptr) ret += relativeIndices->size();
            if (posEmbedding != nullptr) ret += posEmbedding->size();
            ret += softMax.parameterCount();
            return ret;
        }

    private:
        CyclicShift<T> *cyclicShift = nullptr;
        CyclicShift<T> *cyclicBackShift = nullptr;
        Linear<T> *qLinear = nullptr;
        Linear<T> *kLinear = nullptr;
        Linear<T> *vLinear = nullptr;
        Linear<T> *outLinear = nullptr;
        Tensor<T> *upperLowerMask = nullptr;
        Tensor<T> *leftRightMask = nullptr;
        Tensor<int> *relativeIndices = nullptr;
        Tensor<T> *posEmbedding = nullptr;
        SoftMax<T> softMax;
        int windowSize;
        bool shifted;
        int innerDim;
        int headDim;
        int heads;
        T scale;
        bool relativePosEmbedding;
    };
}
#endif //SWIN_TRANSFORMER_CPP_WINDOW_ATTENTION_H
