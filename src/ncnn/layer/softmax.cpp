// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "softmax.h"
#include <float.h>
#include <math.h>
#include <algorithm>

namespace ncnn {

    DEFINE_LAYER_CREATOR(Softmax)

    Softmax::Softmax() {
        one_blob_only = true;
        support_inplace = true;
    }

    int Softmax::load_param(const ParamDict &pd) {
        axis = pd.get(0, 0);

        return 0;
    }

    int Softmax::forward_inplace(Mat &bottom_top_blob) const {
        // value = exp( value - global max value )
        // sum all value
        // value = value / sum

        int dims = bottom_top_blob.dims;

        if (dims == 1) // axis == 0
        {
            int w = bottom_top_blob.w;

            float *ptr = bottom_top_blob;

            float max = -FLT_MAX;
            for (int i = 0; i < w; i++) {
                max = std::max(max, ptr[i]);
            }

            for (int i = 0; i < w; i++) {
                ptr[i] = expf(ptr[i] - max);
            }

            float sum = 0.f;
            for (int i = 0; i < w; i++) {
                sum += ptr[i];
            }

            for (int i = 0; i < w; i++) {
                ptr[i] /= sum;
            }

            return 0;
        }

        if (dims == 2 && axis == 0) {
            int w = bottom_top_blob.w;
            int h = bottom_top_blob.h;

            Mat max_mat;
            max_mat.create(w);
            if (max_mat.empty())
                return -100;
            max_mat.fill(-FLT_MAX);

            for (int i = 0; i < h; i++) {
                const float *ptr = bottom_top_blob.row(i);
                for (int j = 0; j < w; j++) {
                    max_mat[j] = std::max(max_mat[j], ptr[j]);
                }
            }

            for (int i = 0; i < h; i++) {
                float *ptr = bottom_top_blob.row(i);
                for (int j = 0; j < w; j++) {
                    ptr[j] = expf(ptr[j] - max_mat[j]);
                }
            }

            Mat sum_mat;
            sum_mat.create(w);
            if (sum_mat.empty())
                return -100;
            sum_mat.fill(0.f);

            for (int i = 0; i < h; i++) {
                const float *ptr = bottom_top_blob.row(i);
                for (int j = 0; j < w; j++) {
                    sum_mat[j] += ptr[j];
                }
            }

            for (int i = 0; i < h; i++) {
                float *ptr = bottom_top_blob.row(i);
                for (int j = 0; j < w; j++) {
                    ptr[j] /= sum_mat[j];
                }
            }

            return 0;
        }

        if (dims == 2 && axis == 1) {
            int w = bottom_top_blob.w;
            int h = bottom_top_blob.h;

            Mat max_mat;
            max_mat.create(h);
            if (max_mat.empty())
                return -100;

            for (int i = 0; i < h; i++) {
                const float *ptr = bottom_top_blob.row(i);

                float m = -FLT_MAX;
                for (int j = 0; j < w; j++) {
                    m = std::max(m, ptr[j]);
                }

                max_mat[i] = m;
            }

            for (int i = 0; i < h; i++) {
                float *ptr = bottom_top_blob.row(i);

                float m = max_mat[i];
                for (int j = 0; j < w; j++) {
                    ptr[j] = expf(ptr[j] - m);
                }
            }

            Mat sum_mat;
            sum_mat.create(h);
            if (sum_mat.empty())
                return -100;

            for (int i = 0; i < h; i++) {
                const float *ptr = bottom_top_blob.row(i);

                float s = 0.f;
                for (int j = 0; j < w; j++) {
                    s += ptr[j];
                }

                sum_mat[i] = s;
            }

            for (int i = 0; i < h; i++) {
                float *ptr = bottom_top_blob.row(i);

                float s = sum_mat[i];
                for (int j = 0; j < w; j++) {
                    ptr[j] /= s;
                }
            }

            return 0;
        }

        if (dims == 3 && axis == 0) {
            int w = bottom_top_blob.w;
            int h = bottom_top_blob.h;
            int channels = bottom_top_blob.c;
            int size = w * h;

            Mat max_mat;
            max_mat.create(w, h);
            if (max_mat.empty())
                return -100;
            max_mat.fill(-FLT_MAX);
            for (int q = 0; q < channels; q++) {
                const float *ptr = bottom_top_blob.channel(q);

                for (int i = 0; i < size; i++) {
                    max_mat[i] = std::max(max_mat[i], ptr[i]);
                }
            }

#pragma omp parallel for
            for (int q = 0; q < channels; q++) {
                float *ptr = bottom_top_blob.channel(q);

                for (int i = 0; i < size; i++) {
                    ptr[i] = exp(ptr[i] - max_mat[i]);
                }
            }

            Mat sum_mat;
            sum_mat.create(w, h);
            if (sum_mat.empty())
                return -100;
            sum_mat.fill(0.f);
            for (int q = 0; q < channels; q++) {
                const float *ptr = bottom_top_blob.channel(q);

                for (int i = 0; i < size; i++) {
                    sum_mat[i] += ptr[i];
                }
            }

#pragma omp parallel for
            for (int q = 0; q < channels; q++) {
                float *ptr = bottom_top_blob.channel(q);

                for (int i = 0; i < size; i++) {
                    ptr[i] /= sum_mat[i];
                }
            }

            return 0;
        }

        if (dims == 3 && axis == 1) {
            int w = bottom_top_blob.w;
            int h = bottom_top_blob.h;
            int channels = bottom_top_blob.c;

            Mat max_mat;
            max_mat.create(h, channels);
            if (max_mat.empty())
                return -100;
            max_mat.fill(-FLT_MAX);
#pragma omp parallel for
            for (int q = 0; q < channels; q++) {
                const float *ptr = bottom_top_blob.channel(q);
                float *maxptr = max_mat.row(q);

                for (int i = 0; i < h; i++) {
                    float cur_max = -FLT_MAX;
                    for (int j = 0; j < w; j++) {
                        cur_max = std::max(cur_max, ptr[j]);
                    }

                    maxptr[i] = cur_max;
                    ptr += w;
                }
            }

#pragma omp parallel for
            for (int q = 0; q < channels; q++) {
                float *ptr = bottom_top_blob.channel(q);
                float *maxptr = max_mat.row(q);

                for (int i = 0; i < h; i++) {
                    float cur_max = maxptr[i];
                    for (int j = 0; j < w; j++) {
                        ptr[j] = expf(ptr[j] - cur_max);
                    }

                    ptr += w;
                }
            }

            Mat sum_mat;
            sum_mat.create(h, channels);
            if (sum_mat.empty())
                return -100;
            sum_mat.fill(0.f);
#pragma omp parallel for
            for (int q = 0; q < channels; q++) {
                const float *ptr = bottom_top_blob.channel(q);
                float *sumptr = sum_mat.row(q);

                for (int i = 0; i < h; i++) {
                    float cur_sum = 0.f;
                    for (int j = 0; j < w; j++) {
                        cur_sum += ptr[j];
                    }

                    sumptr[i] = cur_sum;
                    ptr += w;
                }
            }

#pragma omp parallel for
            for (int q = 0; q < channels; q++) {
                float *ptr = bottom_top_blob.channel(q);
                float *sumptr = sum_mat.row(q);

                for (int i = 0; i < h; i++) {
                    float inv_cur_sum = 1.0f / sumptr[i];
                    for (int j = 0; j < w; j++) {
                        ptr[j] *= inv_cur_sum;
                    }

                    ptr += w;
                }
            }

            return 0;
        }

        if (dims == 3 && axis == 2) {
            int w = bottom_top_blob.w;
            int h = bottom_top_blob.h;
            int channels = bottom_top_blob.c;

            Mat max_mat;
            max_mat.create(w, channels);
            if (max_mat.empty())
                return -100;
            max_mat.fill(-FLT_MAX);
#pragma omp parallel for
            for (int q = 0; q < channels; q++) {
                const float *ptr = bottom_top_blob.channel(q);
                float *maxptr = max_mat.row(q);

                for (int i = 0; i < h; i++) {
                    for (int j = 0; j < w; j++) {
                        maxptr[j] = std::max(maxptr[j], ptr[j]);
                    }

                    ptr += w;
                }
            }

#pragma omp parallel for
            for (int q = 0; q < channels; q++) {
                float *ptr = bottom_top_blob.channel(q);
                float *maxptr = max_mat.row(q);

                for (int i = 0; i < h; i++) {
                    for (int j = 0; j < w; j++) {
                        ptr[j] = exp(ptr[j] - maxptr[j]);
                    }

                    ptr += w;
                }
            }

            Mat sum_mat;
            sum_mat.create(w, channels);
            if (sum_mat.empty())
                return -100;
            sum_mat.fill(0.f);
#pragma omp parallel for
            for (int q = 0; q < channels; q++) {
                const float *ptr = bottom_top_blob.channel(q);
                float *sumptr = sum_mat.row(q);

                for (int i = 0; i < h; i++) {
                    for (int j = 0; j < w; j++) {
                        sumptr[j] += ptr[j];
                    }

                    ptr += w;
                }
            }

#pragma omp parallel for
            for (int q = 0; q < channels; q++) {
                float *ptr = bottom_top_blob.channel(q);
                float *sumptr = sum_mat.row(q);

                for (int i = 0; i < h; i++) {
                    for (int j = 0; j < w; j++) {
                        ptr[j] /= sumptr[j];
                    }

                    ptr += w;
                }
            }
            return 0;
        }
        return 0;
    }

} // namespace ncnn
