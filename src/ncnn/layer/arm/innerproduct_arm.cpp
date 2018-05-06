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
#if defined(__ARM_NEON)
#include "innerproduct_arm.h"


 


namespace ncnn {

    DEFINE_LAYER_CREATOR(InnerProduct_arm)

    int InnerProduct_arm::forward(const Mat &bottom_blob, Mat &top_blob) const {
        int w = bottom_blob.w;
        int h = bottom_blob.h;
        int channels = bottom_blob.c;
        int size = w * h;

        top_blob.create(1, 1, num_output);
        if (top_blob.empty())
            return -100;

        if (size == 1) {
            // num_output
            const float *weight_data_ptr = weight_data;
#pragma omp parallel for
            for (int p = 0; p < num_output; p++) {
                float *outptr = top_blob.channel(p);
                float sum = 0.f;

                if (bias_term)
                    sum = bias_data.data[p];

                const float *line_w = weight_data_ptr + channels * p;

                // channels
                const float *m = bottom_blob;
                for (int q = 0; q < channels; q++) {
                    sum += *m * *line_w++;
                    m += 4;
                }

                outptr[0] = sum;
            }

            return 0;
        }

        // num_output
        const float *weight_data_ptr = weight_data;
#pragma omp parallel for
        for (int p = 0; p < num_output; p++) {
            float *outptr = top_blob.channel(p);
            float sum = 0.f;

            if (bias_term)
                sum = bias_data.data[p];

            const float *line_w = weight_data_ptr + size * channels * p;
            // const float *w2 = line_w + size;

#if defined(__ARM_NEON)
            float32x4_t _sum = vdupq_n_f32(0.f);
            float32x4_t _sum2 = vdupq_n_f32(0.f);
#endif // __ARM_NEON

            // channels
            for (int q = 0; q < channels; q++) {
                const float *m = bottom_blob.channel(q);

#if defined(__ARM_NEON)
                int nn = size >> 3;
                int remain = size & 7;
#else
                int remain = size;
#endif // __ARM_NEON

#if defined(__ARM_NEON)
#if __aarch64__
                for (; nn>0; nn--)
                {
                    float32x4_t _m = vld1q_f32(m);
                    float32x4_t _w = vld1q_f32(line_w);
                    _sum = vfmaq_f32(_sum, _m, _w);

                    _m = vld1q_f32(m + 4);
                    _w = vld1q_f32(line_w + 4);
                    _sum2 = vfmaq_f32(_sum2, _m, _w);

                    m += 8;
                    w += 8;
                }
#else
                if (nn > 0)
                {
                asm volatile(
                    "0:                             \n"
                    "pld        [%1, #256]          \n"
                    "vld1.f32   {d0-d3}, [%1 :128]! \n"
                    "pld        [%2, #256]          \n"
                    "vld1.f32   {d4-d7}, [%2]!      \n"
                    "vmla.f32   %q3, q0, q2         \n"
                    "subs       %0, #1              \n"
                    "vmla.f32   %q4, q1, q3         \n"
                    "bne        0b                  \n"
                    : "=r"(nn),     // %0
                      "=r"(m),      // %1
                      "=r"(line_w),      // %2
                      "=w"(_sum),   // %3
                      "=w"(_sum2)   // %4
                    : "0"(nn),
                      "1"(m),
                      "2"(line_w),
                      "3"(_sum),
                      "4"(_sum2)
                    : "cc", "memory", "q0", "q1", "q2", "q3"
                );
                }
#endif // __aarch64__
#endif // __ARM_NEON
                for (; remain > 0; remain--) {
                    sum += *m * *line_w;

                    m++;
                    w++;
                }
            }

#if defined(__ARM_NEON)
            _sum = vaddq_f32(_sum, _sum2);
#if __aarch64__
            sum += vaddvq_f32(_sum);
#else
            float32x2_t _sumss = vadd_f32(vget_low_f32(_sum), vget_high_f32(_sum));
            _sumss = vpadd_f32(_sumss, _sumss);
            sum += vget_lane_f32(_sumss, 0);
#endif // __aarch64__
#endif // __ARM_NEON

            outptr[0] = sum;
        }

        return 0;
    }

} // namespace ncnn
#endif // __ARM_NEON