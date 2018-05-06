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

#ifndef NCNN_PLATFORM_H
#define NCNN_PLATFORM_H

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
#ifdef __ARM_NEON
#undef __ARM_NEON 
#endif
#ifdef _MSC_VER
#define _OPENMP
#pragma comment(lib, "vcomp.lib")
#endif 
#else
#define  __ARM_NEON
#include <arm_neon.h>
#endif
#define NCNN_STDIO 1
#define NCNN_STRING 1

#endif // NCNN_PLATFORM_H
