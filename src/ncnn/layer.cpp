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


#include "layer.h" 
#include <string.h>

namespace ncnn {

	Layer::Layer() {
		one_blob_only = false;
		support_inplace = false;
	}

	Layer::~Layer() {
	}

	int Layer::load_param(const ParamDict & /*pd*/) {
		return 0;
	}

#if NCNN_STDIO

	int Layer::load_model(FILE * /*binfp*/) {
		return 0;
	}

#endif // NCNN_STDIO

	int Layer::load_model(const unsigned char *& /*mem*/) {
		return 0;
	}

	int Layer::forward(const std::vector<Mat> &bottom_blobs, std::vector<Mat> &top_blobs) const {
		if (!support_inplace)
			return -1;

		top_blobs = bottom_blobs;
		for (int i = 0; i < (int)top_blobs.size(); i++) {
			top_blobs[i] = bottom_blobs[i].clone();
			if (top_blobs[i].empty())
				return -100;
		}

		return forward_inplace(top_blobs);
	}

	int Layer::forward(const Mat &bottom_blob, Mat &top_blob) const {
		if (!support_inplace)
			return -1;

		top_blob = bottom_blob.clone();
		if (top_blob.empty())
			return -100;

		return forward_inplace(top_blob);
	}

	int Layer::forward_inplace(std::vector<Mat> & /*bottom_top_blobs*/) const {
		return -1;
	}

	int Layer::forward_inplace(Mat & /*bottom_top_blob*/) const {
		return -1;
	}
#if defined(__SSE__)
	extern Layer *Convolution_x86_layer_creator();
#endif
#if defined(__ARM_NEON)
	extern Layer *Convolution_arm_layer_creator();
#endif
	extern Layer *InnerProduct_layer_creator();

	extern Layer *Pooling_layer_creator();

	extern Layer *Input_layer_creator();

	extern Layer *Split_layer_creator();

	extern Layer *PReLU_layer_creator();

	extern Layer *Softmax_layer_creator();

	extern Layer *Dropout_layer_creator();


	static const layer_registry_entry layer_registry[] =
	{
#if defined(__SSE__)
#if NCNN_STRING
					{"Convolution", Convolution_x86_layer_creator},
#else
					{Convolution_x86_layer_creator},
#endif
#endif


#if defined(__ARM_NEON)
#if NCNN_STRING
	{ "Convolution", Convolution_arm_layer_creator },
#else
	{ Convolution_arm_layer_creator },
#endif
#endif
#if NCNN_STRING
					{"InnerProduct", InnerProduct_layer_creator},
#else
					{InnerProduct_layer_creator},
#endif
#if NCNN_STRING
					{"Pooling", Pooling_layer_creator},
#else
					{Pooling_layer_creator},
#endif
#if NCNN_STRING
					{"Input", Input_layer_creator},
#else
					{Input_layer_creator},
#endif
#if NCNN_STRING
					{"Split", Split_layer_creator},
#else
					{Split_layer_creator},
#endif
#if NCNN_STRING
					{"PReLU", PReLU_layer_creator},
#else
					{PReLU_layer_creator},
#endif
#if NCNN_STRING
					{"Softmax", Softmax_layer_creator},
#else
					{Softmax_layer_creator},
#endif
#if NCNN_STRING
					{"Dropout", Dropout_layer_creator},
#else
					{Dropout_layer_creator},
#endif

	};

	static const int layer_registry_entry_count = sizeof(layer_registry) / sizeof(layer_registry_entry);

#if NCNN_STRING

	int layer_to_index(const char *type) {
		for (int i = 0; i < layer_registry_entry_count; i++) {
			if (strcmp(type, layer_registry[i].name) == 0)
				return i;
		}

		return -1;
	}

#endif // NCNN_STRING

	Layer *create_layer(int index) {
		if (index < 0 || index >= layer_registry_entry_count)
			return 0;

		layer_creator_func layer_creator = layer_registry[index].creator;
		if (!layer_creator)
			return 0;

		return layer_creator();
	}

} // namespace ncnn
