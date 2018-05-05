#include "mtcnn.h"
#include "browse.h"
#define USE_SHELL_OPEN
#ifndef  nullptr
#define nullptr 0
#endif
#if defined(_MSC_VER)
#define _CRT_SECURE_NO_WARNINGS
#include <windows.h> 
#else
#include <unistd.h>
#endif
#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION

#include "stb_image.h"
//ref:https://github.com/nothings/stb/blob/master/stb_image.h
#define TJE_IMPLEMENTATION

#include "tiny_jpeg.h"
//ref:https://github.com/serge-rgb/TinyJPEG/blob/master/tiny_jpeg.h

#include <stdint.h>
#include "timing.h"

char saveFile[1024];

unsigned char *loadImage(const char *filename, int *Width, int *Height, int *Channels) {
	return stbi_load(filename, Width, Height, Channels, 0);
}

void saveImage(const char *filename, int Width, int Height, int Channels, unsigned char *Output) {
	memcpy(saveFile + strlen(saveFile), filename, strlen(filename));
	*(saveFile + strlen(saveFile) + 1) = 0;
	//保存为jpg
	if (!tje_encode_to_file(saveFile, Width, Height, Channels, true, Output)) {
		fprintf(stderr, "save JPEG fail.\n");
		return;
	}

#ifdef USE_SHELL_OPEN
	browse(saveFile);
#endif
}

void splitpath(const char *path, char *drv, char *dir, char *name, char *ext) {
	const char *end;
	const char *p;
	const char *s;
	if (path[0] && path[1] == ':') {
		if (drv) {
			*drv++ = *path++;
			*drv++ = *path++;
			*drv = '\0';
		}
	}
	else if (drv)
		*drv = '\0';
	for (end = path; *end && *end != ':';)
		end++;
	for (p = end; p > path && *--p != '\\' && *p != '/';)
		if (*p == '.') {
			end = p;
			break;
		}
	if (ext)
		for (s = end; (*ext = *s++);)
			ext++;
	for (p = end; p > path;)
		if (*--p == '\\' || *p == '/') {
			p++;
			break;
		}
	if (name) {
		for (s = p; s < end;)
			*name++ = *s++;
		*name = '\0';
	}
	if (dir) {
		for (s = path; s < p;)
			*dir++ = *s++;
		*dir = '\0';
	}
}

void getCurrentFilePath(const char *filePath, char *saveFile) {
	char drive[_MAX_DRIVE];
	char dir[_MAX_DIR];
	char fname[_MAX_FNAME];
	char ext[_MAX_EXT];
	splitpath(filePath, drive, dir, fname, ext);
	size_t n = strlen(filePath);
	memcpy(saveFile, filePath, n);
	char *cur_saveFile = saveFile + (n - strlen(ext));
	cur_saveFile[0] = '_';
	cur_saveFile[1] = 0;
}

void drawPoint(unsigned char *bits, int width, int depth, int x, int y, const uint8_t *color) {
	for (int i = 0; i < min(depth, 3); ++i) {
		bits[(y * width + x) * depth + i] = color[i];
	}
}

void drawLine(unsigned char *bits, int width, int depth, int startX, int startY, int endX, int endY,
	const uint8_t *col) {
	if (endX == startX) {
		if (startY > endY) {
			int a = startY;
			startY = endY;
			endY = a;
		}
		for (int y = startY; y <= endY; y++) {
			drawPoint(bits, width, depth, startX, y, col);
		}
	}
	else {
		float m = 1.0f * (endY - startY) / (endX - startX);
		int y = 0;
		if (startX > endX) {
			int a = startX;
			startX = endX;
			endX = a;
		}
		for (int x = startX; x <= endX; x++) {
			y = (int)(m * (x - startX) + startY);
			drawPoint(bits, width, depth, x, y, col);
		}
	}
}

void drawRectangle(unsigned char *bits, int width, int depth, int x1, int y1, int x2, int y2, const uint8_t *col) {
	drawLine(bits, width, depth, x1, y1, x2, y1, col);
	drawLine(bits, width, depth, x2, y1, x2, y2, col);
	drawLine(bits, width, depth, x2, y2, x1, y2, col);
	drawLine(bits, width, depth, x1, y2, x1, y1, col);
}

int main(int argc, char **argv) {
	printf("mtcnn face detection\n");
	printf("blog:http://cpuimage.cnblogs.com/\n");

	if (argc < 2) {
		printf("usage: %s  model_path image_file \n ", argv[0]);
		printf("eg: %s  ../models ../sample.jpg \n ", argv[0]);
		printf("press any key to exit. \n");
		getchar();
		return 0;
	}
	const char *model_path = argv[1];
	char *szfile = argv[2];
	getCurrentFilePath(szfile, saveFile);
	int Width = 0;
	int Height = 0;
	int Channels = 0;
	unsigned char *inputImage = loadImage(szfile, &Width, &Height, &Channels);
	if (inputImage == nullptr || Channels != 3) return -1;
	ncnn::Mat ncnn_img = ncnn::Mat::from_pixels(inputImage, ncnn::Mat::PIXEL_RGB, Width, Height);
	std::vector<Bbox> finalBbox;
	MTCNN mtcnn(model_path);
	double startTime = now();
	mtcnn.detect(ncnn_img, finalBbox);
	double nDetectTime = calcElapsed(startTime, now());
	printf("time: %d ms.\n ", (int)(nDetectTime * 1000));
	int num_box = finalBbox.size();
	printf("face num: %u \n", num_box);
	for (int i = 0; i < num_box; i++) {
		const uint8_t red[3] = { 255, 0, 0 };
		drawRectangle(inputImage, Width, Channels, finalBbox[i].x1, finalBbox[i].y1,
			finalBbox[i].x2,
			finalBbox[i].y2, red);
		const uint8_t blue[3] = { 0, 0, 255 };
		for (int num = 0; num < 5; num++) {
			drawPoint(inputImage, Width, Channels, (int)(finalBbox[i].ppoint[num] + 0.5f),
				(int)(finalBbox[i].ppoint[num + 5] + 0.5f), blue);
		}
	}
	saveImage("_done.jpg", Width, Height, Channels, inputImage);
	free(inputImage);
	getchar();
	return 0;
}