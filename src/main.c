#define STB_IMAGE_IMPLEMENTATION
#include "../lib/stb_image.h" 

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../lib/stb_image_write.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "filters.h"

#define MAX_FILTER_SIZE 9
#define FILTER_WIDTH 3
#define FILTER_HEIGHT 3

double factor = 1.0;
double bias = 0.0;


static inline unsigned char clamp(double value) {
    int v = (int)(factor * value + bias);
    if (v < 0) return 0;
    if (v > 255) return 255;
    return (unsigned char)v;
}

void seq_convolution (unsigned char* src, unsigned char* result, int w, int h, Filter* f){
    for (int x = 0; x<w;++x)
    for (int y = 0; y<h;++y){
        double sum = 0.0;
        for (int filterX = 0; filterX<FILTER_WIDTH;++filterX)
        for (int filterY = 0; filterY<FILTER_HEIGHT;++filterY){
            int imageX = (x-FILTER_WIDTH/2+filterX+w)%w;
            int imageY = (y-FILTER_HEIGHT/2+filterY+h)%h;
            sum += src[imageY * w + imageX] * f->kernel[filterY][filterX];
        }
        result[y*w+x] = clamp(sum);
    }
}

int main(int argc, char* argv[]){
    if (argc < 3) {
        fprintf(stderr, "usage: %s <input.png> <output> [--filter=name]\n", argv[0]);
        printf("Available filters:\n");
        printf("  blur_3x3, blur_5x5\n");
        printf("  gaussian_3x3, gaussian_5x5\n");
        printf("  motion_9x9\n");
        printf("  edge_3x3, edge_5x5_h\n");
        printf("  sharpen_3x3, sharpen_5x5\n");
        printf("  emboss_3x3\n");
        printf("  mean_3x3\n");
        printf("  identity (default)\n");
        return EXIT_FAILURE;
    }
    char* filter_name = "identity";

    for (int i = 3; i < argc; ++i) {
        if (strncmp(argv[i], "--filter=", 9) == 0)
            filter_name = argv[i] + 9;
    }

    Filter filter = get_filter(filter_name);

    int w, h, channels;
    unsigned char* src = stbi_load(argv[1], &w, &h, &channels, 1); // 1 = grayscale
    if (!src) {
        fprintf(stderr, "Ошибка загрузки: %s\n", stbi_failure_reason());
        return 1;
    }

    unsigned char* result = malloc(w * h * sizeof(unsigned char));
    if (!result) {
        fprintf(stderr, "Ошибка выделения памяти\n");
        stbi_image_free(src);
        return 1;
    }

    clock_t start_time = clock();
    seq_convolution(src,result,w,h,&filter);
    clock_t end_time = clock();
    double elapsed = (double)(end_time-start_time)/ CLOCKS_PER_SEC;
    printf("время выполнения: %4f sec\n",elapsed);

    //здесь делаем статистики
    
    stbi_write_png(argv[2], w, h, 1, result, w);
    
    free(result);
    stbi_image_free(src);
    return 0;
}