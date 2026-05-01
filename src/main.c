#define STB_IMAGE_IMPLEMENTATION
#include "../lib/stb_image.h" 

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../lib/stb_image_write.h"

#include "../lib/filters.h"
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>


#define MAX_FILTER_SIZE 9
#define FILTER_WIDTH 3
#define FILTER_HEIGHT 3

double factor = 1.0;
double bias = 0.0;

typedef enum {
    MODE_SEQ,
    MODE_PIXEL,
    MODE_ROW,
    MODE_COL,
    MODE_TILE
} ParallelMode;

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
void pixel_convolution(unsigned char* src, unsigned char* result, int w, int h, Filter* f){
    const long total = (long)w * h;
    #pragma omp parallel for
    for (long idx = 0; idx < total; ++idx)
    {
        int ox = (int)(idx % w);
        int oy = (int)(idx / w);
        double sum = 0.0;
        for (int filterX = 0; filterX<FILTER_WIDTH;++filterX)
        for (int filterY = 0; filterY<FILTER_HEIGHT;++filterY){
            int imageX = (ox-FILTER_WIDTH/2+filterX+w)%w;
            int imageY = (oy-FILTER_HEIGHT/2+filterY+h)%h;
            sum += src[imageY * w + imageX] * f->kernel[filterY][filterX];
        }
        result[idx] = clamp(sum);
    }
}

void column_convolution (unsigned char* src, unsigned char* result, int w, int h, Filter* f){
    #pragma omp parallel for
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

void row_convolution (unsigned char* src, unsigned char* result, int w, int h, Filter* f){
    #pragma omp parallel for
    for (int y = 0; y<h;++y)
    for (int x = 0; x<w;++x){
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

void tile_convolution (unsigned char* src, unsigned char* result, int w, int h, int TILE_W, int TILE_H, Filter* f){

    int tiles_x = (w + TILE_W - 1) / TILE_W;
    int tiles_y = (h + TILE_H - 1) / TILE_H;
    int total_tiles = tiles_x * tiles_y;

    #pragma omp parallel for
    for (int t = 0; t < total_tiles; ++t) {
        int tx = t % tiles_x;
        int ty = t / tiles_x;
        
        int x0 = tx * TILE_W;
        int y0 = ty * TILE_H;
        int x1 = (x0 + TILE_W < w) ? x0 + TILE_W : w;
        int y1 = (y0 + TILE_H < h) ? y0 + TILE_H : h;
        
        for (int y = y0; y < y1; ++y) {
            for (int x = x0; x < x1; ++x) {
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
    }
}

int main(int argc, char* argv[]){
    if (argc < 3) {
        fprintf(stderr, "usage: %s <input.png> <output> [--filter=name] [--parallel=mode]\n", argv[0]);
        printf("  3×3: identity, blur3, gaussian3, edge, sharpen3, emboss, mean\n");
        printf("  5×5: blur5, gaussian5, edge_h5, sharpen5\n");
        printf("  9×9: motion9\n");
        printf("Parallel modes: seq, pixel, row, col, tile (default: seq)\n");
        return EXIT_FAILURE;
    }
    char* filter_name = "identity";
    ParallelMode mode = MODE_SEQ;
    int tile_w = 32, tile_h = 32;

    for (int i = 3; i < argc; ++i) {
        if (strncmp(argv[i], "--filter=", 9) == 0)
            filter_name = argv[i] + 9;
        else if (strncmp(argv[i], "--parallel=", 11) == 0) {
            const char* m = argv[i] + 11;
            if (strcmp(m, "pixel") == 0) mode = MODE_PIXEL;
            else if (strcmp(m, "row") == 0) mode = MODE_ROW;
            else if (strcmp(m, "col") == 0) mode = MODE_COL;
            else if (strcmp(m, "tile") == 0) mode = MODE_TILE;
            else mode = MODE_SEQ;
        }
        else if (strncmp(argv[i], "--tile=", 7) == 0) {
            sscanf(argv[i] + 7, "%dx%d", &tile_w, &tile_h);
        }
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
    switch (mode) {
        case MODE_PIXEL: pixel_convolution(src, result, w, h, &filter); break;
        case MODE_ROW:   row_convolution(src, result, w, h, &filter);   break;
        case MODE_COL:   column_convolution(src, result, w, h, &filter);   break;
        case MODE_TILE:  tile_convolution(src, result, w, h, tile_w, tile_h, &filter); break;
        default:         seq_convolution(src, result, w, h, &filter);
    }
    clock_t end_time = clock();
    double elapsed = (double)(end_time-start_time)/ CLOCKS_PER_SEC;
    printf("время выполнения: %4f sec\n",elapsed);

    //здесь делаем статистики
    
    stbi_write_png(argv[2], w, h, 1, result, w);
    
    free(result);
    stbi_image_free(src);
    return 0;
}