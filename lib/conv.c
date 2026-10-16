#include "conv.h"

#include <math.h>

unsigned char clamp(double value, const Filter *f) {
  double v = floor(f->factor * value + f->bias + 0.5);
  if (v < 0.0)
    return 0;
  if (v > 255.0)
    return 255;
  return (unsigned char)v;
}

void seq_convolution(unsigned char *src, unsigned char *result, int w, int h,
                     Filter *f) {
  const int r = f->size / 2;
  for (int x = 0; x < w; ++x)
    for (int y = 0; y < h; ++y) {
      double sum = 0.0;
      for (int filterX = 0; filterX < f->size; ++filterX)
        for (int filterY = 0; filterY < f->size; ++filterY) {
          int imageX = (x - r + filterX + w) % w;
          int imageY = (y - r + filterY + h) % h;
          sum += src[imageY * w + imageX] * f->kernel[filterY][filterX];
        }
      result[y * w + x] = clamp(sum, f);
    }
}

void pixel_convolution(unsigned char *src, unsigned char *result, int w, int h,
                       Filter *f) {
  const long total = (long)w * h;
  const int r = f->size / 2;
#pragma omp parallel for
  for (long idx = 0; idx < total; ++idx) {
    int ox = (int)(idx % w);
    int oy = (int)(idx / w);
    double sum = 0.0;
    for (int filterX = 0; filterX < f->size; ++filterX)
      for (int filterY = 0; filterY < f->size; ++filterY) {
        int imageX = (ox - r + filterX + w) % w;
        int imageY = (oy - r + filterY + h) % h;
        sum += src[imageY * w + imageX] * f->kernel[filterY][filterX];
      }
    result[idx] = clamp(sum,f);
  }
}

void column_convolution(unsigned char *src, unsigned char *result, int w, int h,
                        Filter *f) {
  const int r = f->size / 2;
#pragma omp parallel for
  for (int x = 0; x < w; ++x)
    for (int y = 0; y < h; ++y) {
      double sum = 0.0;
      for (int filterX = 0; filterX < f->size; ++filterX)
        for (int filterY = 0; filterY < f->size; ++filterY) {
          int imageX = (x - r + filterX + w) % w;
          int imageY = (y - r + filterY + h) % h;
          sum += src[imageY * w + imageX] * f->kernel[filterY][filterX];
        }
      result[y * w + x] = clamp(sum,f);
    }
}

void row_convolution(unsigned char *src, unsigned char *result, int w, int h,
                     Filter *f) {
  const int r = f->size / 2;
#pragma omp parallel for
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      double sum = 0.0;
      for (int filterX = 0; filterX < f->size; ++filterX)
        for (int filterY = 0; filterY < f->size; ++filterY) {
          int imageX = (x - r + filterX + w) % w;
          int imageY = (y - r + filterY + h) % h;
          sum += src[imageY * w + imageX] * f->kernel[filterY][filterX];
        }
      result[y * w + x] = clamp(sum,f);
    }
}

void tile_convolution(unsigned char *src, unsigned char *result, int w, int h,
                      int TILE_W, int TILE_H, Filter *f) {

  int tiles_x = (w + TILE_W - 1) / TILE_W;
  int tiles_y = (h + TILE_H - 1) / TILE_H;
  int total_tiles = tiles_x * tiles_y;
  const int r = f->size / 2;
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
        for (int filterX = 0; filterX < f->size; ++filterX)
          for (int filterY = 0; filterY < f->size; ++filterY) {
            int imageX = (x - r + filterX + w) % w;
            int imageY = (y - r + filterY + h) % h;
            sum += src[imageY * w + imageX] * f->kernel[filterY][filterX];
          }
        result[y * w + x] = clamp(sum,f);
      }
    }
  }
}