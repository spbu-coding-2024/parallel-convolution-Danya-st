#ifndef CONV_H
#define CONV_H

#include <stddef.h>

#include "filters.h"

typedef enum {
  MODE_SEQ,
  MODE_PIXEL,
  MODE_ROW,
  MODE_COL,
  MODE_TILE
} ParallelMode;

void seq_convolution(const unsigned char *src, unsigned char *result, int w,
                     int h, const Filter *f);

void pixel_convolution(const unsigned char *src, unsigned char *result, int w,
                       int h, const Filter *f);

void column_convolution(const unsigned char *src, unsigned char *result, int w,
                        int h, const Filter *f);

void row_convolution(const unsigned char *src, unsigned char *result, int w,
                     int h, const Filter *f);

void tile_convolution(const unsigned char *src, unsigned char *result, int w,
                      int h, int TILE_W, int TILE_H, const Filter *f);

int conv_built_with_openmp(void);

#endif