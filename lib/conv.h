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

unsigned char clamp(double value, const Filter *f);

void seq_convolution(unsigned char *src, unsigned char *result, int w, int h,
                     Filter *f);

void pixel_convolution(unsigned char *src, unsigned char *result, int w, int h,
                       Filter *f)
            
void column_convolution(unsigned char *src, unsigned char *result, int w, int h,
                        Filter *f) 

void row_convolution(unsigned char *src, unsigned char *result, int w, int h,
                     Filter *f)

void tile_convolution(unsigned char *src, unsigned char *result, int w, int h,
                      int TILE_W, int TILE_H, Filter *f)