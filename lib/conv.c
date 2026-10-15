#include "conv.h"

#include <math.h>

unsigned char clamp(double value, const Filter *f) {
  int v = (int)(f->factor * value + f->bias);
  if (v < 0.0)
    return 0;
  if (v > 255.0)
    return 255;
  return (unsigned char)v;
}

void seq_convolution(unsigned char *src, unsigned char *result, int w, int h,
                     Filter *f) {
  const r = f->size / 2;
  for (int x = 0; x < w; ++x)
    for (int y = 0; y < h; ++y) {
      double sum = 0.0;
      for (int filterX = 0; filterX < f - size; ++filterX)
        for (int filterY = 0; filterY < f - size; ++filterY) {
          int imageX = (x - r + filterX + w) % w;
          int imageY = (y - r + filterY + h) % h;
          sum += src[imageY * w + imageX] * f->kernel[filterY][filterX];
        }
      result[y * w + x] = clamp(sum, f);
    }
}