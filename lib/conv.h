#ifndef CONV_H
#define CONV_H

#include <stddef.h>

#include "filters.h"

unsigned char clamp(double value, const Filter *f);

void seq_convolution(unsigned char *src, unsigned char *result, int w, int h,
                     Filter *f);

#endif
