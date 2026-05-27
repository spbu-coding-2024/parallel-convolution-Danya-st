#define STB_IMAGE_IMPLEMENTATION
#include "../lib/stb_image.h"
#include "../lib/stb_image_write.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include "filters.h"

#define FILTER_WIDTH 3
#define FILTER_HEIGHT 3
#define WARMUP_RUNS 3
#define BENCH_RUNS 10

typedef enum { MODE_SEQ, MODE_PIXEL, MODE_ROW, MODE_COL, MODE_TILE } ParMode;

static inline unsigned char clamp(double v, double f, double b) {
  int val = (int)(f * v + b);
  return (val < 0) ? 0 : (val > 255) ? 255 : (unsigned char)val;
}

void seq_conv(unsigned char *s, unsigned char *r, int w, int h, double k[3][3],
              double f, double b) {
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      double sum = 0;
      for (int fy = 0; fy < 3; ++fy)
        for (int fx = 0; fx < 3; ++fx) {
          int ix = (x - 1 + fx + w) % w;
          int iy = (y - 1 + fy + h) % h;
          sum += s[iy * w + ix] * k[fy][fx];
        }
      r[y * w + x] = clamp(sum, f, b);
    }
}

void par_pixel(unsigned char *s, unsigned char *r, int w, int h, double k[3][3],
               double f, double b) {
#pragma omp parallel for
  for (long idx = 0; idx < (long)w * h; ++idx) {
    int x = idx % w, y = idx / w;
    double sum = 0;
    for (int fy = 0; fy < 3; ++fy)
      for (int fx = 0; fx < 3; ++fx) {
        int ix = (x - 1 + fx + w) % w;
        int iy = (y - 1 + fy + h) % h;
        sum += s[iy * w + ix] * k[fy][fx];
      }
    r[y * w + x] = clamp(sum, f, b);
  }
}

void par_row(unsigned char *s, unsigned char *r, int w, int h, double k[3][3],
             double f, double b) {
#pragma omp parallel for
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      double sum = 0;
      for (int fy = 0; fy < 3; ++fy)
        for (int fx = 0; fx < 3; ++fx) {
          int ix = (x - 1 + fx + w) % w;
          int iy = (y - 1 + fy + h) % h;
          sum += s[iy * w + ix] * k[fy][fx];
        }
      r[y * w + x] = clamp(sum, f, b);
    }
}

void par_col(unsigned char *s, unsigned char *r, int w, int h, double k[3][3],
             double f, double b) {
#pragma omp parallel for
  for (int x = 0; x < w; ++x)
    for (int y = 0; y < h; ++y) {
      double sum = 0;
      for (int fy = 0; fy < 3; ++fy)
        for (int fx = 0; fx < 3; ++fx) {
          int ix = (x - 1 + fx + w) % w;
          int iy = (y - 1 + fy + h) % h;
          sum += s[iy * w + ix] * k[fy][fx];
        }
      r[y * w + x] = clamp(sum, f, b);
    }
}

void par_tile(unsigned char *s, unsigned char *r, int w, int h, int tw, int th,
              double k[3][3], double f, double b) {
  int tx = (w + tw - 1) / tw, ty = (h + th - 1) / th;
#pragma omp parallel for
  for (int t = 0; t < tx * ty; ++t) {
    int cx = t % tx, cy = t / tx;
    int x0 = cx * tw, y0 = cy * th;
    int x1 = (x0 + tw < w) ? x0 + tw : w;
    int y1 = (y0 + th < h) ? y0 + th : h;
    for (int y = y0; y < y1; ++y)
      for (int x = x0; x < x1; ++x) {
        double sum = 0;
        for (int fy = 0; fy < 3; ++fy)
          for (int fx = 0; fx < 3; ++fx) {
            int ix = (x - 1 + fx + w) % w;
            int iy = (y - 1 + fy + h) % h;
            sum += s[iy * w + ix] * k[fy][fx];
          }
        r[y * w + x] = clamp(sum, f, b);
      }
  }
}

double bench_once(unsigned char *s, unsigned char *r, int w, int h,
                  double k[3][3], double f, double b, ParMode mode, int tile_w,
                  int tile_h) {
  double t0 = omp_get_wtime();
  switch (mode) {
  case MODE_SEQ:
    seq_conv(s, r, w, h, k, f, b);
    break;
  case MODE_PIXEL:
    par_pixel(s, r, w, h, k, f, b);
    break;
  case MODE_ROW:
    par_row(s, r, w, h, k, f, b);
    break;
  case MODE_COL:
    par_col(s, r, w, h, k, f, b);
    break;
  case MODE_TILE:
    par_tile(s, r, w, h, tile_w, tile_h, k, f, b);
    break;
  }
  double t1 = omp_get_wtime();
  return (t1 - t0);
}

double bench_avg(unsigned char *s, unsigned char *r, int w, int h,
                 double k[3][3], double f, double b, ParMode mode, int tw,
                 int th) {
  for (int i = 0; i < WARMUP_RUNS; ++i)
    bench_once(s, r, w, h, k, f, b, mode, tw, th);

  double total = 0;
  for (int i = 0; i < BENCH_RUNS; ++i)
    total += bench_once(s, r, w, h, k, f, b, mode, tw, th);
  return total / BENCH_RUNS;
}

const char *mode_name(ParMode m) {
  static const char *names[] = {"seq", "pixel", "row", "col", "tile"};
  return names[m];
}

int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr, "Usage: %s <image.png> [--csv=output.csv] [--threads=N]\n",
            argv[0]);
    return 1;
  }

  const char *input = argv[1];
  const char *csv_out = NULL;
  int max_threads = omp_get_max_threads();

  for (int i = 2; i < argc; ++i) {
    if (strncmp(argv[i], "--csv=", 6) == 0)
      csv_out = argv[i] + 6;
    if (strncmp(argv[i], "--threads=", 10) == 0)
      max_threads = atoi(argv[i] + 10);
  }

  int w, h, ch;
  unsigned char *src = stbi_load(input, &w, &h, &ch, 1);
  if (!src) {
    fprintf(stderr, "Error loading %s: %s\n", input, stbi_failure_reason());
    return 1;
  }

  double kernel[3][3] = {{1, 2, 1}, {2, 4, 2}, {1, 2, 1}};
  double factor = 1.0 / 16.0, bias = 0.0;

  int sizes[][2] = {{128, 128}, {512, 512}, {1024, 1024}, {2048, 2048}};
  ParMode modes[] = {MODE_SEQ, MODE_PIXEL, MODE_ROW, MODE_COL, MODE_TILE};

  if (csv_out) {
    FILE *f = fopen(csv_out, "w");
    if (f) {
      fprintf(f, "size,mode,threads,time_sec,mpix_per_sec\n");
      fclose(f);
    }
  }

  printf("Benchmark: %s (%dx%d), filter: gaussian_3x3\n", input, w, h);
  printf("Threads: 1..%d, Runs: %d (warmup: %d)\n\n", max_threads, BENCH_RUNS,
         WARMUP_RUNS);

  for (int s = 0; s < 4; ++s) {
    int sw = sizes[s][0], sh = sizes[s][1];
    if (sw > w || sh > h)
      continue;

    unsigned char *test_src = malloc(sw * sh);
    unsigned char *test_res = malloc(sw * sh);
    for (int y = 0; y < sh; ++y)
      memcpy(test_src + y * sw, src + y * w, sw);

    printf("Size: %dx%d\n", sw, sh);
    for (int m = 0; m < 5; ++m) {
      for (int t = 1; t <= max_threads; t *= 2) {
        omp_set_num_threads(t);
        double avg = bench_avg(test_src, test_res, sw, sh, kernel, factor, bias,
                               modes[m], 32, 32);
        long pixels = (long)sw * sh;
        double mpix = pixels / avg / 1e6;

        printf("  %-6s t=%2d: %.4f s (%.2f MPix/s)\n", mode_name(modes[m]), t,
               avg, mpix);

        if (csv_out) {
          FILE *f = fopen(csv_out, "a");
          if (f) {
            fprintf(f, "%dx%d,%s,%d,%.6f,%.2f\n", sw, sh, mode_name(modes[m]),
                    t, avg, mpix);
            fclose(f);
          }
        }
      }
    }
    printf("\n");
    free(test_src);
    free(test_res);
  }

  stbi_image_free(src);
  printf("Results saved to: %s\n", csv_out ? csv_out : "(stdout only)");
  return 0;
}
