#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#include "stb_image.h"
#include "filters.h"
#include "conv.h"

#define WARMUP_RUNS 3
#define BENCH_RUNS 10

#define MAX_ITEMS 32

double bench_once(unsigned char *s, unsigned char *r, int w, int h,
                  Filter *f, ParallelMode mode, int tile_w,
                  int tile_h) {
  double t0 = omp_get_wtime();
  switch (mode) {
  case MODE_SEQ:
    seq_convolution(s, r, w, h,f);
    break;
  case MODE_PIXEL:
    pixel_convolution(s, r, w, h,f);
    break;
  case MODE_ROW:
    row_convolution(s, r, w, h, f);
    break;
  case MODE_COL:
    column_convolution(s, r, w, h, f);
    break;
  case MODE_TILE:
    tile_convolution(s, r, w, h, tile_w, tile_h, f);
    break;
  }
  double t1 = omp_get_wtime();
  return (t1 - t0);
}

static int cmp_double(const void *a, const void *b) {
  double x = *(const double *)a, y = *(const double *)b;
  return (x > y) - (x < y);
}

double bench_median(const unsigned char *s, unsigned char *r, int w, int h,
                    const Filter *f, ParallelMode mode, int tw, int th,
                    double *min_out, double *mean_out) {
  double times[BENCH_RUNS];
  for (int i = 0; i < WARMUP_RUNS; ++i)
    bench_once(s, r, w, h, f, mode, tw, th);

  double total = 0, min = 0;
  for (int i = 0; i < BENCH_RUNS; ++i) {
    times[i] = bench_once(s, r, w, h, f, mode, tw, th);
    total += times[i];
    if (i == 0 || times[i] < min)
      min = times[i];
  }
  qsort(times, BENCH_RUNS, sizeof times[0], cmp_double);
  *min_out = min;
  *mean_out = total / BENCH_RUNS;
  return 0.5 * (times[BENCH_RUNS / 2 - 1] + times[BENCH_RUNS / 2]);
}

const char *mode_name(ParallelMode m) {
  static const char *names[] = {"seq", "pixel", "row", "col", "tile"};
  return names[m];
}

static int run_case(FILE *csv, const char *experiment, int size,
                    const char *filter_name, const Filter *filter,
                    const unsigned char *src, unsigned char *res,
                    const unsigned char *ref, ParallelMode mode, int threads,
                    int tw, int th) {
  if (mode == MODE_SEQ)
    threads = 1;
  omp_set_num_threads(threads);

  double min, mean;
  double med = bench_median(src, res, size, size, filter, mode, tw, th, &min,
                            &mean);
  int bad = memcmp(res, ref, (size_t)size * size) != 0;
  if (bad)
    fprintf(stderr, "error: режим %s не совпал с эталоном\n", mode_name(mode));

  char tile[24] = "-";
  if (mode == MODE_TILE)
    snprintf(tile, sizeof tile, "%dx%d", tw, th);
  double mpix = (double)size * size / med / 1e6;

  printf("  %-10s %-10s %-5s %-8s t=%-2d: %.4f s (%.2f MPix/s)\n", experiment,
         filter_name, mode_name(mode), tile, threads, med, mpix);
  if (csv) {
    fprintf(csv, "%s,%d,%s,%d,%s,%s,%d,%d,%.9f,%.9f,%.9f,%.3f\n", experiment,
            size, filter_name, filter->size, mode_name(mode), tile, threads,
            BENCH_RUNS, med, min, mean, mpix);
    fflush(csv);
  }
  return bad;
}

static int parse_int_list(const char *s, int *out, int max) {
  int n = 0;
  while (*s && n < max) {
    char *end;
    long v = strtol(s, &end, 10);
    if (end == s || v < 1 || (*end != ',' && *end != '\0'))
      return -1;
    out[n++] = (int)v;
    s = (*end == ',') ? end + 1 : end;
  }
  return n > 0 ? n : -1;
}

int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr,
            "Usage: %s <image.png> [--csv=output.csv] "
            "[--threads=1,2,4,8,12,16,20]\n",
            argv[0]);
    return 1;
  }

  const char *input = argv[1];
  const char *csv_out = NULL;
  int threads[MAX_ITEMS] = {1, 2, 4, 8, 12, 16, 20};
  int nthreads = 7;

  int sizes[] = {128, 512, 1024, 2048};
  int nsizes = 4;
  const char *filter_names[] = {"gaussian3", "gaussian5", "motion9"};
  int nfilters = 3;
  const char *tiles[] = {"8x8",    "16x16",  "32x32", "64x64",
                         "128x128", "16x256", "256x16"};
  int ntiles = 7;

  for (int i = 2; i < argc; ++i) {
    if (strncmp(argv[i], "--csv=", 6) == 0) {
      csv_out = argv[i] + 6;
    } else if (strncmp(argv[i], "--threads=", 10) == 0) {
      nthreads = parse_int_list(argv[i] + 10, threads, MAX_ITEMS);
      if (nthreads < 0) {
        fprintf(stderr, "Некорректный --threads: %s\n", argv[i] + 10);
        return 1;
      }
    } else {
      fprintf(stderr, "Неизвестный параметр: %s\n", argv[i]);
      return 1;
    }
  }

  int w, h, ch;
  unsigned char *src = stbi_load(input, &w, &h, &ch, 1);
  if (!src) {
    fprintf(stderr, "Error loading %s: %s\n", input, stbi_failure_reason());
    return 1;
  }

  FILE *csv = NULL;
  if (csv_out) {
    csv = fopen(csv_out, "w");
    if (!csv) {
      fprintf(stderr, "Не удалось открыть %s\n", csv_out);
      stbi_image_free(src);
      return 1;
    }
    fprintf(csv, "experiment,size,filter,filter_size,mode,tile,threads,runs,"
                 "median_sec,min_sec,mean_sec,mpix_per_sec\n");
  }

  printf("Benchmark: %s (%dx%d), Runs: %d (warmup: %d), procs: %d\n\n", input,
         w, h, BENCH_RUNS, WARMUP_RUNS, omp_get_num_procs());

  ParallelMode modes[] = {MODE_PIXEL, MODE_ROW, MODE_COL, MODE_TILE};
  int errors = 0;

  for (int s = 0; s < nsizes; ++s) {
    int size = sizes[s];
    size_t bytes = (size_t)size * size;
    unsigned char *test_src = malloc(bytes);
    unsigned char *test_res = malloc(bytes);
    unsigned char *ref = malloc(bytes);
    if (!test_src || !test_res || !ref) {
      fprintf(stderr, "Недостаточно памяти\n");
      return 1;
    }
    for (int y = 0; y < size; ++y)
      for (int x = 0; x < size; ++x)
        test_src[(size_t)y * size + x] = src[(size_t)(y % h) * w + (x % w)];

    printf("Size: %dx%d\n", size, size);

    for (int fi = 0; fi < nfilters; ++fi) {
      Filter filter = get_filter(filter_names[fi]);
      seq_convolution(test_src, ref, size, size, &filter);
      errors += run_case(csv, "strategies", size, filter_names[fi], &filter,
                         test_src, test_res, ref, MODE_SEQ, 1, 0, 0);
      for (int m = 0; m < 4; ++m)
        for (int t = 0; t < nthreads; ++t)
          errors += run_case(csv, "strategies", size, filter_names[fi],
                             &filter, test_src, test_res, ref, modes[m],
                             threads[t], 32, 32);
    }

    if (s == nsizes - 1) {
      Filter filter = get_filter(filter_names[0]);
      seq_convolution(test_src, ref, size, size, &filter);
      errors += run_case(csv, "tiles", size, filter_names[0], &filter,
                         test_src, test_res, ref, MODE_SEQ, 1, 0, 0);
      for (int k = 0; k < ntiles; ++k) {
        int tw, th;
        sscanf(tiles[k], "%dx%d", &tw, &th);
        for (int t = 0; t < nthreads; ++t)
          errors += run_case(csv, "tiles", size, filter_names[0], &filter,
                             test_src, test_res, ref, MODE_TILE, threads[t],
                             tw, th);
      }
    }
    printf("\n");
    free(test_src);
    free(test_res);
    free(ref);
  }

  if (csv)
    fclose(csv);
  stbi_image_free(src);
  printf("Results saved to: %s\n", csv_out ? csv_out : "(stdout only)");
  if (errors) {
    fprintf(stderr, "error: %d замеров не совпали с последовательным\n",
            errors);
    return 2;
  }
  return 0;
}
