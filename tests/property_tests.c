#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#define FILTER_WIDTH 3
#define FILTER_HEIGHT 3

#define EPSILON 1e-6
#define TEST_IMG_W 64
#define TEST_IMG_H 64

double factor = 1.0;
double bias = 0.0;

static inline unsigned char clamp(double value) {
  int v = (int)(factor * value + bias);
  if (v < 0)
    return 0;
  if (v > 255)
    return 255;
  return (unsigned char)v;
}

void seq_convolution(unsigned char *src, unsigned char *result, int w, int h,
                     double kernel[FILTER_HEIGHT][FILTER_WIDTH]) {
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      double sum = 0.0;
      for (int filterY = 0; filterY < FILTER_HEIGHT; ++filterY) {
        for (int filterX = 0; filterX < FILTER_WIDTH; ++filterX) {
          int ix = (x - FILTER_WIDTH / 2 + filterX + w) % w;
          int iy = (y - FILTER_HEIGHT / 2 + filterY + h) % h;
          sum += src[iy * w + ix] * kernel[filterY][filterX];
        }
      }
      result[y * w + x] = clamp(sum);
    }
  }
}

void pixel_convolution(unsigned char *src, unsigned char *result, int w, int h,
                       double kernel[FILTER_HEIGHT][FILTER_WIDTH]) {
  const long total = (long)w * h;
#pragma omp parallel for
  for (long idx = 0; idx < total; ++idx) {
    int ox = (int)(idx % w);
    int oy = (int)(idx / w);
    double sum = 0.0;
    for (int filterX = 0; filterX < FILTER_WIDTH; ++filterX)
      for (int filterY = 0; filterY < FILTER_HEIGHT; ++filterY) {
        int imageX = (ox - FILTER_WIDTH / 2 + filterX + w) % w;
        int imageY = (oy - FILTER_HEIGHT / 2 + filterY + h) % h;
        sum += src[imageY * w + imageX] * kernel[filterY][filterX];
      }
    result[idx] = clamp(sum);
  }
}

void column_convolution(unsigned char *src, unsigned char *result, int w, int h,
                        double kernel[FILTER_HEIGHT][FILTER_WIDTH]) {
#pragma omp parallel for
  for (int x = 0; x < w; ++x)
    for (int y = 0; y < h; ++y) {
      double sum = 0.0;
      for (int filterX = 0; filterX < FILTER_WIDTH; ++filterX)
        for (int filterY = 0; filterY < FILTER_HEIGHT; ++filterY) {
          int imageX = (x - FILTER_WIDTH / 2 + filterX + w) % w;
          int imageY = (y - FILTER_HEIGHT / 2 + filterY + h) % h;
          sum += src[imageY * w + imageX] * kernel[filterY][filterX];
        }
      result[y * w + x] = clamp(sum);
    }
}

void row_convolution(unsigned char *src, unsigned char *result, int w, int h,
                     double kernel[FILTER_HEIGHT][FILTER_WIDTH]) {
#pragma omp parallel for
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      double sum = 0.0;
      for (int filterX = 0; filterX < FILTER_WIDTH; ++filterX)
        for (int filterY = 0; filterY < FILTER_HEIGHT; ++filterY) {
          int imageX = (x - FILTER_WIDTH / 2 + filterX + w) % w;
          int imageY = (y - FILTER_HEIGHT / 2 + filterY + h) % h;
          sum += src[imageY * w + imageX] * kernel[filterY][filterX];
        }
      result[y * w + x] = clamp(sum);
    }
}

void tile_convolution(unsigned char *src, unsigned char *result, int w, int h,
                      int TILE_W, int TILE_H,
                      double kernel[FILTER_HEIGHT][FILTER_WIDTH]) {

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
        for (int filterX = 0; filterX < FILTER_WIDTH; ++filterX)
          for (int filterY = 0; filterY < FILTER_HEIGHT; ++filterY) {
            int imageX = (x - FILTER_WIDTH / 2 + filterX + w) % w;
            int imageY = (y - FILTER_HEIGHT / 2 + filterY + h) % h;
            sum += src[imageY * w + imageX] * kernel[filterY][filterX];
          }
        result[y * w + x] = clamp(sum);
      }
    }
  }
}
static unsigned char *make_random_image(int w, int h) {
  unsigned char *img = malloc(w * h);
  for (int i = 0; i < w * h; ++i)
    img[i] = rand() % 256;
  return img;
}

static int images_equal(const unsigned char *a, const unsigned char *b, int w,
                        int h) {
  for (int i = 0; i < w * h; ++i)
    if (abs((int)a[i] - (int)b[i]) > 1)
      return 0;
  return 1;
}

static int image_is_constant(const unsigned char *img, int w, int h,
                             unsigned char value) {
  for (int i = 0; i < w * h; ++i)
    if (img[i] != value)
      return 0;
  return 1;
}

static int test_parallel_vs_seq(void (*par_func)(unsigned char *,
                                                 unsigned char *, int, int,
                                                 double[3][3]),
                                const char *name, int w, int h) {
  unsigned char *src = make_random_image(w, h);
  unsigned char *ref = calloc(w * h, 1);
  unsigned char *par = calloc(w * h, 1);
  if (!src || !ref || !par) {
    free(src);
    free(ref);
    free(par);
    return 0;
  }

  double kernel[FILTER_HEIGHT][FILTER_WIDTH];
  for (int i = 0; i < FILTER_HEIGHT; ++i)
    for (int j = 0; j < FILTER_WIDTH; ++j)
      kernel[i][j] = (rand() % 21 - 10) / 10.0; // от -1.0 до 1.0

  double sf = factor, sb = bias;
  factor = 1.0;
  bias = 0.0;

  seq_convolution(src, ref, w, h, kernel);

  par_func(src, par, w, h, kernel);

  factor = sf;
  bias = sb;

  int pass = images_equal(ref, par, w, h);

  free(src);
  free(ref);
  free(par);
  return pass;
}

// Test 1
static int test_identity() {
  unsigned char *src = make_random_image(TEST_IMG_W, TEST_IMG_H);
  unsigned char *result = calloc(TEST_IMG_W * TEST_IMG_H, 1);
  if (!src || !result) {
    free(src);
    free(result);
    return 0;
  }

  double identity[FILTER_HEIGHT][FILTER_WIDTH] = {
      {0, 0, 0}, {0, 1, 0}, {0, 0, 0}};

  double sf = factor, sb = bias;
  factor = 1.0;
  bias = 0.0;

  seq_convolution(src, result, TEST_IMG_W, TEST_IMG_H, identity);

  factor = sf;
  bias = sb;
  int pass = images_equal(src, result, TEST_IMG_W, TEST_IMG_H);

  free(src);
  free(result);
  return pass;
}

// Test 2
static int test_zero_filter() {
  unsigned char *src = make_random_image(TEST_IMG_W, TEST_IMG_H);
  unsigned char *result = calloc(TEST_IMG_W * TEST_IMG_H, 1);
  if (!src || !result) {
    free(src);
    free(result);
    return 0;
  }

  double zero[FILTER_HEIGHT][FILTER_WIDTH] = {{0}};

  double sf = factor, sb = bias;
  factor = 1.0;
  bias = 0.0;

  seq_convolution(src, result, TEST_IMG_W, TEST_IMG_H, zero);

  factor = sf;
  bias = sb;
  int pass = image_is_constant(result, TEST_IMG_W, TEST_IMG_H, 0);

  free(src);
  free(result);
  return pass;
}

// Test 3
static int test_composition() {
  const int w = 16, h = 16;
  unsigned char *src = make_random_image(w, h);
  unsigned char *tmp = calloc(w * h, 1);
  unsigned char *result = calloc(w * h, 1);
  if (!src || !tmp || !result) {
    free(src);
    free(tmp);
    free(result);
    return 0;
  }

  double sr[FILTER_HEIGHT][FILTER_WIDTH] = {{0, 0, 0}, {0, 0, 1}, {0, 0, 0}};
  double sl[FILTER_HEIGHT][FILTER_WIDTH] = {{0, 0, 0}, {1, 0, 0}, {0, 0, 0}};

  double sf = factor, sb = bias;
  factor = 1.0;
  bias = 0.0;

  seq_convolution(src, tmp, w, h, sr);
  seq_convolution(tmp, result, w, h, sl);

  factor = sf;
  bias = sb;
  int pass = images_equal(src, result, w, h);

  free(src);
  free(tmp);
  free(result);
  return pass;
}

// Test 4
static int test_zero_padding() {
  const int w = 32, h = 32;
  unsigned char *src = make_random_image(w, h);
  unsigned char *result = calloc(w * h, 1);
  if (!src || !result) {
    free(src);
    free(result);
    return 0;
  }

  double pf[FILTER_HEIGHT][FILTER_WIDTH] = {{0, 0, 0}, {0, 0.5, 0}, {0, 0, 0}};

  double sf = factor, sb = bias;
  factor = 1.0;
  bias = 0.0;

  seq_convolution(src, result, w, h, pf);

  factor = sf;
  bias = sb;

  int pass = 1;
  for (int i = 0; i < w * h && pass; ++i) {
    int expected = (int)(src[i] * 0.5 + 0.5);
    if (abs((int)result[i] - expected) > 1)
      pass = 0;
  }

  free(src);
  free(result);
  return pass;
}

// Test 5
static int test_parallel_pixel() {
  return test_parallel_vs_seq(pixel_convolution, "pixel", 128, 128);
}

// Test 6
static int test_parallel_row() {
  return test_parallel_vs_seq(row_convolution, "row", 128, 128);
}

// Test 7
static int test_parallel_col() {
  return test_parallel_vs_seq(column_convolution, "col", 128, 128);
}

// Test 8
static int test_parallel_tile() {
  const int w = 128, h = 128;
  unsigned char *src = make_random_image(w, h);
  unsigned char *ref = calloc(w * h, 1);
  unsigned char *par = calloc(w * h, 1);
  if (!src || !ref || !par) {
    free(src);
    free(ref);
    free(par);
    return 0;
  }

  double kernel[FILTER_HEIGHT][FILTER_WIDTH];
  for (int i = 0; i < FILTER_HEIGHT; ++i)
    for (int j = 0; j < FILTER_WIDTH; ++j)
      kernel[i][j] = (rand() % 21 - 10) / 10.0;

  double sf = factor, sb = bias;
  factor = 1.0;
  bias = 0.0;

  seq_convolution(src, ref, w, h, kernel);
  tile_convolution(src, par, w, h, 32, 32, kernel); // тайлы 32×32

  factor = sf;
  bias = sb;

  int pass = images_equal(ref, par, w, h);

  free(src);
  free(ref);
  free(par);
  return pass;
}

// Test 9
static int test_parallel_sizes() {
  int sizes[] = {64, 128, 256};
  int all_pass = 1;

  for (int s = 0; s < 3; ++s) {
    int w = sizes[s], h = sizes[s];
    if (!test_parallel_vs_seq(row_convolution, "row-size", w, h)) {
      all_pass = 0;
      break;
    }
  }
  return all_pass;
}

int main() {
  srand((unsigned)time(NULL));

  int passed = 0, total = 0;

  printf("Property tests:\n");

#define RUN_TEST(name, num)                                                    \
  do {                                                                         \
    total++;                                                                   \
    if (name()) {                                                              \
      printf("Test %d: Ok\n", num);                                            \
      passed++;                                                                \
    } else                                                                     \
      printf("Test %d: Error\n", num);                                         \
  } while (0)

  RUN_TEST(test_identity, 1);
  RUN_TEST(test_zero_filter, 2);
  RUN_TEST(test_composition, 3);
  RUN_TEST(test_zero_padding, 4);
  RUN_TEST(test_parallel_pixel, 5);
  RUN_TEST(test_parallel_row, 6);
  RUN_TEST(test_parallel_col, 7);
  RUN_TEST(test_parallel_tile, 8);
  RUN_TEST(test_parallel_sizes, 9);

  printf("\nResult: %d/%d tests passed\n", passed, total);
  return (passed == total) ? EXIT_SUCCESS : EXIT_FAILURE;
}