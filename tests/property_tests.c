#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define TEST_IMG_W 64
#define TEST_IMG_H 64

#include "conv.h"
#include "filters.h"

static inline unsigned char *make_random_image(int w, int h) {
  unsigned char *img = malloc((size_t)w * h);
  if (!img)
    return NULL;
  for (int i = 0; i < w * h; ++i)
    img[i] = rand() % 256;
  return img;
}

// ИСПРАВЛЕНО: сравнение точное (раньше допускалась разница в 1).
static inline int images_equal(const unsigned char *a, const unsigned char *b,
                               int w, int h) {
  for (int i = 0; i < w * h; ++i)
    if (a[i] != b[i])
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

static Filter make_filter(int size) {
  Filter f;
  memset(&f, 0, sizeof f);
  f.size = size;
  f.factor = 1.0;
  f.bias = 0.0;
  f.name = "test";
  return f;
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

  Filter identity = make_filter(3);
  identity.kernel[1][1] = 1;

  seq_convolution(src, result, TEST_IMG_W, TEST_IMG_H, &identity);
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

  Filter zero = make_filter(3);

  seq_convolution(src, result, TEST_IMG_W, TEST_IMG_H, &zero);
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

  Filter sr = make_filter(3);
  Filter sl = make_filter(3);
  sr.kernel[1][2] = 1;
  sl.kernel[1][0] = 1;

  seq_convolution(src, tmp, w, h, &sr);
  seq_convolution(tmp, result, w, h, &sl);
  int pass = images_equal(src, result, w, h);

  free(src);
  free(tmp);
  free(result);
  return pass;
}

// Test 4
static int test_zero_padding() {
  const int w = 32, h = 32;
  const char *names[] = {"gaussian3", "sharpen3", "emboss"};
  int pass = 1;

  for (int n = 0; n < 3 && pass; ++n) {
    unsigned char *src = make_random_image(w, h);
    unsigned char *small_res = calloc(w * h, 1);
    unsigned char *big_res = calloc(w * h, 1);
    if (!src || !small_res || !big_res) {
      free(src);
      free(small_res);
      free(big_res);
      return 0;
    }

    Filter small = get_filter(names[n]);
    Filter big = make_filter(5);
    big.factor = small.factor;
    big.bias = small.bias;
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
        big.kernel[i + 1][j + 1] = small.kernel[i][j];

    seq_convolution(src, small_res, w, h, &small);
    seq_convolution(src, big_res, w, h, &big);
    pass = images_equal(small_res, big_res, w, h);

    free(src);
    free(small_res);
    free(big_res);
  }
  return pass;
}

// Test 5
static int test_golden() {
  unsigned char src[16], result[16];
  for (int i = 0; i < 16; ++i)
    src[i] = (unsigned char)(i * 10);
  const unsigned char expected[16] = {67, 63, 73,  70, 53, 50, 60, 57,
                                      93, 90, 100, 97, 80, 77, 87, 83};

  Filter mean = get_filter("mean");
  seq_convolution(src, result, 4, 4, &mean);
  return images_equal(expected, result, 4, 4);
}

// Test 6
static int test_parallel_pixel() {
  return test_parallel_vs_seq(pixel_convolution, "pixel", 128, 128);
}

// Test 7
static int test_parallel_row() {
  return test_parallel_vs_seq(row_convolution, "row", 128, 128);
}

// Test 8
static int test_parallel_col() {
  return test_parallel_vs_seq(column_convolution, "col", 128, 128);
}

// Test 9
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

// Test 10
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
  RUN_TEST(test_golden, 5);
  RUN_TEST(test_parallel_pixel, 6);
  RUN_TEST(test_parallel_row, 7);
  RUN_TEST(test_parallel_col, 8);
  RUN_TEST(test_parallel_tile, 9);
  RUN_TEST(test_parallel_sizes, 10);

  printf("\nResult: %d/%d tests passed\n", passed, total);
  return (passed == total) ? EXIT_SUCCESS : EXIT_FAILURE;
}