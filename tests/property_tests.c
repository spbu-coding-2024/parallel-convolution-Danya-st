#include <omp.h>
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

static Filter make_random_filter(int size) {
  Filter f = make_filter(size);
  for (int i = 0; i < size; ++i)
    for (int j = 0; j < size; ++j)
      f.kernel[i][j] = (rand() % 21 - 10) / 10.0; // от -1.0 до 1.0
  f.factor = 0.5 + (rand() % 100) / 100.0;
  f.bias = rand() % 41 - 20;
  return f;
}

typedef void (*ConvFunc)(const unsigned char *, unsigned char *, int, int,
                         const Filter *);


static int test_parallel_vs_seq(ConvFunc par_func, int w, int h,
                                int filter_size) {
  unsigned char *src = make_random_image(w, h);
  unsigned char *ref = calloc((size_t)w * h, 1);
  unsigned char *par = malloc((size_t)w * h);
  if (!src || !ref || !par) {
    free(src);
    free(ref);
    free(par);
    return 0;
  }
  memset(par, 0xA5, (size_t)w * h);

  Filter f = make_random_filter(filter_size);
  seq_convolution(src, ref, w, h, &f);
  par_func(src, par, w, h, &f);

  int pass = images_equal(ref, par, w, h);
  if (!pass)
    fprintf(stderr, "  расхождение: %dx%d, фильтр %dx%d\n", w, h, filter_size,
            filter_size);

  free(src);
  free(ref);
  free(par);
  return pass;
}

static int test_parallel_all(ConvFunc par_func) {
  const int sizes[][2] = {{64, 64}, {128, 128}, {256, 256}, {97, 61},
                          {33, 131}};
  const int filter_sizes[] = {3, 5, 9};
  for (int s = 0; s < 5; ++s)
    for (int k = 0; k < 3; ++k)
      if (!test_parallel_vs_seq(par_func, sizes[s][0], sizes[s][1],
                                filter_sizes[k]))
        return 0;
  return 1;
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
  return test_parallel_all(pixel_convolution);
}

// Test 7
static int test_parallel_row() {
  return test_parallel_all(row_convolution);
}

// Test 8
static int test_parallel_col() {
  return test_parallel_all(column_convolution);
}

// Test 9
static int test_parallel_tile() {
  const int w = 97, h = 61;
  const int tiles[][2] = {{1, 1},   {7, 13},  {8, 8},  {32, 32}, {16, 256},
                          {256, 16}, {1000, 1000}, {0, 0}, {-5, 3}};
  unsigned char *src = make_random_image(w, h);
  unsigned char *ref = calloc(w * h, 1);
  unsigned char *par = calloc(w * h, 1);
  if (!src || !ref || !par) {
    free(src);
    free(ref);
    free(par);
    return 0;
  }

  Filter f = make_random_filter(5);
  seq_convolution(src, ref, w, h, &f);

  int pass = 1;
  for (int t = 0; t < 9 && pass; ++t) {
    memset(par, 0xA5, w * h);
    tile_convolution(src, par, w, h, tiles[t][0], tiles[t][1], &f);
    pass = images_equal(ref, par, w, h);
    if (!pass)
      fprintf(stderr, "  плитка %dx%d\n", tiles[t][0], tiles[t][1]);
  }

  free(src);
  free(ref);
  free(par);
  return pass;
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

  printf("\nResult: %d/%d tests passed\n", passed, total);
  return (passed == total) ? EXIT_SUCCESS : EXIT_FAILURE;
}