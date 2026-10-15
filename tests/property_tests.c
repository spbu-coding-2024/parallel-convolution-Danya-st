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

  conv_seq(src, result, TEST_IMG_W, TEST_IMG_H, &identity);
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

  conv_seq(src, result, TEST_IMG_W, TEST_IMG_H, &zero);
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

  conv_seq(src, tmp, w, h, &sr);
  conv_seq(tmp, result, w, h, &sl);
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

    conv_seq(src, small_res, w, h, &small);
    conv_seq(src, big_res, w, h, &big);
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
  conv_seq(src, result, 4, 4, &mean);
  return images_equal(expected, result, 4, 4);
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

  printf("\nResult: %d/%d tests passed\n", passed, total);
  return (passed == total) ? EXIT_SUCCESS : EXIT_FAILURE;
}