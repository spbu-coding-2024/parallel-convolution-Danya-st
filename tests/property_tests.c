#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define FILTER_WIDTH 3
#define FILTER_HEIGHT 3

#include "../lib/stb_image.h"
#include "../lib/stb_image_write.h"

#define EPSILON 1e-6
#define TEST_IMG_W 64
#define TEST_IMG_H 64

double factor = 1.0;
double bias = 0.0;

static inline unsigned char clamp(double value) {
    int v = (int)(factor * value + bias);
    if (v < 0) return 0;
    if (v > 255) return 255;
    return (unsigned char)v;
}

void seq_convolution(unsigned char* src, unsigned char* result, int w, int h, 
                     double kernel[FILTER_HEIGHT][FILTER_WIDTH]) {
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            double sum = 0.0;
            for (int fy = 0; fy < FILTER_HEIGHT; ++fy) {
                for (int fx = 0; fx < FILTER_WIDTH; ++fx) {
                    int ix = (x - FILTER_WIDTH/2 + fx + w) % w;
                    int iy = (y - FILTER_HEIGHT/2 + fy + h) % h;
                    sum += src[iy * w + ix] * kernel[fy][fx];
                }
            }
            result[y * w + x] = clamp(sum);
        }
    }
}

static unsigned char* make_random_image(int w, int h) {
    unsigned char* img = malloc(w * h);
    for (int i = 0; i < w * h; ++i) img[i] = rand() % 256;
    return img;
}

static int images_equal(const unsigned char* a, const unsigned char* b, int w, int h) {
    for (int i = 0; i < w * h; ++i)
        if (abs((int)a[i] - (int)b[i]) > 1) return 0;
    return 1;
}

static int image_is_constant(const unsigned char* img, int w, int h, unsigned char value) {
    for (int i = 0; i < w * h; ++i)
        if (img[i] != value) return 0;
    return 1;
}

// Test 1
static int test_identity() {
    unsigned char* src = make_random_image(TEST_IMG_W, TEST_IMG_H);
    unsigned char* result = calloc(TEST_IMG_W * TEST_IMG_H, 1);
    if (!src || !result) { free(src); free(result); return 0; }
    
    double identity[FILTER_HEIGHT][FILTER_WIDTH] = {{0,0,0},{0,1,0},{0,0,0}};
    
    double sf = factor, sb = bias;
    factor = 1.0; bias = 0.0;
    
    seq_convolution(src, result, TEST_IMG_W, TEST_IMG_H, identity);
    
    factor = sf; bias = sb;
    int pass = images_equal(src, result, TEST_IMG_W, TEST_IMG_H);
    
    free(src); free(result);
    return pass;
}

// Test 2
static int test_zero_filter() {
    unsigned char* src = make_random_image(TEST_IMG_W, TEST_IMG_H);
    unsigned char* result = calloc(TEST_IMG_W * TEST_IMG_H, 1);
    if (!src || !result) { free(src); free(result); return 0; }
    
    double zero[FILTER_HEIGHT][FILTER_WIDTH] = {{0}};
    
    double sf = factor, sb = bias;
    factor = 1.0; bias = 0.0;
    
    seq_convolution(src, result, TEST_IMG_W, TEST_IMG_H, zero);
    
    factor = sf; bias = sb;
    int pass = image_is_constant(result, TEST_IMG_W, TEST_IMG_H, 0);
    
    free(src); free(result);
    return pass;
}

// Test 3
static int test_composition() {
    const int w = 16, h = 16;
    unsigned char* src = make_random_image(w, h);
    unsigned char* tmp = calloc(w * h, 1);
    unsigned char* result = calloc(w * h, 1);
    if (!src || !tmp || !result) { free(src); free(tmp); free(result); return 0; }
    
    double sr[FILTER_HEIGHT][FILTER_WIDTH] = {{0,0,0},{0,0,1},{0,0,0}};
    double sl[FILTER_HEIGHT][FILTER_WIDTH] = {{0,0,0},{1,0,0},{0,0,0}};
    
    double sf = factor, sb = bias;
    factor = 1.0; bias = 0.0;
    
    seq_convolution(src, tmp, w, h, sr);
    seq_convolution(tmp, result, w, h, sl);
    
    factor = sf; bias = sb;
    int pass = images_equal(src, result, w, h);
    
    free(src); free(tmp); free(result);
    return pass;
}

// Test 4
static int test_zero_padding() {
    const int w = 32, h = 32;
    unsigned char* src = make_random_image(w, h);
    unsigned char* result = calloc(w * h, 1);
    if (!src || !result) { free(src); free(result); return 0; }
    
    double pf[FILTER_HEIGHT][FILTER_WIDTH] = {{0,0,0},{0,0.5,0},{0,0,0}};
    
    double sf = factor, sb = bias;
    factor = 1.0; bias = 0.0;
    
    seq_convolution(src, result, w, h, pf);
    
    factor = sf; bias = sb;
    
    int pass = 1;
    for (int i = 0; i < w * h && pass; ++i) {
        int expected = (int)(src[i] * 0.5 + 0.5);
        if (abs((int)result[i] - expected) > 1) pass = 0;
    }
    
    free(src); free(result);
    return pass;
}

int main() {
    srand((unsigned)time(NULL));
    
    int passed = 0, total = 0;
    
    printf("Property tests:\n");
    
    #define RUN_TEST(name, num) do { \
        total++; \
        if (name()) { printf("Test %d: Ok\n", num); passed++; } \
        else printf("Test %d: Error\n", num); \
    } while(0)
    
    RUN_TEST(test_identity, 1);
    RUN_TEST(test_zero_filter, 2);
    RUN_TEST(test_composition, 3);
    RUN_TEST(test_zero_padding, 4);
    
    printf("\nResult: %d/%d tests passed\n", passed, total);
    return (passed == total) ? EXIT_SUCCESS : EXIT_FAILURE;
}