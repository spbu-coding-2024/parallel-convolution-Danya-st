#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "stb_image.h"
#include "stb_image_write.h"
#include "filters.h"
#include "conv.h"

int main(int argc, char *argv[]) {
  if (argc < 3) {
    fprintf(stderr, "usage: %s <input.png> <output> [--filter=name]\n",
            argv[0]);
    print_filters();
    return EXIT_FAILURE;
  }
  char *filter_name = "identity";

  for (int i = 3; i < argc; ++i) {
    if (strncmp(argv[i], "--filter=", 9) == 0)
      filter_name = argv[i] + 9;
  }

  Filter filter = get_filter(filter_name);

  int w, h, channels;
  unsigned char *src =
      stbi_load(argv[1], &w, &h, &channels, 1); // 1 = grayscale
  if (!src) {
    fprintf(stderr, "Ошибка загрузки: %s\n", stbi_failure_reason());
    return 1;
  }

  unsigned char *result = malloc(w * h * sizeof(unsigned char));
  if (!result) {
    fprintf(stderr, "Ошибка выделения памяти\n");
    stbi_image_free(src);
    return 1;
  }

  clock_t start_time = clock();
  seq_convolution(src, result, w, h, &filter);
  clock_t end_time = clock();
  double elapsed = (double)(end_time - start_time) / CLOCKS_PER_SEC;
  printf("время выполнения: %4f sec\n", elapsed);

  stbi_write_png(argv[2], w, h, 1, result, w);

  free(result);
  stbi_image_free(src);
  return 0;
}
