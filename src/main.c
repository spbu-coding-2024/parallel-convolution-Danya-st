#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#include "stb_image.h"
#include "stb_image_write.h"
#include "filters.h"
#include "conv.h"

int main(int argc, char *argv[]) {
  if (argc < 3) {
    fprintf(
        stderr,
        "usage: %s <input.png> <output> [--filter=name] [--parallel=mode]\n",
        argv[0]);
    printf("  3×3: identity, blur3, gaussian3, edge, sharpen3, emboss, mean\n");
    printf("  5×5: blur5, gaussian5, edge_h5, sharpen5\n");
    printf("  9×9: motion9\n");
    printf("Parallel modes: seq, pixel, row, col, tile (default: seq)\n");
    return EXIT_FAILURE;
  }
  char *filter_name = "identity";
  ParallelMode mode = MODE_SEQ;
  int tile_w = 32, tile_h = 32;

  for (int i = 3; i < argc; ++i) {
    if (strncmp(argv[i], "--filter=", 9) == 0)
      filter_name = argv[i] + 9;
    else if (strncmp(argv[i], "--parallel=", 11) == 0) {
      const char *m = argv[i] + 11;
      if (strcmp(m, "pixel") == 0)
        mode = MODE_PIXEL;
      else if (strcmp(m, "row") == 0)
        mode = MODE_ROW;
      else if (strcmp(m, "col") == 0)
        mode = MODE_COL;
      else if (strcmp(m, "tile") == 0)
        mode = MODE_TILE;
      else
        mode = MODE_SEQ;
    } else if (strncmp(argv[i], "--tile=", 7) == 0) {
      sscanf(argv[i] + 7, "%dx%d", &tile_w, &tile_h);
    }
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

  double start_time = omp_get_wtime();
  switch (mode) {
  case MODE_PIXEL:
    pixel_convolution(src, result, w, h, &filter);
    break;
  case MODE_ROW:
    row_convolution(src, result, w, h, &filter);
    break;
  case MODE_COL:
    column_convolution(src, result, w, h, &filter);
    break;
  case MODE_TILE:
    tile_convolution(src, result, w, h, tile_w, tile_h, &filter);
    break;
  default:
    seq_convolution(src, result, w, h, &filter);
  }
  double elapsed = omp_get_wtime() - start_time;
  printf("время выполнения: %4f sec\n", elapsed);

  stbi_write_png(argv[2], w, h, 1, result, w);

  free(result);
  stbi_image_free(src);
  return 0;
}