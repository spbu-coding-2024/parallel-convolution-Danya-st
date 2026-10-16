#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "conv.h"
#include "filters.h"
#include "pipeline.h"

// Что изменено по сравнению со старым stream_main.c:
//  - ИСПРАВЛЕНО (дубликаты кода): очередь, потоки читателя/воркеров/писателя
//    и копии свёртки (seq/pixel/row/col/tile, глобальные factor/bias,
//    FILTER_WIDTH = 3) удалены - всё в lib/pipeline.c и lib/conv.c, общих с
//    bench_stream.c. Это же исправляет фильтры 5x5/9x9, factor/bias и
//    переполнение w * h;
//  - ИСПРАВЛЕНО (лимит 16 воркеров): раньше pthread_t workers[16] без проверки
//    num_workers, 17+ воркеров портили стек;
//  - ИСПРАВЛЕНО (зависание): отсутствующая входная папка давала вечное
//    ожидание; теперь ошибка и код возврата 1;
//  - ИСПРАВЛЕНО (утечка): "таблетка" завершения больше не теряется - очереди
//    закрываются явно;
//  - ИСПРАВЛЕНО (jpg): обрабатываются .png/.jpg/.jpeg/.bmp в любом регистре;
//    результат всегда PNG, поэтому photo.jpg сохраняется как photo.jpg.png;
//  - ИСПРАВЛЕНО: результат записи проверяется, ошибки чтения/записи
//    считаются, неверные --parallel/--tile/число воркеров дают ошибку (раньше
//    молча превращались в seq/мусор/1 воркер);
//  - новые параметры --omp-threads и --queue; в конце печатается время по
//    стадиям (чтение / свёртка / запись) и простои на очередях.

static int parse_positive(const char *s, int *out) {
  char *end = NULL;
  long v = strtol(s, &end, 10);
  if (end == s || *end != '\0' || v < 1 || v > 100000)
    return -1;
  *out = (int)v;
  return 0;
}

int main(int argc, char *argv[]) {
  if (argc < 4) {
    fprintf(stderr,
            "usage: %s <input_dir> <output_dir> <num_workers> [--filter=name] "
            "[--parallel=mode] [--tile=WxH] [--omp-threads=N] [--queue=N]\n",
            argv[0]);
    printf("Filters: identity, blur3, blur5, gaussian3, gaussian5, edge, "
           "edge_h5, sharpen3, sharpen5, emboss, mean, motion9\n");
    printf("Modes: seq, pixel, row, col, tile (default: seq)\n");
    return EXIT_FAILURE;
  }

  const char *input_dir = argv[1], *output_dir = argv[2];
  PipelineConfig cfg;
  memset(&cfg, 0, sizeof cfg);
  cfg.queue_capacity = 8;
  cfg.mode = MODE_SEQ;
  cfg.tile_w = 32;
  cfg.tile_h = 32;
  cfg.omp_threads = 0;
  if (parse_positive(argv[3], &cfg.num_workers) != 0) {
    fprintf(stderr, "Некорректное число воркеров: %s\n", argv[3]);
    return EXIT_FAILURE;
  }

  // Парсинг аргументов
  char *filter_name = "identity";

  for (int i = 4; i < argc; ++i) {
    if (strncmp(argv[i], "--filter=", 9) == 0) {
      filter_name = argv[i] + 9;
    } else if (strncmp(argv[i], "--parallel=", 11) == 0) {
      const char *m = argv[i] + 11;
      if (strcmp(m, "pixel") == 0)
        cfg.mode = MODE_PIXEL;
      else if (strcmp(m, "row") == 0)
        cfg.mode = MODE_ROW;
      else if (strcmp(m, "col") == 0)
        cfg.mode = MODE_COL;
      else if (strcmp(m, "tile") == 0)
        cfg.mode = MODE_TILE;
      else if (strcmp(m, "seq") == 0)
        cfg.mode = MODE_SEQ;
      else {
        fprintf(stderr, "Неизвестный режим: %s\n", m);
        return EXIT_FAILURE;
      }
    } else if (strncmp(argv[i], "--tile=", 7) == 0) {
      int n = 0;
      if (sscanf(argv[i] + 7, "%dx%d%n", &cfg.tile_w, &cfg.tile_h, &n) != 2 ||
          argv[i][7 + n] != '\0' || cfg.tile_w < 1 || cfg.tile_h < 1) {
        fprintf(stderr, "Некорректный --tile (нужно WxH, W,H >= 1)\n");
        return EXIT_FAILURE;
      }
    } else if (strncmp(argv[i], "--omp-threads=", 14) == 0) {
      if (parse_positive(argv[i] + 14, &cfg.omp_threads) != 0) {
        fprintf(stderr, "Некорректный --omp-threads: %s\n", argv[i] + 14);
        return EXIT_FAILURE;
      }
    } else if (strncmp(argv[i], "--queue=", 8) == 0) {
      if (parse_positive(argv[i] + 8, &cfg.queue_capacity) != 0) {
        fprintf(stderr, "Некорректный --queue: %s\n", argv[i] + 8);
        return EXIT_FAILURE;
      }
    } else {
      fprintf(stderr, "Неизвестный параметр: %s\n", argv[i]);
      return EXIT_FAILURE;
    }
  }

  Filter filter = get_filter(filter_name);

  PipelineStats st;
  if (pipeline_run(input_dir, output_dir, &filter, &cfg, &st) != 0)
    return EXIT_FAILURE;

  printf("Processed %d images in %.3f s (read failed: %d, write failed: %d)\n",
         st.written, st.wall_sec, st.read_failed, st.write_failed);
  printf("Время стадий (CPU, сумма по потокам): чтение %.3f s, свёртка %.3f s, "
         "запись %.3f s\n",
         st.read_busy_sec, st.conv_busy_sec, st.write_busy_sec);
  printf("Простои: читатель ждёт место %.3f s, воркеры ждут данные %.3f s, "
         "писатель ждёт данные %.3f s\n",
         st.reader_blocked_sec, st.workers_starved_sec, st.writer_starved_sec);
  return (st.read_failed || st.write_failed) ? EXIT_FAILURE : EXIT_SUCCESS;
}