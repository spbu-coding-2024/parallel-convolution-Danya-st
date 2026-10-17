#include <dirent.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "conv.h"
#include "filters.h"
#include "pipeline.h"

#define MAX_ITEMS 32
#define MAX_NAME 32
#define MAX_RUNS 128

typedef struct {
  int count;
  char item[MAX_ITEMS][MAX_NAME];
} List;

static int list_parse(const char *s, List *l) {
  l->count = 0;
  while (*s) {
    int n = 0;
    while (s[n] && s[n] != ',')
      ++n;
    if (n == 0 || n >= MAX_NAME || l->count >= MAX_ITEMS)
      return -1;
    memcpy(l->item[l->count], s, (size_t)n);
    l->item[l->count][n] = '\0';
    ++l->count;
    s += n;
    if (*s == ',')
      ++s;
  }
  return l->count > 0 ? 0 : -1;
}

static const char *const FILTER_NAMES[] = {
    "identity", "blur3",   "gaussian3", "edge",    "sharpen3", "emboss",
    "mean",     "blur5",   "gaussian5", "edge_h5", "sharpen5", "motion9"};


static int is_known_filter(const char *name) {
  for (size_t i = 0; i < sizeof FILTER_NAMES / sizeof FILTER_NAMES[0]; ++i)
    if (strcmp(name, FILTER_NAMES[i]) == 0)
      return 1;
  return 0;
}

static int parse_mode(const char *s, ParallelMode *out) {
  static const char *const names[] = {"seq", "pixel", "row", "col", "tile"};
  for (int i = 0; i < 5; ++i)
    if (strcmp(s, names[i]) == 0) {
      *out = (ParallelMode)i;
      return 0;
    }
  return -1;
}

static int cmp_double(const void *x, const void *y) {
  const double a = *(const double *)x, b = *(const double *)y;
  return (a > b) - (a < b);
}

/* Удаляет файлы из каталога и сам каталог (без вызова shell). */
static void remove_dir(const char *path) {
  DIR *d = opendir(path);
  if (d) {
    struct dirent *e;
    while ((e = readdir(d)) != NULL) {
      if (strcmp(e->d_name, ".") == 0 || strcmp(e->d_name, "..") == 0)
        continue;
      char f[4096];
      snprintf(f, sizeof f, "%s/%s", path, e->d_name);
      unlink(f);
    }
    closedir(d);
  }
  rmdir(path);
}

static char g_out_dir[] = "/tmp/bench_stream_XXXXXX";

static void cleanup_out_dir(void) { remove_dir(g_out_dir); }

static double run_series(const char *in_dir, const char *out_dir,
                         const Filter *f, const PipelineConfig *cfg, int runs,
                         PipelineStats *rep) {
  PipelineStats all[MAX_RUNS];
  double sorted[MAX_RUNS];
  for (int i = 0; i < runs; ++i) {
    if (pipeline_run(in_dir, out_dir, f, cfg, &all[i]) != 0) {
      fprintf(stderr, "pipeline_run не удался\n");
      exit(EXIT_FAILURE);
    }
    if (all[i].written == 0 || all[i].write_failed || all[i].read_failed) {
      fprintf(stderr, "Прогон завершился с ошибками: записано %d, "
                      "ошибок чтения %d, ошибок записи %d\n",
              all[i].written, all[i].read_failed, all[i].write_failed);
      exit(EXIT_FAILURE);
    }
    sorted[i] = all[i].wall_sec;
  }
  qsort(sorted, (size_t)runs, sizeof sorted[0], cmp_double);
  const double med = (runs % 2)
                         ? sorted[runs / 2]
                         : 0.5 * (sorted[runs / 2 - 1] + sorted[runs / 2]);
  int best = 0;
  for (int i = 1; i < runs; ++i)
    if (fabs(all[i].wall_sec - med) < fabs(all[best].wall_sec - med))
      best = i;
  *rep = all[best];
  return med;
}

static void usage(const char *prog) {
  fprintf(stderr,
          "Usage: %s <input_dir> [--csv=out.csv] [--workers=1,2,4,8,12,16,20]\n"
          "  [--modes=seq,pixel,row,col,tile] [--omp-threads=2] "
          "[--filter=gaussian5]\n"
          "  [--runs=N] [--queue=N] [--tile=WxH]\n",
          prog);
}

int main(int argc, char **argv) {
  if (argc < 2) {
    usage(argv[0]);
    return EXIT_FAILURE;
  }

  const char *input_dir = argv[1];
  const char *csv_path = NULL;
  const char *filter_name = "gaussian5";
  int omp_threads = 2, runs = 5, queue_capacity = 8;
  int tile_w = 32, tile_h = 32;
  List workers_list, modes_list;
  list_parse("1,2,4,8,12,16,20", &workers_list);
  list_parse("seq,pixel,row,col,tile", &modes_list);

  for (int i = 2; i < argc; ++i) {
    const char *a = argv[i];
    int rc = 0;
    if (strncmp(a, "--csv=", 6) == 0)
      csv_path = a + 6;
    else if (strncmp(a, "--workers=", 10) == 0)
      rc = list_parse(a + 10, &workers_list);
    else if (strncmp(a, "--modes=", 8) == 0)
      rc = list_parse(a + 8, &modes_list);
    else if (strncmp(a, "--filter=", 9) == 0)
      filter_name = a + 9;
    else if (strncmp(a, "--omp-threads=", 14) == 0)
      omp_threads = atoi(a + 14);
    else if (strncmp(a, "--runs=", 7) == 0)
      runs = atoi(a + 7);
    else if (strncmp(a, "--queue=", 8) == 0)
      queue_capacity = atoi(a + 8);
    else if (strncmp(a, "--tile=", 7) == 0) {
      int n = 0;
      if (sscanf(a + 7, "%dx%d%n", &tile_w, &tile_h, &n) != 2 ||
          a[7 + n] != '\0' || tile_w < 1 || tile_h < 1)
        rc = -1;
    }
    else {
      fprintf(stderr, "Неизвестный параметр: %s\n", a);
      usage(argv[0]);
      return EXIT_FAILURE;
    }
    if (rc != 0) {
      fprintf(stderr, "Некорректное значение параметра: %s\n", a);
      return EXIT_FAILURE;
    }
  }
  if (omp_threads < 1 || runs < 1 || runs > MAX_RUNS || queue_capacity < 1) {
    fprintf(stderr, "Некорректные --omp-threads / --runs / --queue\n");
    return EXIT_FAILURE;
  }

  if (!is_known_filter(filter_name)) {
    fprintf(stderr, "Неизвестный фильтр: %s\n", filter_name);
    return EXIT_FAILURE;
  }
  Filter filter = get_filter(filter_name);
  ParallelMode modes[MAX_ITEMS];
  int workers[MAX_ITEMS];
  for (int i = 0; i < modes_list.count; ++i)
    if (parse_mode(modes_list.item[i], &modes[i]) != 0) {
      fprintf(stderr, "Неизвестный режим: %s\n", modes_list.item[i]);
      return EXIT_FAILURE;
    }
  for (int i = 0; i < workers_list.count; ++i) {
    workers[i] = atoi(workers_list.item[i]);
    if (workers[i] < 1) {
      fprintf(stderr, "Некорректное число воркеров: %s\n",
              workers_list.item[i]);
      return EXIT_FAILURE;
    }
  }

  if (!mkdtemp(g_out_dir)) {
    perror("mkdtemp");
    return EXIT_FAILURE;
  }
  atexit(cleanup_out_dir);
  const char *out_dir = g_out_dir;

  FILE *csv = NULL;
  if (csv_path) {
    csv = fopen(csv_path, "w");
    if (!csv) {
      fprintf(stderr, "Не удалось открыть %s для записи\n", csv_path);
      return EXIT_FAILURE;
    }
    fprintf(csv, "workers,mode,omp_threads,filter,images,time_sec,"
                 "baseline_sec,speedup,read_busy,conv_busy,write_busy,"
                 "reader_blocked,workers_starved,workers_blocked,"
                 "writer_starved,read_q_peak,proc_q_peak\n");
  }

  printf("Бенчмарк конвейера: %s, фильтр %s, запусков: %d, логических "
         "процессоров: %d\n",
         input_dir, filter_name, runs, omp_get_num_procs());

  PipelineConfig base_cfg;
  base_cfg.num_workers = 1;
  base_cfg.queue_capacity = queue_capacity;
  base_cfg.mode = MODE_SEQ;
  base_cfg.tile_w = tile_w;
  base_cfg.tile_h = tile_h;
  base_cfg.omp_threads = 1;

  PipelineStats rep;
  run_series(input_dir, out_dir, &filter, &base_cfg, 1, &rep); /* прогрев */
  const double baseline =
      run_series(input_dir, out_dir, &filter, &base_cfg, runs, &rep);
  printf("Baseline (1 воркер, seq): %.3f s, изображений: %d\n"
         "  стадии: чтение %.3f s, свёртка %.3f s, запись %.3f s\n\n",
         baseline, rep.written, rep.read_busy_sec, rep.conv_busy_sec,
         rep.write_busy_sec);

  for (int m = 0; m < modes_list.count; ++m)
    for (int w = 0; w < workers_list.count; ++w) {
      PipelineConfig cfg;
      cfg.num_workers = workers[w];
      cfg.queue_capacity = queue_capacity;
      cfg.mode = modes[m];
      cfg.omp_threads = omp_threads;
      cfg.tile_w = tile_w;
      cfg.tile_h = tile_h;

      const double t = run_series(input_dir, out_dir, &filter, &cfg, runs,
                                  &rep);
      const double speedup = baseline / t;
      printf("workers=%-2d mode=%-5s: %.3f s, speedup x%.2f | чтение %.3f, "
             "свёртка %.3f, запись %.3f | ждут: читатель %.3f, воркеры "
             "%.3f, писатель %.3f\n",
             workers[w], modes_list.item[m], t, speedup, rep.read_busy_sec,
             rep.conv_busy_sec, rep.write_busy_sec, rep.reader_blocked_sec,
             rep.workers_starved_sec, rep.writer_starved_sec);
      fflush(stdout);
      if (csv) {
        fprintf(csv, "%d,%s,%d,%s,%d,%.6f,%.6f,%.4f,%.6f,%.6f,%.6f,%.6f,"
                     "%.6f,%.6f,%.6f,%d,%d\n",
                workers[w], modes_list.item[m],
                modes[m] == MODE_SEQ ? 1 : omp_threads, filter_name,
                rep.written, t, baseline, speedup, rep.read_busy_sec,
                rep.conv_busy_sec, rep.write_busy_sec, rep.reader_blocked_sec,
                rep.workers_starved_sec, rep.workers_blocked_sec,
                rep.writer_starved_sec, rep.read_q_peak, rep.proc_q_peak);
        fflush(csv);
      }
    }

  if (csv)
    fclose(csv);
  return EXIT_SUCCESS;
}