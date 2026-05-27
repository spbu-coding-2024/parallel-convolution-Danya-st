#define STB_IMAGE_IMPLEMENTATION
#include "../lib/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../lib/stb_image_write.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include <sys/stat.h>
#include <dirent.h>
#include <unistd.h>
#include "../lib/filters.h"
#include <omp.h>

#define QUEUE_SIZE 32
#define MAX_PATH 512
#define MAX_IMAGES 100

typedef enum { MODE_SEQ, MODE_PIXEL, MODE_ROW, MODE_COL, MODE_TILE } ConvMode;

typedef struct {
  unsigned char *data;
  int w, h;
  char filename[MAX_PATH];
  int done;
} ImageTask;

typedef struct {
  ImageTask *items[QUEUE_SIZE];
  int head, tail, count;
  pthread_mutex_t mutex;
  pthread_cond_t not_empty, not_full;
  int shutdown;
} TaskQueue;

void queue_init(TaskQueue *q) {
  q->head = q->tail = q->count = 0;
  q->shutdown = 0;
  pthread_mutex_init(&q->mutex, NULL);
  pthread_cond_init(&q->not_empty, NULL);
  pthread_cond_init(&q->not_full, NULL);
}

void queue_destroy(TaskQueue *q) {
  pthread_mutex_destroy(&q->mutex);
  pthread_cond_destroy(&q->not_empty);
  pthread_cond_destroy(&q->not_full);
}

int queue_push(TaskQueue *q, ImageTask *task) {
  pthread_mutex_lock(&q->mutex);
  while (q->count == QUEUE_SIZE && !q->shutdown)
    pthread_cond_wait(&q->not_full, &q->mutex);
  if (q->shutdown) {
    pthread_mutex_unlock(&q->mutex);
    return -1;
  }
  q->items[q->tail] = task;
  q->tail = (q->tail + 1) % QUEUE_SIZE;
  q->count++;
  pthread_cond_signal(&q->not_empty);
  pthread_mutex_unlock(&q->mutex);
  return 0;
}

ImageTask *queue_pop(TaskQueue *q) {
  pthread_mutex_lock(&q->mutex);
  while (q->count == 0 && !q->shutdown)
    pthread_cond_wait(&q->not_empty, &q->mutex);
  if (q->count == 0) {
    pthread_mutex_unlock(&q->mutex);
    return NULL;
  }
  ImageTask *task = q->items[q->head];
  q->head = (q->head + 1) % QUEUE_SIZE;
  q->count--;
  pthread_cond_signal(&q->not_full);
  pthread_mutex_unlock(&q->mutex);
  return task;
}

void queue_shutdown(TaskQueue *q) {
  pthread_mutex_lock(&q->mutex);
  q->shutdown = 1;
  pthread_cond_broadcast(&q->not_empty);
  pthread_cond_broadcast(&q->not_full);
  pthread_mutex_unlock(&q->mutex);
}

double factor = 1.0, bias = 0.0;

static inline unsigned char clamp(double value) {
  int v = (int)(factor * value + bias);
  return (v < 0) ? 0 : (v > 255) ? 255 : (unsigned char)v;
}

static void conv_seq(unsigned char *src, unsigned char *dst, int w, int h,
                     Filter *f) {
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      double sum = 0.0;
      for (int fy = 0; fy < f->size; ++fy)
        for (int fx = 0; fx < f->size; ++fx) {
          int ix = (x - f->size / 2 + fx + w) % w;
          int iy = (y - f->size / 2 + fy + h) % h;
          sum += src[iy * w + ix] * f->kernel[fy][fx];
        }
      dst[y * w + x] = clamp(sum);
    }
}

static void conv_pixel(unsigned char *src, unsigned char *dst, int w, int h,
                       Filter *f) {
#pragma omp parallel for
  for (long idx = 0; idx < (long)w * h; ++idx) {
    int x = idx % w, y = idx / w;
    double sum = 0.0;
    for (int fy = 0; fy < f->size; ++fy)
      for (int fx = 0; fx < f->size; ++fx) {
        int ix = (x - f->size / 2 + fx + w) % w;
        int iy = (y - f->size / 2 + fy + h) % h;
        sum += src[iy * w + ix] * f->kernel[fy][fx];
      }
    dst[idx] = clamp(sum);
  }
}

static void conv_row(unsigned char *src, unsigned char *dst, int w, int h,
                     Filter *f) {
#pragma omp parallel for
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      double sum = 0.0;
      for (int fy = 0; fy < f->size; ++fy)
        for (int fx = 0; fx < f->size; ++fx) {
          int ix = (x - f->size / 2 + fx + w) % w;
          int iy = (y - f->size / 2 + fy + h) % h;
          sum += src[iy * w + ix] * f->kernel[fy][fx];
        }
      dst[y * w + x] = clamp(sum);
    }
}

static void conv_col(unsigned char *src, unsigned char *dst, int w, int h,
                     Filter *f) {
#pragma omp parallel for
  for (int x = 0; x < w; ++x)
    for (int y = 0; y < h; ++y) {
      double sum = 0.0;
      for (int fy = 0; fy < f->size; ++fy)
        for (int fx = 0; fx < f->size; ++fx) {
          int ix = (x - f->size / 2 + fx + w) % w;
          int iy = (y - f->size / 2 + fy + h) % h;
          sum += src[iy * w + ix] * f->kernel[fy][fx];
        }
      dst[y * w + x] = clamp(sum);
    }
}

static void conv_tile(unsigned char *src, unsigned char *dst, int w, int h,
                      Filter *f) {
  int tw = 32, th = 32;
  int tx = (w + tw - 1) / tw, ty = (h + th - 1) / th;
#pragma omp parallel for
  for (int t = 0; t < tx * ty; ++t) {
    int cx = t % tx, cy = t / tx;
    int x0 = cx * tw, y0 = cy * th;
    int x1 = (x0 + tw < w) ? x0 + tw : w;
    int y1 = (y0 + th < h) ? y0 + th : h;
    for (int y = y0; y < y1; ++y)
      for (int x = x0; x < x1; ++x) {
        double sum = 0.0;
        for (int fy = 0; fy < f->size; ++fy)
          for (int fx = 0; fx < f->size; ++fx) {
            int ix = (x - f->size / 2 + fx + w) % w;
            int iy = (y - f->size / 2 + fy + h) % h;
            sum += src[iy * w + ix] * f->kernel[fy][fx];
          }
        dst[y * w + x] = clamp(sum);
      }
  }
}

typedef struct {
  TaskQueue *read_q, *proc_q;
  const char *input_dir, *output_dir;
  Filter *filter;
  ConvMode mode;
  int omp_threads;
  int processed;
  pthread_mutex_t count_mutex;
} PipelineArgs;

void *reader_thread(void *arg) {
  PipelineArgs *args = (PipelineArgs *)arg;
  DIR *dir = opendir(args->input_dir);
  if (!dir)
    return NULL;

  struct dirent *entry;
  while ((entry = readdir(dir)) != NULL) {
    const char *name = entry->d_name;
    int len = strlen(name);
    if (len < 4 || strcmp(name + len - 4, ".png") != 0)
      continue;

    char path[MAX_PATH];
    snprintf(path, sizeof(path), "%s/%s", args->input_dir, name);
    struct stat st;
    if (stat(path, &st) != 0 || !S_ISREG(st.st_mode))
      continue;
    int w, h, ch;
    unsigned char *src = stbi_load(path, &w, &h, &ch, 1);
    if (!src)
      continue;

    ImageTask *task = malloc(sizeof(ImageTask));
    task->data = src;
    task->w = w;
    task->h = h;
    strncpy(task->filename, name, MAX_PATH - 1);
    task->done = 0;

    if (queue_push(args->read_q, task) < 0) {
      free(task->data);
      free(task);
      break;
    }
  }
  closedir(dir);

  ImageTask *done = calloc(1, sizeof(ImageTask));
  done->done = 1;
  queue_push(args->read_q, done);
  return NULL;
}

void *worker_thread(void *arg) {
  PipelineArgs *args = (PipelineArgs *)arg;
  omp_set_num_threads(args->omp_threads);

  while (1) {
    ImageTask *task = queue_pop(args->read_q);
    if (!task)
      break;
    if (task->done) {
      queue_push(args->read_q, task);
      break;
    }

    unsigned char *result = malloc(task->w * task->h);
    if (!result) {
      free(task->data);
      free(task);
      continue;
    }

    switch (args->mode) {
    case MODE_PIXEL:
      conv_pixel(task->data, result, task->w, task->h, args->filter);
      break;
    case MODE_ROW:
      conv_row(task->data, result, task->w, task->h, args->filter);
      break;
    case MODE_COL:
      conv_col(task->data, result, task->w, task->h, args->filter);
      break;
    case MODE_TILE:
      conv_tile(task->data, result, task->w, task->h, args->filter);
      break;
    default:
      conv_seq(task->data, result, task->w, task->h, args->filter);
    }

    free(task->data);
    task->data = result;

    if (queue_push(args->proc_q, task) < 0) {
      free(task->data);
      free(task);
      break;
    }

    pthread_mutex_lock(&args->count_mutex);
    args->processed++;
    pthread_mutex_unlock(&args->count_mutex);
  }
  return NULL;
}

void *writer_thread(void *arg) {
  PipelineArgs *args = (PipelineArgs *)arg;

  while (1) {
    ImageTask *task = queue_pop(args->proc_q);
    if (!task)
      break;
    if (task->done) {
      free(task);
      break;
    }

    char out_path[MAX_PATH];
    snprintf(out_path, sizeof(out_path), "%s/%s", args->output_dir,
             task->filename);
    stbi_write_png(out_path, task->w, task->h, 1, task->data, task->w);

    free(task->data);
    free(task);
  }
  return NULL;
}

double run_benchmark(const char *input_dir, const char *output_dir,
                     int num_workers, ConvMode mode, Filter *filter,
                     int omp_threads, int runs) {
  double times[100];

  for (int run = 0; run < runs; ++run) {

    char tmp_in[256], tmp_out[256];
    snprintf(tmp_in, sizeof(tmp_in), "/tmp/bench_in_%d", run);
    snprintf(tmp_out, sizeof(tmp_out), "/tmp/bench_out_%d", run);
    mkdir(tmp_in, 0755);
    mkdir(tmp_out, 0755);

    DIR *dir = opendir(input_dir);
    struct dirent *entry;
    int img_count = 0;
    while ((entry = readdir(dir)) != NULL && img_count < MAX_IMAGES) {
      const char *name = entry->d_name;
      int len = strlen(name);
      if (len < 4 || strcmp(name + len - 4, ".png") != 0)
        continue;

      char src[512], dst[512];
      snprintf(src, sizeof(src), "%s/%s", input_dir, name);
      snprintf(dst, sizeof(dst), "%s/%s", tmp_in, name);
      struct stat st;
      if (stat(src, &st) != 0 || !S_ISREG(st.st_mode))
        continue;
      int w, h, ch;
      unsigned char *data = stbi_load(src, &w, &h, &ch, 1);
      if (data) {
        stbi_write_png(dst, w, h, 1, data, w);
        stbi_image_free(data);
        img_count++;
      }
    }
    closedir(dir);

    TaskQueue read_q, proc_q;
    queue_init(&read_q);
    queue_init(&proc_q);

    PipelineArgs args = {.read_q = &read_q,
                         .proc_q = &proc_q,
                         .input_dir = tmp_in,
                         .output_dir = tmp_out,
                         .filter = filter,
                         .mode = mode,
                         .omp_threads = omp_threads,
                         .processed = 0};
    pthread_mutex_init(&args.count_mutex, NULL);

    pthread_t reader, writer, workers[16];
    double start = omp_get_wtime();
    pthread_create(&reader, NULL, reader_thread, &args);
    pthread_create(&writer, NULL, writer_thread, &args);
    for (int i = 0; i < num_workers; ++i)
      pthread_create(&workers[i], NULL, worker_thread, &args);

    pthread_join(reader, NULL);
    for (int i = 0; i < num_workers; ++i)
      pthread_join(workers[i], NULL);

    ImageTask *done = calloc(1, sizeof(ImageTask));
    done->done = 1;
    queue_push(&proc_q, done);
    pthread_join(writer, NULL);

    double end = omp_get_wtime();
    times[run] = (end - start);

    // Очистка
    queue_shutdown(&read_q);
    queue_shutdown(&proc_q);
    queue_destroy(&read_q);
    queue_destroy(&proc_q);
    pthread_mutex_destroy(&args.count_mutex);

    char cmd[1024];
    snprintf(cmd, sizeof(cmd), "rm -rf %s", tmp_in);
    system(cmd);
    snprintf(cmd, sizeof(cmd), "rm -rf %s", tmp_out);
    system(cmd);
  }

  double sum = 0;
  for (int i = 0; i < runs; ++i)
    sum += times[i];
  return sum / runs;
}

int main(int argc, char **argv) {
  if (argc < 5) {
    fprintf(
        stderr,
        "Usage: %s <input_dir> <output_dir> <workers> <mode> [--csv=out.csv]\n",
        argv[0]);
    fprintf(stderr, "Modes: seq, pixel, row, col, tile\n");
    return 1;
  }

  const char *input_dir = argv[1], *output_dir = argv[2];
  int num_workers = atoi(argv[3]);

  ConvMode mode = MODE_SEQ;
  if (strcmp(argv[4], "pixel") == 0)
    mode = MODE_PIXEL;
  else if (strcmp(argv[4], "row") == 0)
    mode = MODE_ROW;
  else if (strcmp(argv[4], "col") == 0)
    mode = MODE_COL;
  else if (strcmp(argv[4], "tile") == 0)
    mode = MODE_TILE;

  const char *csv_file = NULL;
  for (int i = 5; i < argc; ++i)
    if (strncmp(argv[i], "--csv=", 6) == 0)
      csv_file = argv[i] + 6;

  Filter filter = filter_gaussian_5x5();
  int omp_threads = 2;
  int runs = 10;

  printf("Pipeline benchmark: %d workers, mode=%s, omp_threads=%d\n",
         num_workers, argv[4], omp_threads);

  printf("Running baseline (sequential)...\n");
  double baseline =
      run_benchmark(input_dir, output_dir, 1, MODE_SEQ, &filter, 1, runs);
  printf("Baseline: %.3f s\n\n", baseline);
  printf("Running pipeline...\n");
  double pipeline_time = run_benchmark(input_dir, output_dir, num_workers, mode,
                                       &filter, omp_threads, runs);
  double speedup = baseline / pipeline_time;

  printf("Pipeline: %.3f s\n", pipeline_time);
  printf("Speedup: %.2fx\n\n", speedup);

  if (csv_file) {
    FILE *f = fopen(csv_file, "a");
    if (f) {
      fprintf(f, "%d,%s,%.3f,%.2f\n", num_workers, argv[4], pipeline_time,
              speedup);
      fclose(f);
    }
  }

  return 0;
}