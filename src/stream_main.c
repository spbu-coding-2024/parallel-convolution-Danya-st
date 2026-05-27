#define STB_IMAGE_IMPLEMENTATION
#include "../lib/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../lib/stb_image_write.h"

#include "../lib/filters.h"
#include <omp.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <dirent.h>
#include <sys/stat.h>

#define MAX_FILTER_SIZE 9
#define FILTER_WIDTH 3
#define FILTER_HEIGHT 3
#define QUEUE_CAPACITY 8
#define MAX_PATH 1024

double factor = 1.0;
double bias = 0.0;

typedef enum {
  MODE_SEQ,
  MODE_PIXEL,
  MODE_ROW,
  MODE_COL,
  MODE_TILE
} ParallelMode;

typedef struct {
  unsigned char *data;
  int w, h;
  char filename[MAX_PATH];
  int done;
} ImageTask;

typedef struct {
  ImageTask *items[QUEUE_CAPACITY];
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
  while (q->count == QUEUE_CAPACITY && !q->shutdown)
    pthread_cond_wait(&q->not_full, &q->mutex);
  if (q->shutdown) {
    pthread_mutex_unlock(&q->mutex);
    return -1;
  }
  q->items[q->tail] = task;
  q->tail = (q->tail + 1) % QUEUE_CAPACITY;
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
  q->head = (q->head + 1) % QUEUE_CAPACITY;
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

static inline unsigned char clamp(double value) {
  int v = (int)(factor * value + bias);
  if (v < 0)
    return 0;
  if (v > 255)
    return 255;
  return (unsigned char)v;
}

void seq_convolution(unsigned char *src, unsigned char *result, int w, int h,
                     Filter *f) {
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      double sum = 0.0;
      for (int fy = 0; fy < FILTER_HEIGHT; ++fy)
        for (int fx = 0; fx < FILTER_WIDTH; ++fx) {
          int ix = (x - FILTER_WIDTH / 2 + fx + w) % w;
          int iy = (y - FILTER_HEIGHT / 2 + fy + h) % h;
          sum += src[iy * w + ix] * f->kernel[fy][fx];
        }
      result[y * w + x] = clamp(sum);
    }
}

void pixel_convolution(unsigned char *src, unsigned char *result, int w, int h,
                       Filter *f) {
  const long total = (long)w * h;
#pragma omp parallel for
  for (long idx = 0; idx < total; ++idx) {
    int x = idx % w, y = idx / w;
    double sum = 0.0;
    for (int fy = 0; fy < FILTER_HEIGHT; ++fy)
      for (int fx = 0; fx < FILTER_WIDTH; ++fx) {
        int ix = (x - FILTER_WIDTH / 2 + fx + w) % w;
        int iy = (y - FILTER_HEIGHT / 2 + fy + h) % h;
        sum += src[iy * w + ix] * f->kernel[fy][fx];
      }
    result[idx] = clamp(sum);
  }
}

void row_convolution(unsigned char *src, unsigned char *result, int w, int h,
                     Filter *f) {
#pragma omp parallel for
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      double sum = 0.0;
      for (int fy = 0; fy < FILTER_HEIGHT; ++fy)
        for (int fx = 0; fx < FILTER_WIDTH; ++fx) {
          int ix = (x - FILTER_WIDTH / 2 + fx + w) % w;
          int iy = (y - FILTER_HEIGHT / 2 + fy + h) % h;
          sum += src[iy * w + ix] * f->kernel[fy][fx];
        }
      result[y * w + x] = clamp(sum);
    }
}

void column_convolution(unsigned char *src, unsigned char *result, int w, int h,
                        Filter *f) {
#pragma omp parallel for
  for (int x = 0; x < w; ++x)
    for (int y = 0; y < h; ++y) {
      double sum = 0.0;
      for (int fy = 0; fy < FILTER_HEIGHT; ++fy)
        for (int fx = 0; fx < FILTER_WIDTH; ++fx) {
          int ix = (x - FILTER_WIDTH / 2 + fx + w) % w;
          int iy = (y - FILTER_HEIGHT / 2 + fy + h) % h;
          sum += src[iy * w + ix] * f->kernel[fy][fx];
        }
      result[y * w + x] = clamp(sum);
    }
}

void tile_convolution(unsigned char *src, unsigned char *result, int w, int h,
                      int tw, int th, Filter *f) {
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
        for (int fy = 0; fy < FILTER_HEIGHT; ++fy)
          for (int fx = 0; fx < FILTER_WIDTH; ++fx) {
            int ix = (x - FILTER_WIDTH / 2 + fx + w) % w;
            int iy = (y - FILTER_HEIGHT / 2 + fy + h) % h;
            sum += src[iy * w + ix] * f->kernel[fy][fx];
          }
        result[y * w + x] = clamp(sum);
      }
  }
}

typedef struct {
  TaskQueue *read_q, *proc_q;
  const char *input_dir, *output_dir;
  Filter *filter;
  ParallelMode mode;
  int tile_w, tile_h;
  int processed_count;
  pthread_mutex_t count_mutex;
} WorkerArgs;

void *reader_thread(void *arg) {
  WorkerArgs *args = (WorkerArgs *)arg;
  DIR *dir = opendir(args->input_dir);
  if (!dir) {
    fprintf(stderr, "Cannot open %s\n", args->input_dir);
    return NULL;
  }

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
    if (!src) {
      fprintf(stderr, "Load error: %s\n", name);
      continue;
    }

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

  ImageTask *done = malloc(sizeof(ImageTask));
  done->done = 1;
  queue_push(args->read_q, done);
  return NULL;
}

void *worker_thread(void *arg) {
  WorkerArgs *args = (WorkerArgs *)arg;

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
      pixel_convolution(task->data, result, task->w, task->h, args->filter);
      break;
    case MODE_ROW:
      row_convolution(task->data, result, task->w, task->h, args->filter);
      break;
    case MODE_COL:
      column_convolution(task->data, result, task->w, task->h, args->filter);
      break;
    case MODE_TILE:
      tile_convolution(task->data, result, task->w, task->h, args->tile_w,
                       args->tile_h, args->filter);
      break;
    default:
      seq_convolution(task->data, result, task->w, task->h, args->filter);
    }

    free(task->data);
    task->data = result;

    if (queue_push(args->proc_q, task) < 0) {
      free(task->data);
      free(task);
      break;
    }

    pthread_mutex_lock(&args->count_mutex);
    args->processed_count++;
    pthread_mutex_unlock(&args->count_mutex);
  }
  return NULL;
}

void *writer_thread(void *arg) {
  WorkerArgs *args = (WorkerArgs *)arg;

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

int main(int argc, char *argv[]) {
  if (argc < 4) {
    fprintf(stderr,
            "usage: %s <input_dir> <output_dir> <num_workers> [--filter=name] "
            "[--parallel=mode] [--tile=WxH]\n",
            argv[0]);
    printf("Filters: identity, blur3, gaussian3, edge, sharpen3, emboss, mean, "
           "motion9\n");
    printf("Modes: seq, pixel, row, col, tile (default: seq)\n");
    return EXIT_FAILURE;
  }

  const char *input_dir = argv[1], *output_dir = argv[2];
  int num_workers = atoi(argv[3]);
  if (num_workers < 1)
    num_workers = 1;

  // Парсинг аргументов
  char *filter_name = "identity";
  ParallelMode mode = MODE_SEQ;
  int tile_w = 32, tile_h = 32;

  for (int i = 4; i < argc; ++i) {
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
    } else if (strncmp(argv[i], "--tile=", 7) == 0) {
      sscanf(argv[i] + 7, "%dx%d", &tile_w, &tile_h);
    }
  }

  Filter filter = get_filter(filter_name);

  TaskQueue read_q, proc_q;
  queue_init(&read_q);
  queue_init(&proc_q);

  WorkerArgs args = {.read_q = &read_q,
                     .proc_q = &proc_q,
                     .input_dir = input_dir,
                     .output_dir = output_dir,
                     .filter = &filter,
                     .mode = mode,
                     .tile_w = tile_w,
                     .tile_h = tile_h,
                     .processed_count = 0};
  pthread_mutex_init(&args.count_mutex, NULL);

  pthread_t reader, writer, workers[16];
  pthread_create(&reader, NULL, reader_thread, &args);
  pthread_create(&writer, NULL, writer_thread, &args);
  for (int i = 0; i < num_workers; ++i)
    pthread_create(&workers[i], NULL, worker_thread, &args);

  pthread_join(reader, NULL);
  for (int i = 0; i < num_workers; ++i)
    pthread_join(workers[i], NULL);

  ImageTask *done = malloc(sizeof(ImageTask));
  done->done = 1;
  queue_push(&proc_q, done);
  pthread_join(writer, NULL);

  queue_shutdown(&read_q);
  queue_shutdown(&proc_q);
  queue_destroy(&read_q);
  queue_destroy(&proc_q);
  pthread_mutex_destroy(&args.count_mutex);

  printf("Processed %d images\n", args.processed_count);
  return 0;
}