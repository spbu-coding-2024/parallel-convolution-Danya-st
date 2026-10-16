#include "pipeline.h"

#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <omp.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>

#include "stb_image.h"
#include "stb_image_write.h"

#define NAME_MAX_LEN 256
#define PATH_MAX_LEN 4096

/*
 * Время стадий меряется процессорным временем ПОТОКА (CLOCK_THREAD_CPUTIME_ID),
 * а не настенным: на машине, где потоков больше, чем ядер, настенное время
 * стадии включает ожидание процессора и раздувает суммы (сумма "занятого"
 * времени получалась больше времени всего прогона). Для свёртки в режиме с
 * OpenMP считается настенное время, т.к. работу делает команда потоков, а не
 * только вызывающий поток.
 */
static double cpu_now(void) {
  struct timespec ts;
  if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts) != 0)
    return omp_get_wtime();
  return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

typedef struct {
  unsigned char *data;
  int w, h;
  char name[NAME_MAX_LEN];
} ImageTask;

/*
 * Ограниченная блокирующая очередь с ЯВНЫМ закрытием.
 *
 * ИСПРАВЛЕНО (задача 3, "Утечка" и "зависание при отсутствии папки"):
 * раньше завершение передавалось "таблетками" (ImageTask с done = 1).
 *  - Последний воркер возвращал таблетку обратно в read_q и выходил, и её
 *    никто не освобождал - утечка памяти на каждый запуск.
 *  - Если читатель не мог открыть директорию, он выходил, НЕ положив
 *    таблетку, и воркеры навсегда оставались ждать на queue_pop (зависание).
 * Теперь вместо таблеток - queue_close(): после закрытия pop отдаёт
 * оставшиеся элементы, а затем NULL. Закрывает очередь тот, кто в неё
 * пишет, на ЕДИНСТВЕННОМ пути выхода (читатель - всегда перед return, воркеры
 * - главный поток после join всех воркеров), поэтому зависнуть нечему.
 */
typedef struct {
  ImageTask **items;
  int capacity, head, tail, count, closed, peak;
  pthread_mutex_t mutex;
  pthread_cond_t not_empty, not_full;
  /* суммарное время, проведённое в ожидании */
  double push_wait_sec, pop_wait_sec;
} TaskQueue;

static int queue_init(TaskQueue *q, int capacity) {
  memset(q, 0, sizeof *q);
  q->items = calloc((size_t)capacity, sizeof *q->items);
  if (!q->items)
    return -1;
  q->capacity = capacity;
  pthread_mutex_init(&q->mutex, NULL);
  pthread_cond_init(&q->not_empty, NULL);
  pthread_cond_init(&q->not_full, NULL);
  return 0;
}

static void queue_destroy(TaskQueue *q) {
  pthread_mutex_destroy(&q->mutex);
  pthread_cond_destroy(&q->not_empty);
  pthread_cond_destroy(&q->not_full);
  free(q->items);
}

/* Кладёт задачу в очередь, блокируясь, пока очередь полна. Очередь не должна
 * быть закрыта (в нашем конвейере пишет в закрываемую очередь только тот
 * поток, который её закрывает, и только после последнего push). */
static void queue_push(TaskQueue *q, ImageTask *task) {
  pthread_mutex_lock(&q->mutex);
  if (q->count == q->capacity) {
    const double t0 = omp_get_wtime();
    while (q->count == q->capacity)
      pthread_cond_wait(&q->not_full, &q->mutex);
    q->push_wait_sec += omp_get_wtime() - t0;
  }
  q->items[q->tail] = task;
  q->tail = (q->tail + 1) % q->capacity;
  ++q->count;
  if (q->count > q->peak)
    q->peak = q->count;
  pthread_cond_signal(&q->not_empty);
  pthread_mutex_unlock(&q->mutex);
}

/* Берёт задачу; блокируется, пока очередь пуста и не закрыта. NULL означает
 * "очередь закрыта и пуста". */
static ImageTask *queue_pop(TaskQueue *q) {
  pthread_mutex_lock(&q->mutex);
  if (q->count == 0 && !q->closed) {
    const double t0 = omp_get_wtime();
    while (q->count == 0 && !q->closed)
      pthread_cond_wait(&q->not_empty, &q->mutex);
    q->pop_wait_sec += omp_get_wtime() - t0;
  }
  if (q->count == 0) {
    pthread_mutex_unlock(&q->mutex);
    return NULL;
  }
  ImageTask *task = q->items[q->head];
  q->head = (q->head + 1) % q->capacity;
  --q->count;
  pthread_cond_signal(&q->not_full);
  pthread_mutex_unlock(&q->mutex);
  return task;
}

static void queue_close(TaskQueue *q) {
  pthread_mutex_lock(&q->mutex);
  q->closed = 1;
  pthread_cond_broadcast(&q->not_empty);
  pthread_mutex_unlock(&q->mutex);
}

typedef struct {
  TaskQueue read_q, proc_q;
  DIR *dir;
  const char *in_dir, *out_dir;
  const Filter *filter;
  PipelineConfig cfg;
  PipelineStats stats;
  /* защищает stats при слиянии локальных счётчиков потоков */
  pthread_mutex_t stats_mutex;
} Pipeline;

/*
 * Расширения без учёта регистра. ИСПРАВЛЕНО (задача 3, "поддержку jpg"):
 * раньше принимался только ровно ".png" в нижнем регистре, поэтому
 * photo.jpg / photo.JPEG / PHOTO.PNG молча пропускались (в input/ лежало два
 * .jpg, которые никогда не обрабатывались). Содержимое определяет сам stb
 * (поэтому файлы вида tiger.png, внутри являющиеся JPEG, тоже читаются).
 */
static int has_ext(const char *name, const char *ext) {
  const size_t n = strlen(name), e = strlen(ext);
  if (n <= e)
    return 0;
  for (size_t i = 0; i < e; ++i)
    if (tolower((unsigned char)name[n - e + i]) != ext[i])
      return 0;
  return 1;
}

static int is_image_name(const char *name) {
  return has_ext(name, ".png") || has_ext(name, ".jpg") ||
         has_ext(name, ".jpeg") || has_ext(name, ".bmp");
}

/*
 * ИСПРАВЛЕНО: раньше результат всегда кодировался в PNG, но записывался под
 * исходным именем, т.е. из photo.jpg получался файл photo.jpg с PNG внутри.
 * Теперь расширение соответствует формату: для не-PNG добавляется ".png".
 */
static void output_name(const char *in_name, char *out, size_t cap) {
  if (has_ext(in_name, ".png"))
    snprintf(out, cap, "%s", in_name);
  else
    snprintf(out, cap, "%s.png", in_name);
}

static void *reader_thread(void *arg) {
  Pipeline *p = (Pipeline *)arg;
  int ok = 0, failed = 0;
  double busy = 0.0;

  struct dirent *entry;
  while ((entry = readdir(p->dir)) != NULL) {
    const char *name = entry->d_name;
    if (!is_image_name(name) || strlen(name) >= NAME_MAX_LEN - 5)
      continue;

    char path[PATH_MAX_LEN];
    struct stat st;
    if (snprintf(path, sizeof path, "%s/%s", p->in_dir, name) >=
            (int)sizeof path ||
        stat(path, &st) != 0 || !S_ISREG(st.st_mode))
      continue;

    const double t0 = cpu_now();
    int w, h, ch;
    unsigned char *data = stbi_load(path, &w, &h, &ch, 1);
    busy += cpu_now() - t0;
    ImageTask *task = data ? malloc(sizeof *task) : NULL;
    if (!task) {
      fprintf(stderr, "Load error: %s (%s)\n", name,
              data ? "нет памяти" : stbi_failure_reason());
      if (data)
        stbi_image_free(data);
      ++failed;
      continue;
    }
    task->data = data;
    task->w = w;
    task->h = h;
    snprintf(task->name, sizeof task->name, "%s", name);
    queue_push(&p->read_q, task);
    ++ok;
  }

  /* Единственная точка выхода: очередь закрывается ВСЕГДА. */
  queue_close(&p->read_q);

  pthread_mutex_lock(&p->stats_mutex);
  p->stats.read_ok += ok;
  p->stats.read_failed += failed;
  p->stats.read_busy_sec += busy;
  pthread_mutex_unlock(&p->stats_mutex);
  return NULL;
}

/* Выбор стратегии свёртки. ИСПРАВЛЕНО: раньше такой switch был скопирован в
 * stream_main.c и bench_stream.c; теперь он один, а сама свёртка - общая из
 * lib/conv.c. */
static void run_convolution(const Pipeline *p, const ImageTask *t,
                            unsigned char *result) {
  switch (p->cfg.mode) {
  case MODE_PIXEL:
    pixel_convolution(t->data, result, t->w, t->h, p->filter);
    break;
  case MODE_ROW:
    row_convolution(t->data, result, t->w, t->h, p->filter);
    break;
  case MODE_COL:
    column_convolution(t->data, result, t->w, t->h, p->filter);
    break;
  case MODE_TILE:
    tile_convolution(t->data, result, t->w, t->h, p->cfg.tile_w,
                     p->cfg.tile_h, p->filter);
    break;
  case MODE_SEQ:
  default:
    seq_convolution(t->data, result, t->w, t->h, p->filter);
    break;
  }
}

static void *worker_thread(void *arg) {
  Pipeline *p = (Pipeline *)arg;
  double busy = 0.0;
  int failed = 0;

  /* omp_set_num_threads действует на вызывающий поток, поэтому вызывается в
   * каждом воркере (как и раньше в bench_stream). */
  if (p->cfg.omp_threads > 0)
    omp_set_num_threads(p->cfg.omp_threads);

  ImageTask *task;
  while ((task = queue_pop(&p->read_q)) != NULL) {
    unsigned char *result = malloc((size_t)task->w * (size_t)task->h);
    if (!result) {
      fprintf(stderr, "Нет памяти для результата: %s\n", task->name);
      stbi_image_free(task->data);
      free(task);
      ++failed;
      continue;
    }

    const int seq = p->cfg.mode == MODE_SEQ;
    const double t0 = seq ? cpu_now() : omp_get_wtime();
    run_convolution(p, task, result);
    busy += (seq ? cpu_now() : omp_get_wtime()) - t0;

    stbi_image_free(task->data);
    task->data = result;
    queue_push(&p->proc_q, task);
  }

  pthread_mutex_lock(&p->stats_mutex);
  p->stats.conv_busy_sec += busy;
  p->stats.write_failed += failed;
  pthread_mutex_unlock(&p->stats_mutex);
  return NULL;
}

static void *writer_thread(void *arg) {
  Pipeline *p = (Pipeline *)arg;
  double busy = 0.0;
  int written = 0, failed = 0;

  ImageTask *task;
  while ((task = queue_pop(&p->proc_q)) != NULL) {
    char name[NAME_MAX_LEN + 8], path[PATH_MAX_LEN];
    output_name(task->name, name, sizeof name);

    /*
     * ИСПРАВЛЕНО (задача 3, "проверку записи"): результат stbi_write_png
     * раньше игнорировался, и при ошибке записи (нет места, нет прав,
     * несуществующая папка) программа молча считала файл сохранённым.
     */
    const double t0 = cpu_now();
    const int ok = snprintf(path, sizeof path, "%s/%s", p->out_dir, name) <
                       (int)sizeof path &&
                   stbi_write_png(path, task->w, task->h, 1, task->data,
                                  task->w);
    busy += cpu_now() - t0;
    if (ok) {
      ++written;
    } else {
      fprintf(stderr, "Write error: %s/%s\n", p->out_dir, name);
      ++failed;
    }
    free(task->data);
    free(task);
  }

  pthread_mutex_lock(&p->stats_mutex);
  p->stats.write_busy_sec += busy;
  p->stats.written += written;
  p->stats.write_failed += failed;
  pthread_mutex_unlock(&p->stats_mutex);
  return NULL;
}

/* Создаёт out_dir, если его нет; -1, если это невозможно. */
static int ensure_dir(const char *path) {
  struct stat st;
  if (stat(path, &st) == 0)
    return S_ISDIR(st.st_mode) ? 0 : -1;
  return (mkdir(path, 0755) == 0 || errno == EEXIST) ? 0 : -1;
}

int pipeline_run(const char *in_dir, const char *out_dir, const Filter *filter,
                 const PipelineConfig *cfg, PipelineStats *stats) {
  if (!in_dir || !out_dir || !filter || !cfg || filter->size < 1 ||
      filter->size > MAX_FILTER_SIZE || filter->size % 2 == 0 ||
      cfg->num_workers < 1 || cfg->queue_capacity < 1)
    return -1;

  /*
   * ИСПРАВЛЕНО (задача 3, "зависание при отсутствии папки"): директория
   * открывается ДО запуска потоков, и при ошибке мы сразу возвращаем -1.
   * Раньше открытие делал читатель, и при ошибке он выходил без таблетки,
   * из-за чего воркеры и писатель ждали вечно.
   */
  DIR *dir = opendir(in_dir);
  if (!dir) {
    fprintf(stderr, "Cannot open %s\n", in_dir);
    return -1;
  }
  if (ensure_dir(out_dir) != 0) {
    fprintf(stderr, "Cannot use output directory %s\n", out_dir);
    closedir(dir);
    return -1;
  }

  Pipeline p;
  memset(&p, 0, sizeof p);
  p.dir = dir;
  p.in_dir = in_dir;
  p.out_dir = out_dir;
  p.filter = filter;
  p.cfg = *cfg;
  pthread_mutex_init(&p.stats_mutex, NULL);

  /*
   * ИСПРАВЛЕНО (задача 3, "лимит 16 воркеров"): массив pthread_t был
   * фиксированным workers[16] без проверки num_workers, т.е. запуск с
   * 17 и более воркерами портил стек. Теперь массив выделяется под
   * запрошенное число потоков.
   */
  pthread_t *workers = calloc((size_t)cfg->num_workers, sizeof *workers);
  if (!workers || queue_init(&p.read_q, cfg->queue_capacity) != 0) {
    free(workers);
    closedir(dir);
    return -1;
  }
  if (queue_init(&p.proc_q, cfg->queue_capacity) != 0) {
    queue_destroy(&p.read_q);
    free(workers);
    closedir(dir);
    return -1;
  }

  int status = 0;
  const double start = omp_get_wtime();

  /* Порядок запуска: писатель, воркеры, читатель - так, что если не удалось
   * создать потоки, читатель ещё не стартовал и зависнуть некому. */
  pthread_t writer, reader;
  int writer_started = 0, reader_started = 0, workers_started = 0;
  if (pthread_create(&writer, NULL, writer_thread, &p) == 0)
    writer_started = 1;
  for (int i = 0; writer_started && i < cfg->num_workers; ++i) {
    if (pthread_create(&workers[i], NULL, worker_thread, &p) != 0)
      break;
    ++workers_started;
  }
  if (workers_started > 0) {
    if (pthread_create(&reader, NULL, reader_thread, &p) == 0)
      reader_started = 1;
  }
  if (!reader_started) {
    status = -1;
    queue_close(&p.read_q);
    closedir(dir);
  }

  if (reader_started)
    pthread_join(reader, NULL);
  for (int i = 0; i < workers_started; ++i)
    pthread_join(workers[i], NULL);
  /* Все воркеры завершились - в proc_q больше никто не пишет. */
  queue_close(&p.proc_q);
  if (writer_started)
    pthread_join(writer, NULL);
  if (reader_started)
    closedir(dir);

  p.stats.wall_sec = omp_get_wtime() - start;
  p.stats.reader_blocked_sec = p.read_q.push_wait_sec;
  p.stats.workers_starved_sec = p.read_q.pop_wait_sec;
  p.stats.workers_blocked_sec = p.proc_q.push_wait_sec;
  p.stats.writer_starved_sec = p.proc_q.pop_wait_sec;
  p.stats.read_q_peak = p.read_q.peak;
  p.stats.proc_q_peak = p.proc_q.peak;
  if (stats)
    *stats = p.stats;

  queue_destroy(&p.read_q);
  queue_destroy(&p.proc_q);
  pthread_mutex_destroy(&p.stats_mutex);
  free(workers);
  return status;
}