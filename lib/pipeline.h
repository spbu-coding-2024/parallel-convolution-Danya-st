#ifndef PIPELINE_H
#define PIPELINE_H

#include "conv.h"
#include "filters.h"

typedef struct {
  int num_workers;
  int queue_capacity;
  ParallelMode mode;
  int tile_w, tile_h;
  int omp_threads;
} PipelineConfig;

typedef struct {
  int read_ok;
  int read_failed;
  int written;
  int write_failed;
  double wall_sec;
  double read_busy_sec;
  double conv_busy_sec;
  double write_busy_sec;
  double reader_blocked_sec;
  double workers_starved_sec;
  double workers_blocked_sec;
  double writer_starved_sec;
  int read_q_peak;
  int proc_q_peak;
} PipelineStats;

int pipeline_run(const char *in_dir, const char *out_dir, const Filter *filter,
                 const PipelineConfig *cfg, PipelineStats *stats);

#endif