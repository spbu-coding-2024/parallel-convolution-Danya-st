#ifndef FILTERS_H
#define FILTERS_H

#define MAX_FILTER_SIZE 9

typedef struct {
  int size;
  double kernel[MAX_FILTER_SIZE][MAX_FILTER_SIZE];
  double factor;
  double bias;
  const char *name;
} Filter;

Filter filter_identity_3x3(void) {
  Filter f = {3, {{0, 0, 0}, {0, 1, 0}, {0, 0, 0}}, 1.0, 0.0, "identity_3x3"};
  return f;
}

Filter filter_blur_3x3(void) {
  Filter f = {3,
              {{0.0, 0.2, 0.0}, {0.2, 0.2, 0.2}, {0.0, 0.2, 0.0}},
              1.0,
              0.0,
              "blur_3x3"};
  return f;
}

Filter filter_gaussian_3x3(void) {
  Filter f = {
      3, {{1, 2, 1}, {2, 4, 2}, {1, 2, 1}}, 1.0 / 16.0, 0.0, "gaussian_3x3"};
  return f;
}

Filter filter_edge_3x3(void) {
  Filter f = {
      3, {{-1, -1, -1}, {-1, 8, -1}, {-1, -1, -1}}, 1.0, 0.0, "edge_3x3"};
  return f;
}

Filter filter_sharpen_3x3(void) {
  Filter f = {
      3, {{-1, -1, -1}, {-1, 9, -1}, {-1, -1, -1}}, 1.0, 0.0, "sharpen_3x3"};
  return f;
}

Filter filter_emboss_3x3(void) {
  Filter f = {
      3, {{-1, -1, 0}, {-1, 0, 1}, {0, 1, 1}}, 1.0, 128.0, "emboss_3x3"};
  return f;
}

Filter filter_mean_3x3(void) {
  Filter f = {3, {{1, 1, 1}, {1, 1, 1}, {1, 1, 1}}, 1.0 / 9.0, 0.0, "mean_3x3"};
  return f;
}

Filter filter_blur_5x5(void) {
  Filter f = {5,
              {{0, 0, 1, 0, 0},
               {0, 1, 1, 1, 0},
               {1, 1, 1, 1, 1},
               {0, 1, 1, 1, 0},
               {0, 0, 1, 0, 0}},
              1.0 / 13.0,
              0.0,
              "blur_5x5"};
  return f;
}

Filter filter_gaussian_5x5(void) {
  Filter f = {5,
              {{1, 4, 6, 4, 1},
               {4, 16, 24, 16, 4},
               {6, 24, 36, 24, 6},
               {4, 16, 24, 16, 4},
               {1, 4, 6, 4, 1}},
              1.0 / 256.0,
              0.0,
              "gaussian_5x5"};
  return f;
}

Filter filter_edge_h_5x5(void) {
  Filter f = {5,
              {{0, 0, -1, 0, 0},
               {0, 0, -1, 0, 0},
               {0, 0, 2, 0, 0},
               {0, 0, 0, 0, 0},
               {0, 0, 0, 0, 0}},
              1.0,
              0.0,
              "edge_h_5x5"};
  return f;
}

Filter filter_sharpen_5x5(void) {
  Filter f = {5,
              {{-1, -1, -1, -1, -1},
               {-1, 2, 2, 2, -1},
               {-1, 2, 8, 2, -1},
               {-1, 2, 2, 2, -1},
               {-1, -1, -1, -1, -1}},
              1.0 / 8.0,
              0.0,
              "sharpen_5x5"};
  return f;
}

Filter filter_motion_9x9(void) {
  Filter f = {9,
              {{1, 0, 0, 0, 0, 0, 0, 0, 0},
               {0, 1, 0, 0, 0, 0, 0, 0, 0},
               {0, 0, 1, 0, 0, 0, 0, 0, 0},
               {0, 0, 0, 1, 0, 0, 0, 0, 0},
               {0, 0, 0, 0, 1, 0, 0, 0, 0},
               {0, 0, 0, 0, 0, 1, 0, 0, 0},
               {0, 0, 0, 0, 0, 0, 1, 0, 0},
               {0, 0, 0, 0, 0, 0, 0, 1, 0},
               {0, 0, 0, 0, 0, 0, 0, 0, 1}},
              1.0 / 9.0,
              0.0,
              "motion_9x9"};
  return f;
}

Filter get_filter(const char *name) {
  if (!name)
    return filter_identity_3x3();

  if (strcmp(name, "identity") == 0)
    return filter_identity_3x3();
  if (strcmp(name, "blur3") == 0)
    return filter_blur_3x3();
  if (strcmp(name, "blur5") == 0)
    return filter_blur_5x5();
  if (strcmp(name, "gaussian3") == 0)
    return filter_gaussian_3x3();
  if (strcmp(name, "gaussian5") == 0)
    return filter_gaussian_5x5();
  if (strcmp(name, "edge") == 0)
    return filter_edge_3x3();
  if (strcmp(name, "edge_h5") == 0)
    return filter_edge_h_5x5();
  if (strcmp(name, "sharpen3") == 0)
    return filter_sharpen_3x3();
  if (strcmp(name, "sharpen5") == 0)
    return filter_sharpen_5x5();
  if (strcmp(name, "emboss") == 0)
    return filter_emboss_3x3();
  if (strcmp(name, "mean") == 0)
    return filter_mean_3x3();
  if (strcmp(name, "motion9") == 0)
    return filter_motion_9x9();

  return filter_identity_3x3();
}

#endif