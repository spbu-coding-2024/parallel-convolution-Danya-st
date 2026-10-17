CC      := gcc
CFLAGS  := -std=c99 -O2 -Wall -Wextra -Werror=unknown-pragmas -fopenmp \
           -pthread -Ilib -D_POSIX_C_SOURCE=200809L -ffp-contract=off \
           -MMD -MP
LDFLAGS := -lm -fopenmp -pthread

# Пути
SRC_DIR     := src
TESTS_DIR   := tests
INPUT_DIR   := input
OUTPUT_DIR  := output
BUILD_DIR   := build
OBJ_DIR     := $(BUILD_DIR)/obj
DATA_DIR    := benchmark_data
PLOTS_DIR   := plots

TARGET        := $(BUILD_DIR)/convolution_seq
STREAM_TARGET := $(BUILD_DIR)/stream_conv
BENCH_BIN     := $(BUILD_DIR)/bench
BENCH_STREAM  := $(BUILD_DIR)/bench_stream
TEST_TARGET   := $(BUILD_DIR)/property_tests

LIB_OBJS     := $(OBJ_DIR)/conv.o $(OBJ_DIR)/stb_impl.o
PIPE_OBJS    := $(OBJ_DIR)/pipeline.o
MAIN_OBJS    := $(OBJ_DIR)/main.o $(LIB_OBJS)
STREAM_OBJS  := $(OBJ_DIR)/stream_main.o $(LIB_OBJS) $(PIPE_OBJS)
BENCH_OBJS   := $(OBJ_DIR)/bench.o $(LIB_OBJS)
BSTREAM_OBJS := $(OBJ_DIR)/bench_stream.o $(LIB_OBJS) $(PIPE_OBJS)
TEST_OBJS    := $(OBJ_DIR)/property_tests.o $(LIB_OBJS) $(PIPE_OBJS)

FORMAT_FILES := $(wildcard $(SRC_DIR)/*.c $(TESTS_DIR)/*.c $(TESTS_DIR)/*.h) \
                lib/conv.c lib/conv.h lib/filters.h lib/pipeline.c \
                lib/pipeline.h

.PHONY: all clean run stream test dirs format format-check help \
        bench bench-run plot bench-stream plot-stream

all: dirs $(TARGET) $(STREAM_TARGET)

$(TARGET): $(MAIN_OBJS)
	$(CC) -o $@ $^ $(LDFLAGS)

$(STREAM_TARGET): $(STREAM_OBJS)
	$(CC) -o $@ $^ $(LDFLAGS)

$(BENCH_BIN): $(BENCH_OBJS)
	$(CC) -o $@ $^ $(LDFLAGS)

$(BENCH_STREAM): $(BSTREAM_OBJS)
	$(CC) -o $@ $^ $(LDFLAGS)

$(TEST_TARGET): $(TEST_OBJS)
	$(CC) -o $@ $^ $(LDFLAGS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c | dirs
	$(CC) $(CFLAGS) -c -o $@ $<

$(OBJ_DIR)/%.o: $(TESTS_DIR)/%.c | dirs
	$(CC) $(CFLAGS) -c -o $@ $<

$(OBJ_DIR)/%.o: lib/%.c | dirs
	$(CC) $(CFLAGS) -c -o $@ $<

$(OBJ_DIR)/stb_impl.o: lib/stb_impl.c | dirs
	$(CC) -std=c99 -O2 -w -Ilib -MMD -MP -c -o $@ $<

dirs:
	@mkdir -p $(OBJ_DIR) $(OUTPUT_DIR)

run: $(TARGET)
	./$(TARGET) $(INPUT_DIR)/tiger.png $(OUTPUT_DIR)/result.png \
		--filter=gaussian3 --parallel=row

stream: $(STREAM_TARGET)
	./$(STREAM_TARGET) $(INPUT_DIR) $(OUTPUT_DIR) 4 --filter=gaussian3 \
		--parallel=row --omp-threads=2

test: $(TEST_TARGET)
	@echo "Запуск property-тестов..."
	@./$(TEST_TARGET)

bench-run: $(BENCH_BIN)
	@mkdir -p $(DATA_DIR)
	./$(BENCH_BIN) $(INPUT_DIR)/tiger.png --csv=$(DATA_DIR)/results.csv \
		$(BENCH_ARGS)

plot:
	@command -v python3 >/dev/null 2>&1 || { echo "Python3 required for plotting"; exit 1; }
	@python3 scripts/plot.py $(DATA_DIR)/results.csv $(PLOTS_DIR)

bench: bench-run plot
	@echo "Бенчмарк завершён. Графики и summary.md: $(PLOTS_DIR)/"

bench-stream: $(BENCH_STREAM)
	@mkdir -p $(DATA_DIR)
	./$(BENCH_STREAM) $(INPUT_DIR) --csv=$(DATA_DIR)/pipeline.csv \
		$(BENCH_STREAM_ARGS)

plot-stream:
	@command -v python3 >/dev/null 2>&1 || { echo "Python3 required for plotting"; exit 1; }
	@python3 scripts/plot_pipeline.py $(DATA_DIR)/pipeline.csv $(PLOTS_DIR)

format-check:
	@command -v clang-format >/dev/null 2>&1 || { echo "clang-format not found"; exit 0; }
	@clang-format --dry-run --Werror $(FORMAT_FILES) 2>/dev/null || \
	(echo "Code formatting check failed. Run 'make format' to fix." && exit 1)

format:
	@clang-format -i $(FORMAT_FILES)

clean:
	rm -rf $(BUILD_DIR) $(OUTPUT_DIR)/*.png

help:
	@echo "Доступные цели:"
	@echo "  make all           # Сборка всех программ"
	@echo "  make run           # Свёртка одного изображения"
	@echo "  make stream        # Потоковая обработка папки"
	@echo "  make test          # Property-тесты"
	@echo "  make bench         # Бенчмарк свёртки + графики"
	@echo "  make bench-run     # Только бенчмарк"
	@echo "  make plot          # Только графики из results.csv"
	@echo "  make bench-stream  # Бенчмарк конвейера"
	@echo "  make plot-stream   # Графики конвейера из pipeline.csv"
	@echo "  make format        # Автоформатирование"
	@echo "  make clean         # Очистка артефактов"
	@echo "  make help          # Эта справка"

-include $(wildcard $(OBJ_DIR)/*.d)