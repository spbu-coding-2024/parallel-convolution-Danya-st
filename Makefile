CC      := gcc
CFLAGS  := -std=c99 -O2 -Wall -Wextra -Ilib -fopenmp
LDFLAGS := -lm -lpng -fopenmp

SRC_DIR     := src
TESTS_DIR   := tests
INPUT_DIR   := input
OUTPUT_DIR  := output
BUILD_DIR   := build


TARGET      := $(BUILD_DIR)/convolution_seq
STREAM_TARGET := $(BUILD_DIR)/stream_conv
TEST_TARGET := $(BUILD_DIR)/property_tests

MAIN_SRC    := $(SRC_DIR)/main.c
STREAM_SRC  := $(SRC_DIR)/stream_main.c
TEST_SRCS   := $(wildcard $(TESTS_DIR)/*.c)

.PHONY: all clean run test dirs stream bench bench-run plot bench-stream

all: dirs $(TARGET)

$(TARGET): $(MAIN_SRC) | dirs
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(STREAM_TARGET): $(STREAM_SRC) | dirs
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(TEST_TARGET): $(TEST_SRCS) $(MAIN_SRC) | dirs
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) -DTEST_MODE -o $@ $(TEST_SRCS) $(LDFLAGS)

dirs:
	@mkdir -p $(BUILD_DIR) $(OUTPUT_DIR)

run: $(TARGET)
	./$(TARGET) $(INPUT_DIR)/tiger.png $(OUTPUT_DIR)/result.png

stream: $(STREAM_TARGET)
	./$(STREAM_TARGET) $(INPUT_DIR) $(OUTPUT_DIR) 4 --filter=gaussian3 --parallel=row

test: $(TEST_TARGET) | dirs
	@echo "Запуск property-тестов..."
	@./$(TEST_TARGET) --input-dir $(INPUT_DIR) --output-dir $(OUTPUT_DIR)

clean:
	rm -rf $(BUILD_DIR) $(OUTPUT_DIR)/*.png

format-check:
	@command -v clang-format >/dev/null 2>&1 || { echo "clang-format not found"; exit 0; }
	@clang-format --dry-run --Werror src/*.c  tests/*.c 2>/dev/null || \
	(echo "Code formatting check failed. Run 'make format' to fix." && exit 1)

help:
	@echo "Доступные цели:"
	@echo "  make all          # Сборка проекта"
	@echo "  make run          # Запуск с фото по умолчанию"
	@echo "  make stream       # запуск с разделением чтения записи свёртки"
	@echo "  make test         # Запуск property-тестов"
	@echo "  make bench        # Бенчмарк на разных размерах"
	@echo "  make clean        # Очистка артефактов"
	@echo "  make help         # Эта справка"

BENCH_BIN := $(BUILD_DIR)/bench
BENCH_STREAM := $(BUILD_DIR)/bench_stream


$(BENCH_BIN): src/bench.c lib/filters.h | dirs
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(BENCH_STREAM): src/bench_stream.c lib/filters.h | dirs
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

bench-run: $(BENCH_BIN) | dirs
	@mkdir -p benchmark_data
	./$(BENCH_BIN) $(INPUT_DIR)/tiger.png --csv=benchmark_data/results.csv --threads=8

bench-stream: $(BENCH_STREAM) | dirs
	@mkdir -p benchmark_data
	@echo "workers,mode,time_sec,speedup" > benchmark_data/pipeline.csv
	@for workers in 1 2 4 8; do \
		for mode in seq pixel row col tile; do \
			echo "  Workers: $$workers, Mode: $$mode"; \
			./$(BENCH_STREAM) $(INPUT_DIR) $(OUTPUT_DIR) $$workers $$mode \
				--csv=benchmark_data/pipeline.csv 2>&1 | grep -E "(Baseline|Pipeline|Speedup)"; \
		done; \
	done
plot: 
	@command -v python3 >/dev/null 2>&1 || { echo "Python3 required for plotting"; exit 1; }
	@python3 scripts/plot.py benchmark_data/results.csv plots

plot-stream: 
	@command -v python3 >/dev/null 2>&1 || { echo "Python3 required for plotting"; exit 1; }
	@python3 scripts/plot_pipeline.py benchmark_data/pipeline.csv plots

bench: bench-run plot
	@echo " Benchmark complete. See plots/"