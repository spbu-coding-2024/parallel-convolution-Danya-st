CC      := gcc

CFLAGS  := -std=c99 -O2 -Wall -Wextra -Werror=unknown-pragmas -fopenmp \
           -Ilib -D_POSIX_C_SOURCE=200809L -ffp-contract=off -MMD -MP
LDFLAGS := -lm -fopenmp

# Пути
SRC_DIR     := src
TESTS_DIR   := tests
INPUT_DIR   := input
OUTPUT_DIR  := output
BUILD_DIR   := build
OBJ_DIR     := $(BUILD_DIR)/obj
DATA_DIR    := benchmark_data
PLOTS_DIR   := benchmarks

TARGET      := $(BUILD_DIR)/convolution_seq
TEST_TARGET := $(BUILD_DIR)/property_tests
BENCH_BIN   := $(BUILD_DIR)/bench


LIB_OBJS    := $(OBJ_DIR)/conv.o $(OBJ_DIR)/stb_impl.o
MAIN_OBJS   := $(OBJ_DIR)/main.o $(LIB_OBJS)
TEST_OBJS   := $(OBJ_DIR)/property_tests.o $(LIB_OBJS)
BENCH_OBJS  := $(OBJ_DIR)/bench.o $(LIB_OBJS)

FORMAT_FILES := $(wildcard $(SRC_DIR)/*.c $(TESTS_DIR)/*.c $(TESTS_DIR)/*.h) \
                lib/conv.c lib/conv.h lib/filters.h

.PHONY: all clean run test dirs format format-check help \
        bench bench-run plot

all: dirs $(TARGET)

$(TARGET): $(MAIN_OBJS)
	$(CC) -o $@ $^ $(LDFLAGS)

$(TEST_TARGET): $(TEST_OBJS)
	$(CC) -o $@ $^ $(LDFLAGS)

$(BENCH_BIN): $(BENCH_OBJS)
	$(CC) -o $@ $^ $(LDFLAGS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c | dirs
	$(CC) $(CFLAGS) -c -o $@ $<

$(OBJ_DIR)/%.o: $(TESTS_DIR)/%.c | dirs
	$(CC) $(CFLAGS) -c -o $@ $<

$(OBJ_DIR)/conv.o: lib/conv.c | dirs
	$(CC) $(CFLAGS) -c -o $@ $<

$(OBJ_DIR)/stb_impl.o: lib/stb_impl.c | dirs
	$(CC) -std=c99 -O2 -w -Ilib -MMD -MP -c -o $@ $<

dirs:
	@mkdir -p $(OBJ_DIR) $(OUTPUT_DIR)

run: $(TARGET)
	./$(TARGET) $(INPUT_DIR)/tiger.png $(OUTPUT_DIR)/result.png \
		--filter=gaussian3 --parallel=row

test: $(TEST_TARGET)
	@echo "Запуск property-тестов..."
	@./$(TEST_TARGET) $(TEST_ARGS)

bench-run: $(BENCH_BIN)
	@mkdir -p $(DATA_DIR)
	./$(BENCH_BIN) $(INPUT_DIR)/tiger.png --csv=$(DATA_DIR)/results.csv \
		$(BENCH_ARGS)

plot:
	@command -v python3 >/dev/null 2>&1 || { echo "Python3 required for plotting"; exit 1; }
	@python3 scripts/plot.py $(DATA_DIR)/results.csv $(PLOTS_DIR)

bench: bench-run plot
	@echo "Бенчмарк завершён. Графики: $(PLOTS_DIR)/"

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
	@echo "  make all          # Сборка проекта"
	@echo "  make run          # Запуск с фото по умолчанию"
	@echo "  make test         # Property-тесты (TEST_ARGS=--seed=N)"
	@echo "  make bench        # Бенчмарк + графики (BENCH_ARGS=...)"
	@echo "  make bench-run    # Только бенчмарк -> $(DATA_DIR)/results.csv"
	@echo "  make plot         # Только графики из CSV"
	@echo "  make format       # Автоформатирование (clang-format)"
	@echo "  make clean        # Очистка артефактов"
	@echo "  make help         # Эта справка"

-include $(wildcard $(OBJ_DIR)/*.d)