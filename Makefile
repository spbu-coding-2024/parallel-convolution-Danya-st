CC      := gcc
CFLAGS  := -std=c99 -O2 -Wall -Wextra -Ilib
LDFLAGS := -lm -lpng 

# Пути
SRC_DIR     := src
TESTS_DIR   := tests
INPUT_DIR   := input
OUTPUT_DIR  := output
BUILD_DIR   := build


TARGET      := $(BUILD_DIR)/convolution_seq
TEST_TARGET := $(BUILD_DIR)/property_tests

MAIN_SRC    := $(SRC_DIR)/main.c
TEST_SRCS   := $(wildcard $(TESTS_DIR)/*.c)

.PHONY: all clean run test dirs

all: dirs $(TARGET)

$(TARGET): $(MAIN_SRC) | dirs
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(TEST_TARGET): $(TEST_SRCS) $(MAIN_SRC) | dirs
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) -DTEST_MODE -o $@ $(TEST_SRCS) $(LDFLAGS)

dirs:
	@mkdir -p $(BUILD_DIR) $(OUTPUT_DIR)

run: $(TARGET)
	./$(TARGET) $(INPUT_DIR)/tiger.png $(OUTPUT_DIR)/result.png

test: $(TEST_TARGET) | dirs
	@echo "Запуск property-тестов..."
	@./$(TEST_TARGET) --input-dir $(INPUT_DIR) --output-dir $(OUTPUT_DIR)


clean:
	rm -rf $(BUILD_DIR) $(OUTPUT_DIR)/*.png


help:
	@echo "Доступные цели:"
	@echo "  make all          # Сборка проекта"
	@echo "  make run          # Запуск с фото по умолчанию"
	@echo "  make test         # Запуск property-тестов"
	@echo "  make bench        # Бенчмарк на разных размерах"
	@echo "  make clean        # Очистка артефактов"
	@echo "  make help         # Эта справка"