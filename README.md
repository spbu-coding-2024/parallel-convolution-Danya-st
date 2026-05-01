# Задача 1

Реализация последовательной свёртки изображений с различными фильтрами на C с использованием OpenMP.

##  Использование
```bash
make all # сборка проекта
make test # запуск тестов 
make run # сборка и запуск с настройками по умолчанию
make clean # очистка
make help # справка
# запуск исполняемого файла
./build/convolution_seq <input/image.png> <ouput/result.png> [--filter=name]
```

## тестовое окружение
|  |  |
|---------|---------|
| **CPU** | 12th Gen Intel(R) Core(TM) i7-12700H |
| **RAM** | 16 ГБ |
| **ОС** | Arch Linux x86_64 |

## доступные фильтры
| размерность | фильтр |
|---------|---------|
| **3x3** | identity, blur3, gaussian3, edge, sharpen3, emboss, mean |
| **5x5** | blur5, gaussian5, edge_h5, sharpen5 |
| **9x9** | motion9 |

### Зависимости
- GCC с поддержкой OpenMP
- Библиотека stb_image для загрузки/сохранения изображений
