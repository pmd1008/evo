# EvoSim CUDA

Небольшая CUDA/SDL симуляция эволюции клеточных организмов.

Организмы растят листья, корни, антенны, детокс-клетки, отростки и семена. Геномы мутируют при размножении, а выживание зависит от света, ресурсов почвы, токсичности, стоимости роста и скученности.

## Сборка и запуск

Нужны:

- C++17 компилятор (`g++`)
- `make`
- NVIDIA driver
- CUDA Toolkit с `nvcc`
- `pkg-config`
- SDL2
- SDL2_ttf
- любой из шрифтов, которые ищет приложение: DejaVu Sans Mono, Liberation Mono или Noto Sans Mono

Сейчас `Makefile` смотрит на CUDA в `/usr/local/cuda-13.2`. Если CUDA стоит в другом месте, поменяй `NVCC` и `CUDA_PATH` в `Makefile`.

Примеры пакетов:

```bash
# Arch / Manjaro
sudo pacman -S base-devel cuda sdl2 sdl2_ttf pkgconf ttf-dejavu

# Fedora
sudo dnf install gcc-c++ make SDL2-devel SDL2_ttf-devel pkgconf-pkg-config dejavu-sans-mono-fonts

# Debian / Ubuntu
sudo apt install build-essential libsdl2-dev libsdl2-ttf-dev pkg-config fonts-dejavu-core
```

CUDA Toolkit и NVIDIA driver ставятся по ситуации: из пакетов дистрибутива или с сайта NVIDIA. После установки проверь, что доступны `nvcc --version` и `nvidia-smi`.

```bash
make
./evosim_cuda
```

## Управление

- `Пробел` - пауза
- `R` - перезапуск мира
- `1-6` - режимы отображения
- колесо мыши - зум
- средняя кнопка мыши - двигать камеру

В правой панели есть настройки баланса. Их можно менять на лету: солнце, силу листьев/корней/антенн, скученность, метаболизм, стоимость роста, стоимость детей и семян, мутации и токсичность.
