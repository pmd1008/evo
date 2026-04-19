#pragma once
#include <cstdint>

static const int W = 600;
static const int H = 600;
static const int GENOME_MAX = 128;
static const int NUM_CLANS  = 4;

enum Action : uint8_t {
    ACT_WAIT=0, ACT_SLEEP,
    ACT_GROW_LEAF_U, ACT_GROW_LEAF_D, ACT_GROW_LEAF_L, ACT_GROW_LEAF_R,
    ACT_GROW_ROOT_D, ACT_GROW_ROOT_L, ACT_GROW_ROOT_R,
    ACT_GROW_ANT_U,  ACT_GROW_ANT_L,  ACT_GROW_ANT_R,
    ACT_GROW_DETOX_D,ACT_GROW_DETOX_L,ACT_GROW_DETOX_R,
    ACT_SPROUT_U, ACT_SPROUT_D, ACT_SPROUT_L, ACT_SPROUT_R,
    ACT_SHOOT_U,  ACT_SHOOT_D,  ACT_SHOOT_L,  ACT_SHOOT_R,
    ACT_EAT_U,    ACT_EAT_D,    ACT_EAT_L,    ACT_EAT_R,
    ACT_SKIP_LOW, ACT_SKIP_HIGH, ACT_SKIP_CROWD, ACT_SKIP_ALONE, ACT_SKIP_TOXIC,
    ACT_JUMP_B8,  ACT_JUMP_F8,
    ACT_SIG_A,    ACT_SIG_B,    ACT_SKIP_NSIG_A, ACT_SKIP_NSIG_B,
    ACT_COUNT
};

enum CellType : uint8_t {
    CT_EMPTY=0, CT_SPROUT, CT_WOOD,
    CT_LEAF, CT_ROOT, CT_ANTENNA, CT_DETOX, CT_SEED
};

// Геном хранится отдельно от клетки (pool)
struct Genome {
    uint8_t  code[GENOME_MAX];
    uint8_t  len;       // реальная длина (<=GENOME_MAX)
    uint8_t  clan;
    uint8_t  _pad[2];
};

// Клетка — компактная, влезает в кэш
struct Cell {
    CellType type;
    uint8_t  signal;
    uint8_t  skip_next;
    uint8_t  sleeping;
    int16_t  genome_idx;  // -1 = нет генома, иначе индекс в пуле
    uint8_t  ip;          // instruction pointer (0..127)
    uint8_t  age_steps;   // счётчик старости (0..127)
    int16_t  sleep_ticks;
    int16_t  age;
    int16_t  age_limit;
    float    energy;
};

// Почва
struct Soil {
    float organic;
    float charge;
    float toxin;
    float light;
};

// Статистика (атомарная, заполняется GPU)
struct Stats {
    int cells;
    int sprouts;
    int leaves;
    int births;
    int deaths;
    int clan_count[NUM_CLANS];
};
