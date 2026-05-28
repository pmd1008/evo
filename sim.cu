#include "types.h"
#include <curand_kernel.h>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <cstdio>

// ─── GPU буферы ────────────────────────────────────────────────
__device__ Cell    d_grid[H][W];
__device__ Soil    d_soil[H][W];
__device__ Genome  d_gpool[GPOOL_MAX];
__device__ int     d_gpool_size;
__device__ uint8_t d_sig[H][W];
__device__ Stats   d_stats;
__device__ Stats   d_count_stats;
__device__ uint64_t d_tick;
__device__ SimParams d_params;

// RNG на каждый поток
__device__ curandState d_rng[H][W];

// ─── wrap координат ────────────────────────────────────────────
__device__ inline int wx(int x){ return ((x % W) + W) % W; }
__device__ inline int wy(int y){ return ((y % H) + H) % H; }

__device__ inline bool valid_genome_idx(int gi){
    return gi >= 0 && gi < GPOOL_MAX && gi < d_gpool_size;
}

__device__ int occupied_neighbors8(int x, int y){
    int n = 0;
    for(int dy=-1; dy<=1; dy++){
        for(int dx=-1; dx<=1; dx++){
            if(dx == 0 && dy == 0) continue;
            if(d_grid[wy(y+dy)][wx(x+dx)].type != CT_EMPTY) n++;
        }
    }
    return n;
}

__device__ float light_for_y(int y, float phase){
    float depth = (float)y / (float)(H - 1);
    float exposure = 0.06f + 0.94f * expf(-4.0f * depth);
    return phase * exposure * d_params.light_scale;
}

__device__ uint8_t random_action(curandState* rng){
    int r = curand(rng) % 100;
    if(r < 8)   return ACT_WAIT;
    if(r < 12)  return ACT_SLEEP;
    if(r < 36)  return (uint8_t)(ACT_GROW_LEAF_U + (curand(rng) % 4));
    if(r < 53)  return (uint8_t)(ACT_GROW_ROOT_D + (curand(rng) % 3));
    if(r < 63)  return (uint8_t)(ACT_GROW_ANT_U + (curand(rng) % 3));
    if(r < 70)  return (uint8_t)(ACT_GROW_DETOX_D + (curand(rng) % 3));
    if(r < 82)  return (uint8_t)(ACT_SPROUT_U + (curand(rng) % 4));
    if(r < 89)  return (uint8_t)(ACT_SHOOT_U + (curand(rng) % 4));
    if(r < 93)  return (uint8_t)(ACT_EAT_U + (curand(rng) % 4));
    if(r < 95)  return ACT_SKIP_LOW;
    if(r < 96)  return ACT_SKIP_HIGH;
    if(r < 97)  return ACT_SKIP_CROWD;
    if(r < 98)  return ACT_SKIP_TOXIC;
    if(r < 99)  return ACT_JUMP_B8;
    return ACT_JUMP_F8;
}

__device__ void make_seed_genome(Genome& g, uint8_t clan, curandState* rng){
    g.len = 24;
    g.clan = clan;
    for(int i=0; i<GENOME_MAX; i++) g.code[i] = ACT_WAIT;

    const uint8_t base0[24] = {
        ACT_GROW_LEAF_U, ACT_GROW_ROOT_D, ACT_GROW_LEAF_L, ACT_GROW_LEAF_R,
        ACT_GROW_ANT_U, ACT_SKIP_LOW, ACT_SPROUT_R, ACT_SKIP_LOW,
        ACT_SPROUT_L, ACT_GROW_DETOX_D, ACT_SKIP_CROWD, ACT_SHOOT_U,
        ACT_WAIT, ACT_WAIT, ACT_WAIT, ACT_WAIT,
        ACT_JUMP_B8, ACT_WAIT, ACT_WAIT, ACT_WAIT,
        ACT_WAIT, ACT_WAIT, ACT_WAIT, ACT_WAIT
    };
    const uint8_t base1[24] = {
        ACT_GROW_LEAF_U, ACT_GROW_LEAF_U, ACT_GROW_ROOT_D, ACT_GROW_ROOT_L,
        ACT_GROW_ROOT_R, ACT_SKIP_LOW, ACT_SPROUT_U, ACT_SKIP_LOW,
        ACT_SPROUT_D, ACT_GROW_DETOX_L, ACT_GROW_ANT_R, ACT_SHOOT_R,
        ACT_WAIT, ACT_WAIT, ACT_WAIT, ACT_WAIT,
        ACT_JUMP_B8, ACT_WAIT, ACT_WAIT, ACT_WAIT,
        ACT_WAIT, ACT_WAIT, ACT_WAIT, ACT_WAIT
    };
    const uint8_t base2[24] = {
        ACT_GROW_LEAF_U, ACT_GROW_ROOT_D, ACT_GROW_ANT_L, ACT_GROW_ANT_R,
        ACT_SKIP_LOW, ACT_SPROUT_L, ACT_SKIP_TOXIC, ACT_GROW_DETOX_R,
        ACT_SKIP_HIGH, ACT_EAT_R, ACT_SKIP_LOW, ACT_SHOOT_L,
        ACT_WAIT, ACT_WAIT, ACT_WAIT, ACT_WAIT,
        ACT_JUMP_B8, ACT_WAIT, ACT_WAIT, ACT_WAIT,
        ACT_WAIT, ACT_WAIT, ACT_WAIT, ACT_WAIT
    };
    const uint8_t base3[24] = {
        ACT_GROW_LEAF_L, ACT_GROW_LEAF_R, ACT_GROW_ROOT_D, ACT_GROW_DETOX_D,
        ACT_GROW_ANT_U, ACT_SKIP_LOW, ACT_SPROUT_D, ACT_SKIP_LOW,
        ACT_SPROUT_R, ACT_SKIP_CROWD, ACT_SHOOT_D, ACT_SLEEP,
        ACT_WAIT, ACT_WAIT, ACT_WAIT, ACT_WAIT,
        ACT_JUMP_B8, ACT_WAIT, ACT_WAIT, ACT_WAIT,
        ACT_WAIT, ACT_WAIT, ACT_WAIT, ACT_WAIT
    };

    const uint8_t* src = base0;
    if(clan == 1) src = base1;
    else if(clan == 2) src = base2;
    else if(clan == 3) src = base3;
    for(int i=0; i<24; i++) g.code[i] = src[i];

    for(int i=0; i<24; i++){
        if(curand_uniform(rng) < 0.08f) g.code[i] = random_action(rng);
    }
}

// ─── инициализация RNG ─────────────────────────────────────────
__global__ void kernel_init_rng(unsigned long long seed){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x>=W||y>=H) return;
    curand_init(seed, y*W+x, 0, &d_rng[y][x]);
}

// ─── аллокация генома (атомарная) ─────────────────────────────
__device__ int alloc_genome(){
    int old = d_gpool_size;
    while(old < GPOOL_MAX){
        int assumed = old;
        old = atomicCAS(&d_gpool_size, assumed, assumed + 1);
        if(old == assumed) return assumed;
    }
    return -1;
}

// ─── мутация генома ────────────────────────────────────────────
__device__ Genome mutate(const Genome& g, curandState* rng){
    Genome ng = g;
    float r = curand_uniform(rng);
    float ms = fminf(fmaxf(d_params.mutation_scale, 0.0f), 5.0f);
    float add_t = 0.002f * ms;
    float del_t = 0.004f * ms;
    float swap_t = 0.04f * ms;
    float edit_t = 0.25f * ms;
    int len = ng.len;
    if(len < 1) len = 1;

    if(r < add_t && len < GENOME_MAX){
        ng.code[len] = random_action(rng);
        ng.len = len + 1;
    } else if(r < del_t && len > 8){
        ng.len = len - 1;
    } else if(r < swap_t){
        int a = curand(rng) % len;
        int b = curand(rng) % len;
        uint8_t tmp = ng.code[a]; ng.code[a] = ng.code[b]; ng.code[b] = tmp;
    } else if(r < edit_t){
        int i = curand(rng) % len;
        ng.code[i] = random_action(rng);
    }
    return ng;
}

// ─── инициализация мира ────────────────────────────────────────
__global__ void kernel_init_world(unsigned long long seed){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x>=W||y>=H) return;

    curandState rng;
    curand_init(seed + 1, y*W+x, 0, &rng);

    // почва
    d_soil[y][x].organic = 0.02f + curand_uniform(&rng) * 0.08f;
    d_soil[y][x].charge  = 0.06f + curand_uniform(&rng) * 0.08f;
    d_soil[y][x].toxin   = (curand_uniform(&rng) < 0.03f) ? curand_uniform(&rng) * 0.08f : 0.0f;
    d_soil[y][x].light   = light_for_y(y, 4.0f);

    // пусто
    d_grid[y][x] = Cell{};
    d_sig[y][x]  = 0;

    // Посев не полностью случайный: каждый клан стартует с жизнеспособной
    // программы, а небольшая начальная вариативность дает материал отбору.
    if(x%10==1 && y%10==1){
        int gi = alloc_genome();
        if(gi >= 0){
            Genome g;
            uint8_t clan = (uint8_t)((x * NUM_CLANS) / W);
            make_seed_genome(g, clan, &rng);
            d_gpool[gi] = g;

            Cell c{};
            c.type       = CT_SPROUT;
            c.energy     = 55.f + curand_uniform(&rng) * 35.f;
            c.genome_idx = gi;
            c.age_limit  = 900 + (int)(curand_uniform(&rng) * 700);
            d_grid[y][x] = c;
        }
    }
}

// ─── обновление света ──────────────────────────────────────────
__global__ void kernel_update_light(uint64_t tick){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x>=W||y>=H) return;

    static const float PHASES[] = {4.0f, 2.0f, 3.0f, 2.0f, 3.0f, 2.0f};
    float tf = PHASES[(tick / 300) % 6];
    d_soil[y][x].light = light_for_y(y, tf);
}

// ─── передача энергии ──────────────────────────────────────────
__device__ void push_energy(int x, int y, float rate, float cap){
    Cell& src = d_grid[y][x];
    const int dx[4]={0,0,-1,1}, dy[4]={-1,1,0,0};
    for(int d=0;d<4;d++){
        int nx=wx(x+dx[d]), ny=wy(y+dy[d]);
        Cell& nb = d_grid[ny][nx];
        CellType t = nb.type;
        if(t==CT_WOOD||t==CT_SPROUT||t==CT_LEAF||t==CT_ROOT||t==CT_ANTENNA){
            float tr = fminf(src.energy * rate, cap);
            if(nb.energy < src.energy){
                atomicAdd(&nb.energy, tr);
                atomicAdd(&src.energy, -tr);
            }
        }
    }
}

// ─── пассивные клетки ──────────────────────────────────────────
__global__ void kernel_passive(uint64_t tick){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x>=W||y>=H) return;

    Cell& c = d_grid[y][x];
    Soil& s = d_soil[y][x];
    if(c.type==CT_EMPTY||c.type==CT_SPROUT) return;

    c.age++;
    if(c.age > c.age_limit){
        // смерть
        for(int dy2=-1;dy2<=1;dy2++) for(int dx2=-1;dx2<=1;dx2++)
            atomicAdd(&d_soil[wy(y+dy2)][wx(x+dx2)].organic, c.energy*0.04f/9.f);
        atomicAdd(&d_stats.deaths, 1);
        c = Cell{};
        return;
    }

    switch(c.type){
        case CT_LEAF: {
            bool open = (d_grid[wy(y-1)][x].type == CT_EMPTY);
            bool nb_leaf = false;
            const int dx[4]={0,0,-1,1}, dy[4]={-1,1,0,0};
            for(int d=0;d<4;d++)
                if(d_grid[wy(y+dy[d])][wx(x+dx[d])].type==CT_LEAF){ nb_leaf=true; break; }
            c.energy += (open ? (nb_leaf ? s.light*0.35f : s.light*2.2f) : s.light*0.25f) * d_params.leaf_gain;
            push_energy(x,y,0.6f,12.f);
            c.energy -= 0.22f * d_params.passive_metabolism;
            break;
        }
        case CT_ROOT: {
            float e = fminf(s.organic, 1.0f);
            c.energy += e * 1.9f * d_params.root_gain; s.organic -= e;
            push_energy(x,y,0.5f,6.f);
            c.energy -= 0.16f * d_params.passive_metabolism;
            break;
        }
        case CT_ANTENNA: {
            float e = fminf(s.charge, 0.9f);
            c.energy += e * 1.4f * d_params.antenna_gain; s.charge -= e;
            push_energy(x,y,0.5f,4.f);
            c.energy -= 0.14f * d_params.passive_metabolism;
            break;
        }
        case CT_DETOX: {
            float e = fminf(s.toxin, 1.0f);
            c.energy += e * 2.f; s.toxin -= e;
            push_energy(x,y,0.4f,4.f);
            c.energy -= 0.10f * d_params.passive_metabolism;
            break;
        }
        case CT_WOOD:
            c.energy -= 0.02f * d_params.passive_metabolism;
            break;
        case CT_SEED:
            c.energy -= 0.01f * d_params.passive_metabolism;
            if(c.age > 35){
                c.type=CT_SPROUT;
                c.age_steps=0;
                c.ip=0;
                c.age=0;
                if(c.age_limit < 500) c.age_limit = 900;
            }
            break;
        default: break;
    }

    c.energy = fminf(c.energy, 180.f);

    int crowd = occupied_neighbors8(x, y) - 3;
    if(crowd > 0){
        float type_mul = (c.type == CT_WOOD) ? 0.35f : 1.0f;
        c.energy -= d_params.crowd_penalty * crowd * type_mul;
    }

    // яды
    if(s.organic>0.7f && c.type!=CT_ROOT)    c.energy -= 1.0f * d_params.toxin_scale;
    if(s.charge >0.7f && c.type!=CT_ANTENNA) c.energy -= 1.0f * d_params.toxin_scale;
    if(s.toxin  >0.6f && c.type!=CT_DETOX)   c.energy -= 1.2f * d_params.toxin_scale;

    if(c.energy <= 0.f){
        atomicAdd(&d_soil[y][x].organic, 0.04f);
        atomicAdd(&d_soil[y][x].toxin,   0.01f);
        atomicAdd(&d_stats.deaths, 1);
        c = Cell{};
    }
}

// ─── отростки ──────────────────────────────────────────────────
__global__ void kernel_sprout(uint64_t tick){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x>=W||y>=H) return;

    Cell& c = d_grid[y][x];
    if(c.type != CT_SPROUT) return;

    curandState* rng = &d_rng[y][x];
    Soil& s = d_soil[y][x];

    if(!valid_genome_idx(c.genome_idx)){ c=Cell{}; return; }
    Genome& g = d_gpool[c.genome_idx];
    int glen = g.len > 0 ? g.len : 1;
    if(glen > GENOME_MAX) glen = GENOME_MAX;

    // сон
    if(c.sleeping){
        c.age++;
        c.sleep_ticks--;
        c.energy += s.light * 0.08f;
        c.energy -= 0.03f * d_params.sprout_metabolism;
        if(c.sleep_ticks <= 0) c.sleeping = 0;
        if(c.energy <= 0.f){ s.organic += 0.05f; c=Cell{}; }
        return;
    }

    // фотосинтез отростка
    c.energy += s.light * 0.16f;

    // старость
    c.age++;
    if(c.age > c.age_limit){
        s.organic += c.energy * 0.12f;
        s.toxin   += 0.03f;
        atomicAdd(&d_stats.deaths, 1);
        c = Cell{}; return;
    }

    c.energy -= 0.24f * d_params.sprout_metabolism;
    int crowd = occupied_neighbors8(x, y);
    if(crowd > 2) c.energy -= d_params.crowd_penalty * (crowd - 2) * 1.35f;
    if(c.energy <= 0.f){
        s.organic += 0.03f;
        atomicAdd(&d_stats.deaths, 1);
        c = Cell{}; return;
    }

    if(c.skip_next){ c.skip_next=0; c.ip=(c.ip+1)%glen; c.age_steps++; return; }

    uint8_t act = g.code[c.ip % GENOME_MAX];
    bool was_wait = false;

    const int DX[4]={0,0,-1,1}, DY[4]={-1,1,0,0};

    auto try_grow_passive = [&](int dx, int dy, CellType t) -> bool {
        int nx=wx(x+dx), ny=wy(y+dy);
        Cell& nb = d_grid[ny][nx];
        float cost = d_params.growth_cost;
        if(nb.type!=CT_EMPTY || c.energy<cost + 1.f) return false;
        c.energy -= cost;
        Cell nc{};
        nc.type      = t;
        nc.energy    = 4.5f;
        nc.genome_idx = c.genome_idx;
        nc.age_limit = 250 + (int)(curand_uniform(rng)*200);
        nb = nc;
        return true;
    };

    auto try_sprout = [&](int dx, int dy) -> bool {
        int nx=wx(x+dx), ny=wy(y+dy);
        Cell& nb = d_grid[ny][nx];
        float cost = d_params.sprout_cost;
        if(nb.type!=CT_EMPTY || c.energy<cost + 4.f) return false;
        int target_crowd = occupied_neighbors8(nx, ny);
        if(target_crowd > 5 && c.energy < 95.f) return false;
        int gi = alloc_genome();
        if(gi >= 0) d_gpool[gi] = mutate(g, rng);
        else gi = c.genome_idx;
        c.energy -= cost + d_params.crowd_penalty * fmaxf(0, target_crowd - 2) * 5.f;
        Cell nc{};
        nc.type       = CT_SPROUT;
        nc.energy     = 24.f;
        nc.genome_idx = gi;
        nc.age_limit  = 800 + (int)(curand_uniform(rng)*800);
        nb = nc;
        atomicAdd(&d_stats.births, 1);
        return true;
    };

    switch((Action)act){
        case ACT_WAIT:  was_wait=true; break;
        case ACT_SLEEP: c.sleeping=1; c.sleep_ticks=20; was_wait=true; break;

        case ACT_GROW_LEAF_U: try_grow_passive( 0,-1,CT_LEAF);    break;
        case ACT_GROW_LEAF_D: try_grow_passive( 0, 1,CT_LEAF);    break;
        case ACT_GROW_LEAF_L: try_grow_passive(-1, 0,CT_LEAF);    break;
        case ACT_GROW_LEAF_R: try_grow_passive( 1, 0,CT_LEAF);    break;
        case ACT_GROW_ROOT_D: try_grow_passive( 0, 1,CT_ROOT);    break;
        case ACT_GROW_ROOT_L: try_grow_passive(-1, 0,CT_ROOT);    break;
        case ACT_GROW_ROOT_R: try_grow_passive( 1, 0,CT_ROOT);    break;
        case ACT_GROW_ANT_U:  try_grow_passive( 0,-1,CT_ANTENNA); break;
        case ACT_GROW_ANT_L:  try_grow_passive(-1, 0,CT_ANTENNA); break;
        case ACT_GROW_ANT_R:  try_grow_passive( 1, 0,CT_ANTENNA); break;
        case ACT_GROW_DETOX_D:try_grow_passive( 0, 1,CT_DETOX);   break;
        case ACT_GROW_DETOX_L:try_grow_passive(-1, 0,CT_DETOX);   break;
        case ACT_GROW_DETOX_R:try_grow_passive( 1, 0,CT_DETOX);   break;

        case ACT_SPROUT_U: try_sprout( 0,-1); break;
        case ACT_SPROUT_D: try_sprout( 0, 1); break;
        case ACT_SPROUT_L: try_sprout(-1, 0); break;
        case ACT_SPROUT_R: try_sprout( 1, 0); break;

        case ACT_SHOOT_U: case ACT_SHOOT_D: case ACT_SHOOT_L: case ACT_SHOOT_R: {
            int di = act - ACT_SHOOT_U;
            int nx=wx(x+DX[di]), ny=wy(y+DY[di]);
            float cost = d_params.seed_cost;
            if(d_grid[ny][nx].type==CT_EMPTY && c.energy>=cost + 4.f){
                int target_crowd = occupied_neighbors8(nx, ny);
                if(target_crowd > 6 && c.energy < 80.f) break;
                int gi=alloc_genome();
                if(gi >= 0) d_gpool[gi]=mutate(g,rng);
                else gi = c.genome_idx;
                if(valid_genome_idx(gi)){
                    c.energy-=cost + d_params.crowd_penalty * fmaxf(0, target_crowd - 2) * 3.f;
                    Cell seed{};
                    seed.type=CT_SEED; seed.energy=16.f;
                    seed.genome_idx=gi; seed.age_limit=900;
                    d_grid[ny][nx]=seed;
                    atomicAdd(&d_stats.births, 1);
                }
            }
            break;
        }
        case ACT_EAT_U: case ACT_EAT_D: case ACT_EAT_L: case ACT_EAT_R: {
            int di = act - ACT_EAT_U;
            int nx=wx(x+DX[di]), ny=wy(y+DY[di]);
            Cell& nb=d_grid[ny][nx];
            if(nb.type!=CT_EMPTY&&nb.type!=CT_ROOT&&nb.type!=CT_WOOD){
                c.energy += nb.energy*0.6f;
                atomicAdd(&d_soil[ny][nx].organic, nb.energy*0.2f);
                atomicAdd(&d_stats.deaths, 1);
                nb=Cell{};
            }
            break;
        }
        case ACT_SKIP_LOW:   if(c.energy<40.f)  c.skip_next=1; break;
        case ACT_SKIP_HIGH:  if(c.energy>130.f) c.skip_next=1; break;
        case ACT_SKIP_CROWD: {
            bool cr=true;
            for(int d=0;d<4;d++) if(d_grid[wy(y+DY[d])][wx(x+DX[d])].type==CT_EMPTY){cr=false;break;}
            if(cr) c.skip_next=1; break;
        }
        case ACT_SKIP_ALONE: {
            bool al=true;
            for(int d=0;d<4;d++) if(d_grid[wy(y+DY[d])][wx(x+DX[d])].type!=CT_EMPTY){al=false;break;}
            if(al) c.skip_next=1; break;
        }
        case ACT_SKIP_TOXIC: if(s.toxin>0.4f||s.organic>0.5f) c.skip_next=1; break;
        case ACT_JUMP_B8:
            c.ip=((c.ip-8)%glen+glen)%glen; c.age_steps++; return;
        case ACT_JUMP_F8:
            c.ip=(c.ip+8)%glen; c.age_steps++; return;
        case ACT_SIG_A:      c.signal|=1; d_sig[y][x]|=1; break;
        case ACT_SIG_B:      c.signal|=2; d_sig[y][x]|=2; break;
        case ACT_SKIP_NSIG_A: if(!(d_sig[y][x]&1)) c.skip_next=1; break;
        case ACT_SKIP_NSIG_B: if(!(d_sig[y][x]&2)) c.skip_next=1; break;
        default: break;
    }

    if(!was_wait) c.age_steps++;
    c.ip = (c.ip+1) % glen;

    if(s.organic>0.7f) c.energy -= 0.7f * d_params.toxin_scale;
    if(s.charge >0.7f) c.energy -= 0.7f * d_params.toxin_scale;
    if(s.toxin  >0.6f) c.energy -= 1.0f * d_params.toxin_scale;
    c.energy = fminf(c.energy, 220.f);

    if(c.energy<=0.f){
        s.organic+=0.08f; s.toxin+=0.03f;
        atomicAdd(&d_stats.deaths,1);
        c=Cell{};
    }
}

// ─── диффузия почвы ────────────────────────────────────────────
__global__ void kernel_diffuse(){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x>=W||y>=H) return;
    Soil& s = d_soil[y][x];
    s.organic *= 0.9996f;
    s.charge   = s.charge*0.9996f + 0.00008f*(0.1f-s.charge);
    s.toxin   *= 0.9992f;
}

// ─── статистика ────────────────────────────────────────────────
__global__ void kernel_stats(){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x>=W||y>=H) return;
    Cell& c = d_grid[y][x];
    if(c.type==CT_EMPTY) return;
    atomicAdd(&d_count_stats.cells,1);
    if(c.type==CT_SPROUT) atomicAdd(&d_count_stats.sprouts,1);
    if(c.type==CT_LEAF)   atomicAdd(&d_count_stats.leaves,1);
    if(valid_genome_idx(c.genome_idx))
        atomicAdd(&d_count_stats.clan_count[d_gpool[c.genome_idx].clan%NUM_CLANS],1);
}

__device__ inline uint32_t rgb(int r, int g, int b){
    r = max(0, min(255, r));
    g = max(0, min(255, g));
    b = max(0, min(255, b));
    return 0xFF000000 | ((uint32_t)r << 16) | ((uint32_t)g << 8) | (uint32_t)b;
}

__device__ inline uint32_t shade(uint32_t col, float mul, int add=0){
    int r = (int)(((col >> 16) & 0xFF) * mul) + add;
    int g = (int)(((col >> 8) & 0xFF) * mul) + add;
    int b = (int)((col & 0xFF) * mul) + add;
    return rgb(r,g,b);
}

__device__ inline uint32_t mix_rgb(uint32_t a, uint32_t b, float t){
    t = fminf(1.0f, fmaxf(0.0f, t));
    int ar=(a>>16)&0xFF, ag=(a>>8)&0xFF, ab=a&0xFF;
    int br=(b>>16)&0xFF, bg=(b>>8)&0xFF, bb=b&0xFF;
    return rgb((int)(ar + (br-ar)*t), (int)(ag + (bg-ag)*t), (int)(ab + (bb-ab)*t));
}

__device__ inline float cell_noise(int x, int y){
    uint32_t n = (uint32_t)(x * 374761393u + y * 668265263u);
    n = (n ^ (n >> 13u)) * 1274126177u;
    n ^= n >> 16u;
    return (float)(n & 255u) / 255.0f;
}

__device__ uint32_t clan_color(int clan){
    const uint32_t CC[8]={0xFFFF5050,0xFF50D250,0xFF5082FF,0xFFFFC828,
                          0xFFFF70C8,0xFF28E1C8,0xFFDC8C28,0xFFAA50FF};
    return CC[clan & 7];
}

// ─── рендер в пиксельный буфер ─────────────────────────────────
// Мир закольцован по обеим осям. При отдалении видно повторы мира (тайлы).
// На границах между копиями рисуется тонкая полоса для наглядности стыков.
__global__ void kernel_render(uint32_t* pixels, int pw, int ph,
                               float cam_x, float cam_y, float zoom, int rmode){
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if(px>=pw||py>=ph) return;

    // мировые координаты с wrap (тайлинг)
    float fx = cam_x + px/zoom;
    float fy = cam_y + py/zoom;
    int gx = ((int)floorf(fx) % W + W) % W;
    int gy = ((int)floorf(fy) % H + H) % H;

    // граница между тайлами — тонкая серая линия на x=0 и y=0
    // видна только при zoom < ~2 (когда виден не один тайл, или близко к краю)
    bool on_border = (gx == 0 || gy == 0);

    const Cell& c = d_grid[gy][gx];
    const Soil& s = d_soil[gy][gx];

    float n = cell_noise(gx,gy);
    uint32_t col = rgb(10 + (int)(s.organic*35.f) + (int)(n*5.f),
                       10 + (int)(s.light*6.f) + (int)(s.organic*20.f),
                       16 + (int)(s.charge*32.f) + (int)(s.toxin*40.f));

    if(rmode == 0){ // клетки
        if(c.type != CT_EMPTY){
            float energy = fminf(2.2f, fmaxf(0.35f, c.energy / 70.0f));
            float age = fminf(1.0f, fmaxf(0.0f, (float)c.age / fmaxf(1.0f, (float)c.age_limit)));
            float vigor = 0.72f + energy * 0.22f - age * 0.18f;
            bool edge = false;
            const int dx4[4]={0,0,-1,1}, dy4[4]={-1,1,0,0};
            for(int d=0; d<4; d++){
                CellType nt = d_grid[wy(gy+dy4[d])][wx(gx+dx4[d])].type;
                if(nt == CT_EMPTY || nt != c.type){ edge = true; break; }
            }
            float u = fx - floorf(fx);
            float v = fy - floorf(fy);
            bool rim = zoom >= 3.0f && (u < 0.10f || v < 0.10f || u > 0.90f || v > 0.90f);

            switch(c.type){
                case CT_SPROUT: {
                    int clan = valid_genome_idx(c.genome_idx) ? d_gpool[c.genome_idx].clan%NUM_CLANS : 0;
                    col = mix_rgb(clan_color(clan), 0xFFFFFFFF, 0.10f + fminf(0.35f, energy*0.08f));
                    break;
                }
                case CT_LEAF:
                    col = rgb(24 + (int)(s.light*10.f), 115 + (int)(energy*38.f), 34 + (int)(n*26.f));
                    break;
                case CT_ROOT:
                    col = rgb(118 + (int)(energy*22.f), 70 + (int)(s.organic*45.f), 30 + (int)(n*24.f));
                    break;
                case CT_ANTENNA:
                    col = rgb(50 + (int)(s.charge*80.f), 110 + (int)(energy*18.f), 190 + (int)(energy*32.f));
                    break;
                case CT_DETOX:
                    col = rgb(150 + (int)(energy*28.f), 55 + (int)(n*30.f), 185 + (int)(s.toxin*75.f));
                    break;
                case CT_WOOD:
                    col = rgb(76 + (int)(n*22.f), 45 + (int)(energy*10.f), 22 + (int)(n*14.f));
                    if(zoom >= 5.0f && ((int)(u*6.0f) % 2 == 0)) col = shade(col,0.78f);
                    break;
                case CT_SEED:
                    col = rgb(215 + (int)(energy*12.f), 190 + (int)(n*35.f), 70);
                    if(zoom >= 4.0f){
                        float du = u - 0.5f, dv = v - 0.5f;
                        if(du*du + dv*dv > 0.20f) col = shade(col,0.55f);
                    }
                    break;
                default: break;
            }
            col = shade(col, vigor);
            if(edge) col = mix_rgb(col, 0xFF050508, 0.28f);
            if(rim) col = shade(col, 0.55f);
            if(c.signal & 1) col = mix_rgb(col, 0xFFFF6060, 0.18f);
            if(c.signal & 2) col = mix_rgb(col, 0xFF60A0FF, 0.18f);
        }
    } else if(rmode==5){ // кланы
        if(c.type!=CT_EMPTY){
            int clan = valid_genome_idx(c.genome_idx) ?
                       d_gpool[c.genome_idx].clan%NUM_CLANS : 0;
            col = clan_color(clan);
        }
    } else {
        float v=0.f;
        if(rmode==1) v=fminf(1.f,s.organic*2.f);
        if(rmode==2) v=fminf(1.f,s.charge*4.f);
        if(rmode==3) v=fminf(1.f,s.toxin*5.f);
        if(rmode==4) v=fminf(1.f,s.light*0.4f);
        uint8_t iv=(uint8_t)(v*255);
        if(rmode==1) col=0xFF000000|(iv<<16)|((iv/3)<<8);
        if(rmode==2) col=0xFF000000|((iv/2)<<8)|iv;
        if(rmode==3) col=0xFF000000|((iv/2)<<16)|iv;
        if(rmode==4) col=0xFF000000|(iv<<16)|(iv<<8)|(iv/2);
        if(c.type!=CT_EMPTY){
            // клетки поверх
            uint8_t r2=(col>>16)&0xFF, g2=(col>>8)&0xFF, b2=col&0xFF;
            col = 0xFF000000|((uint32_t)fminf(255,r2+60)<<16)|
                             ((uint32_t)fminf(255,g2+60)<<8)|
                             (uint32_t)fminf(255,b2+60);
        }
    }

    // тонкая граница между тайлами (смешиваем с текущим цветом)
    // показывается только при зуме < 2 (когда может быть виден не один тайл)
    if(on_border && zoom < 2.0f){
        uint8_t r2=(col>>16)&0xFF, g2=(col>>8)&0xFF, b2=col&0xFF;
        col = 0xFF000000|((uint32_t)((r2+80)/2)<<16)|
                         ((uint32_t)((g2+80)/2)<<8)|
                         (uint32_t)((b2+100)/2);
    }

    pixels[py*pw+px] = col;
}

// ─── C-интерфейс для main.cpp ──────────────────────────────────
extern "C" {

void sim_init(unsigned long long seed){
    dim3 block(16,16);
    dim3 grid_dim((W+15)/16,(H+15)/16);
    kernel_init_rng<<<grid_dim,block>>>(seed);
    int zero=0;
    cudaMemcpyToSymbol(d_gpool_size,&zero,sizeof(int));
    uint64_t tzero=0;
    cudaMemcpyToSymbol(d_tick,&tzero,sizeof(uint64_t));
    Stats szero{};
    cudaMemcpyToSymbol(d_stats,&szero,sizeof(Stats));
    cudaMemcpyToSymbol(d_count_stats,&szero,sizeof(Stats));
    SimParams defaults{};
    defaults.crowd_penalty = 0.18f;
    defaults.light_scale = 1.0f;
    defaults.leaf_gain = 1.0f;
    defaults.root_gain = 1.0f;
    defaults.antenna_gain = 1.0f;
    defaults.sprout_metabolism = 1.0f;
    defaults.passive_metabolism = 1.0f;
    defaults.growth_cost = 6.0f;
    defaults.sprout_cost = 48.0f;
    defaults.seed_cost = 20.0f;
    defaults.mutation_scale = 1.0f;
    defaults.toxin_scale = 1.0f;
    cudaMemcpyToSymbol(d_params,&defaults,sizeof(SimParams));
    kernel_init_world<<<grid_dim,block>>>(seed);
    cudaError_t err = cudaDeviceSynchronize();
    if(err != cudaSuccess)
        fprintf(stderr, "sim_init error: %s\n", cudaGetErrorString(err));
    err = cudaGetLastError();
    if(err != cudaSuccess)
        fprintf(stderr, "sim_init kernel error: %s\n", cudaGetErrorString(err));
}

void sim_tick(){
    dim3 block(16,16);
    dim3 grid_dim((W+15)/16,(H+15)/16);

    uint64_t tick_val;
    cudaMemcpyFromSymbol(&tick_val,d_tick,sizeof(uint64_t));

    kernel_update_light<<<grid_dim,block>>>(tick_val);
    kernel_passive<<<grid_dim,block>>>(tick_val);
    kernel_sprout<<<grid_dim,block>>>(tick_val);
    kernel_diffuse<<<grid_dim,block>>>();

    tick_val++;
    cudaMemcpyToSymbol(d_tick,&tick_val,sizeof(uint64_t));
}

void sim_set_params(const SimParams* params){
    SimParams p = *params;
    p.crowd_penalty = fminf(fmaxf(p.crowd_penalty, 0.0f), 3.0f);
    p.light_scale = fminf(fmaxf(p.light_scale, 0.0f), 10.0f);
    p.leaf_gain = fminf(fmaxf(p.leaf_gain, 0.0f), 8.0f);
    p.root_gain = fminf(fmaxf(p.root_gain, 0.0f), 8.0f);
    p.antenna_gain = fminf(fmaxf(p.antenna_gain, 0.0f), 8.0f);
    p.sprout_metabolism = fminf(fmaxf(p.sprout_metabolism, 0.05f), 10.0f);
    p.passive_metabolism = fminf(fmaxf(p.passive_metabolism, 0.05f), 10.0f);
    p.growth_cost = fminf(fmaxf(p.growth_cost, 0.5f), 40.0f);
    p.sprout_cost = fminf(fmaxf(p.sprout_cost, 4.0f), 240.0f);
    p.seed_cost = fminf(fmaxf(p.seed_cost, 1.0f), 160.0f);
    p.mutation_scale = fminf(fmaxf(p.mutation_scale, 0.0f), 20.0f);
    p.toxin_scale = fminf(fmaxf(p.toxin_scale, 0.0f), 12.0f);
    cudaMemcpyToSymbol(d_params,&p,sizeof(SimParams));
}

void sim_render(uint32_t* d_pixels, int pw, int ph,
                float cam_x, float cam_y, float zoom, int rmode){
    dim3 block(16,16);
    dim3 grid_dim((pw+15)/16,(ph+15)/16);
    kernel_render<<<grid_dim,block>>>(d_pixels,pw,ph,cam_x,cam_y,zoom,rmode);
}

void sim_get_stats(Stats* out){
    Stats zero{};
    cudaMemcpyToSymbol(d_count_stats,&zero,sizeof(Stats));
    dim3 block(16,16);
    dim3 grid_dim((W+15)/16,(H+15)/16);
    kernel_stats<<<grid_dim,block>>>();
    cudaDeviceSynchronize();
    cudaMemcpyFromSymbol(out,d_count_stats,sizeof(Stats));
    Stats life{};
    cudaMemcpyFromSymbol(&life,d_stats,sizeof(Stats));
    out->births = life.births;
    out->deaths = life.deaths;
}

uint64_t sim_get_tick(){
    uint64_t t;
    cudaMemcpyFromSymbol(&t,d_tick,sizeof(uint64_t));
    return t;
}

} // extern "C"
