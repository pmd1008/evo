#include "types.h"
#include <curand_kernel.h>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <cstdio>

// ─── GPU буферы ────────────────────────────────────────────────
__device__ Cell    d_grid[H][W];
__device__ Soil    d_soil[H][W];
__device__ Genome  d_gpool[1 << 17];  // 128k геномов
__device__ int     d_gpool_size;
__device__ uint8_t d_sig[H][W];
__device__ Stats   d_stats;
__device__ uint64_t d_tick;

// RNG на каждый поток
__device__ curandState d_rng[H][W];

// ─── wrap координат ────────────────────────────────────────────
__device__ inline int wx(int x){ return ((x % W) + W) % W; }
__device__ inline int wy(int y){ return ((y % H) + H) % H; }

// ─── инициализация RNG ─────────────────────────────────────────
__global__ void kernel_init_rng(unsigned long long seed){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x>=W||y>=H) return;
    curand_init(seed, y*W+x, 0, &d_rng[y][x]);
}

// ─── аллокация генома (атомарная) ─────────────────────────────
__device__ int alloc_genome(){
    return atomicAdd(&d_gpool_size, 1);
}

// ─── мутация генома ────────────────────────────────────────────
__device__ Genome mutate(const Genome& g, curandState* rng){
    Genome ng = g;
    float r = curand_uniform(rng);
    int len = ng.len;
    if(len < 1) len = 1;

    if(r < 0.002f && len < GENOME_MAX){
        ng.code[len] = (uint8_t)(curand(rng) % ACT_COUNT);
        ng.len = len + 1;
    } else if(r < 0.004f && len > 8){
        ng.len = len - 1;
    } else if(r < 0.04f){
        int a = curand(rng) % len;
        int b = curand(rng) % len;
        uint8_t tmp = ng.code[a]; ng.code[a] = ng.code[b]; ng.code[b] = tmp;
    } else if(r < 0.25f){
        int i = curand(rng) % len;
        ng.code[i] = (uint8_t)(curand(rng) % ACT_COUNT);
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
    d_soil[y][x].organic = curand_uniform(&rng) * 0.03f;
    d_soil[y][x].charge  = 0.05f + curand_uniform(&rng) * 0.05f;
    d_soil[y][x].toxin   = 0.0f;
    d_soil[y][x].light   = (y < H/4) ? 2.0f : 0.0f;

    // пусто
    d_grid[y][x] = Cell{};
    d_sig[y][x]  = 0;

    // посев — каждые 8 клеток
    if(x%8==1 && y%8==1){
        int gi = alloc_genome();
        if(gi < (1<<17)){
            Genome g;
            g.len  = 64;
            g.clan = (uint8_t)((x * NUM_CLANS) / W);
            for(int i=0;i<64;i++) g.code[i] = (uint8_t)(curand(&rng) % ACT_COUNT);
            d_gpool[gi] = g;

            Cell c{};
            c.type       = CT_SPROUT;
            c.energy     = 80.f;
            c.genome_idx = (int16_t)gi;
            c.age_limit  = 200;
            d_grid[y][x] = c;
        }
    }

    if(x==0 && y==0){
        d_gpool_size = 0;
        d_tick = 0;
        memset(&d_stats, 0, sizeof(d_stats));
    }
}

// ─── обновление света ──────────────────────────────────────────
__global__ void kernel_update_light(uint64_t tick){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x>=W||y>=H) return;

    static const float PHASES[] = {4.0f, 2.0f, 3.0f, 2.0f, 3.0f, 2.0f};
    float tf = PHASES[(tick / 300) % 6];
    d_soil[y][x].light = (y < H/4) ? tf : 0.0f;
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
            c.energy += open ? (nb_leaf ? 0.1f : s.light*2.0f) : s.light*0.2f;
            push_energy(x,y,0.6f,12.f);
            c.energy -= 0.4f;
            break;
        }
        case CT_ROOT: {
            float e = fminf(s.organic, 1.5f);
            c.energy += e * 1.5f; s.organic -= e;
            push_energy(x,y,0.5f,6.f);
            c.energy -= 0.3f;
            break;
        }
        case CT_ANTENNA: {
            float e = fminf(s.charge, 1.2f);
            c.energy += e * 1.2f; s.charge -= e;
            push_energy(x,y,0.5f,4.f);
            c.energy -= 0.3f;
            break;
        }
        case CT_DETOX: {
            float e = fminf(s.toxin, 1.0f);
            c.energy += e * 2.f; s.toxin -= e;
            push_energy(x,y,0.4f,4.f);
            c.energy -= 0.15f;
            break;
        }
        case CT_WOOD:
            c.energy -= 0.03f;
            break;
        case CT_SEED:
            c.energy -= 0.01f;
            if(c.age > 50){ c.type=CT_SPROUT; c.age_steps=0; c.ip=0; }
            break;
        default: break;
    }

    // яды
    if(s.organic>0.7f && c.type!=CT_ROOT)    c.energy -= 1.0f;
    if(s.charge >0.7f && c.type!=CT_ANTENNA) c.energy -= 1.0f;
    if(s.toxin  >0.6f && c.type!=CT_DETOX)   c.energy -= 1.2f;

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

    if(c.genome_idx < 0 || c.genome_idx >= d_gpool_size){ c=Cell{}; return; }
    Genome& g = d_gpool[c.genome_idx];
    int glen = g.len > 0 ? g.len : 1;

    // сон
    if(c.sleeping){
        c.sleep_ticks--;
        c.energy -= 0.05f;
        if(c.sleep_ticks <= 0) c.sleeping = 0;
        if(c.energy <= 0.f){ s.organic += 0.05f; c=Cell{}; }
        return;
    }

    // фотосинтез отростка
    c.energy += s.light * 0.3f;

    // старость
    if(c.age_steps >= glen){
        s.organic += c.energy * 0.2f;
        s.toxin   += 0.05f;
        atomicAdd(&d_stats.deaths, 1);
        c = Cell{}; return;
    }

    c.energy -= 0.8f;
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
        if(nb.type!=CT_EMPTY || c.energy<8.f) return false;
        c.energy -= 8.f;
        Cell nc{};
        nc.type      = t;
        nc.energy    = 6.f;
        nc.age_limit = 250 + (int)(curand_uniform(rng)*200);
        nb = nc;
        return true;
    };

    auto try_sprout = [&](int dx, int dy) -> bool {
        int nx=wx(x+dx), ny=wy(y+dy);
        Cell& nb = d_grid[ny][nx];
        if(nb.type!=CT_EMPTY || c.energy<60.f) return false;
        int gi = alloc_genome();
        if(gi >= (1<<17)) return false;
        d_gpool[gi] = mutate(g, rng);
        c.energy -= 60.f;
        Cell nc{};
        nc.type       = CT_SPROUT;
        nc.energy     = 15.f;
        nc.genome_idx = (int16_t)gi;
        nc.age_limit  = 200;
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
            if(d_grid[ny][nx].type==CT_EMPTY && c.energy>=20.f){
                int gi=alloc_genome();
                if(gi<(1<<17)){
                    d_gpool[gi]=mutate(g,rng);
                    c.energy-=18.f;
                    Cell seed{};
                    seed.type=CT_SEED; seed.energy=10.f;
                    seed.genome_idx=(int16_t)gi; seed.age_limit=200;
                    d_grid[ny][nx]=seed;
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

    if(s.organic>0.7f) c.energy -= 0.7f;
    if(s.charge >0.7f) c.energy -= 0.7f;
    if(s.toxin  >0.6f) c.energy -= 1.0f;

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
    atomicAdd(&d_stats.cells,1);
    if(c.type==CT_SPROUT) atomicAdd(&d_stats.sprouts,1);
    if(c.type==CT_LEAF)   atomicAdd(&d_stats.leaves,1);
    if(c.genome_idx>=0&&c.genome_idx<d_gpool_size)
        atomicAdd(&d_stats.clan_count[d_gpool[c.genome_idx].clan%NUM_CLANS],1);
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

    uint32_t col = 0xFF0C0C12; // фон мира

    if(rmode == 0){ // клетки
        switch(c.type){
            case CT_SPROUT: {
                const uint32_t CC[8]={0xFFFF5050,0xFF50D250,0xFF5082FF,0xFFFFC828,
                                      0xFFFF70C8,0xFF28E1C8,0xFFDC8C28,0xFFAA50FF};
                int clan = (c.genome_idx>=0&&c.genome_idx<d_gpool_size) ?
                           d_gpool[c.genome_idx].clan%NUM_CLANS : 0;
                col = CC[clan]; break;
            }
            case CT_LEAF:    col=0xFF28A028; break;
            case CT_ROOT:    col=0xFFA06428; break;
            case CT_ANTENNA: col=0xFF3C78C8; break;
            case CT_DETOX:   col=0xFFB43CC8; break;
            case CT_WOOD:    col=0xFF5A3719; break;
            case CT_SEED:    col=0xFFE6D250; break;
            default: break;
        }
    } else if(rmode==5){ // кланы
        if(c.type!=CT_EMPTY){
            const uint32_t CC[8]={0xFFFF5050,0xFF50D250,0xFF5082FF,0xFFFFC828,
                                  0xFFFF70C8,0xFF28E1C8,0xFFDC8C28,0xFFAA50FF};
            int clan = (c.genome_idx>=0&&c.genome_idx<d_gpool_size) ?
                       d_gpool[c.genome_idx].clan%NUM_CLANS : 0;
            col = CC[clan];
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

void sim_render(uint32_t* d_pixels, int pw, int ph,
                float cam_x, float cam_y, float zoom, int rmode){
    dim3 block(16,16);
    dim3 grid_dim((pw+15)/16,(ph+15)/16);
    kernel_render<<<grid_dim,block>>>(d_pixels,pw,ph,cam_x,cam_y,zoom,rmode);
}

void sim_get_stats(Stats* out){
    Stats zero{};
    cudaMemcpyToSymbol(d_stats,&zero,sizeof(Stats));
    dim3 block(16,16);
    dim3 grid_dim((W+15)/16,(H+15)/16);
    kernel_stats<<<grid_dim,block>>>();
    cudaDeviceSynchronize();
    cudaMemcpyFromSymbol(out,d_stats,sizeof(Stats));
}

uint64_t sim_get_tick(){
    uint64_t t;
    cudaMemcpyFromSymbol(&t,d_tick,sizeof(uint64_t));
    return t;
}

} // extern "C"
