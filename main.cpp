#include <SDL2/SDL.h>
#include <SDL2/SDL_ttf.h>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <string>
#include <algorithm>
#include <cuda_runtime.h>
#include "types.h"

// ─── C-интерфейс к CUDA ────────────────────────────────────────
extern "C" {
    void     sim_init(unsigned long long seed);
    void     sim_tick();
    void     sim_render(uint32_t* d_pixels, int pw, int ph,
                        float cam_x, float cam_y, float zoom, int rmode);
    void     sim_get_stats(Stats* out);
    uint64_t sim_get_tick();
}

// ─── константы окна ────────────────────────────────────────────
// Окно квадратное: viewport 600x600 (ровно мир 600x600) + панель 300px справа
static const int PANEL_W= 300;
static const int VIEW_W = W;       // 600
static const int VIEW_H = H;       // 600
static const int WIN_W  = VIEW_W + PANEL_W;  // 900
static const int WIN_H  = VIEW_H;            // 600

// ─── состояние ─────────────────────────────────────────────────
// Стартовая камера: cam=(0,0), zoom=1.0 → ровно 1 тайл мира в viewport
static float cam_x=0, cam_y=0, zoom=1.0f;
static bool  dragging=false;
static int   drag_sx=0, drag_sy=0;
static float drag_cx=0, drag_cy=0;
static int   sel_x=-1, sel_y=-1;
static int   rmode=0;
static bool  paused=false;
static int   tick_delay=0;
static int   ticks_per_frame=5;

static Stats  cur_stats{};
static Stats  max_stats{};
static uint64_t last_stats_tick=0;

// GPU пиксельный буфер
static uint32_t* d_pixels=nullptr;

// SDL текстура для отображения
static SDL_Texture* world_tex=nullptr;
static SDL_Renderer* g_rnd=nullptr;

static TTF_Font* fnt=nullptr;
static TTF_Font* fnt_sm=nullptr;

static void reset_camera(){
    cam_x = 0;
    cam_y = 0;
    zoom  = 1.0f;
}

// ─── шрифт ─────────────────────────────────────────────────────
static void load_font(){
    const char* paths[]={
        "/usr/share/fonts/liberation-mono/LiberationMono-Regular.ttf",
        "/usr/share/fonts/liberation/LiberationMono-Regular.ttf",
        "/usr/share/fonts/TTF/DejaVuSansMono.ttf",
        "/usr/share/fonts/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/google-noto/NotoSansMono-Regular.ttf",
        nullptr
    };
    for(int i=0;paths[i];i++){
        fnt=TTF_OpenFont(paths[i],14);
        if(fnt){ fnt_sm=TTF_OpenFont(paths[i],12); break; }
    }
}

static void txt(int x,int y,const char* s,SDL_Color c={210,210,220,255}){
    if(!fnt_sm||!s||!s[0]) return;
    SDL_Surface* sf=TTF_RenderUTF8_Blended(fnt_sm,s,c);
    if(!sf) return;
    SDL_Texture* tx=SDL_CreateTextureFromSurface(g_rnd,sf);
    SDL_Rect dst{x,y,sf->w,sf->h};
    SDL_RenderCopy(g_rnd,tx,nullptr,&dst);
    SDL_DestroyTexture(tx); SDL_FreeSurface(sf);
}

static void frect(int x,int y,int w,int h,SDL_Color c){
    SDL_SetRenderDrawColor(g_rnd,c.r,c.g,c.b,255);
    SDL_Rect r{x,y,w,h}; SDL_RenderFillRect(g_rnd,&r);
}

// ─── кнопки ────────────────────────────────────────────────────
struct Btn { SDL_Rect r; const char* label; bool toggle=false; };
static const int PX = VIEW_W + 8;

static const SDL_Color CLAN_COL[8]={
    {255,80,80,255},{80,210,80,255},{80,130,255,255},{255,200,40,255},
    {255,110,200,255},{40,225,200,255},{220,140,40,255},{170,80,255,255}
};

static void draw_btn(const Btn& b, bool active, bool hover){
    SDL_Color bg = active  ? SDL_Color{60,90,140,255} :
                   hover   ? SDL_Color{45,45,60,255}  :
                             SDL_Color{25,25,35,255};
    frect(b.r.x,b.r.y,b.r.w,b.r.h,bg);
    SDL_SetRenderDrawColor(g_rnd, active?120:70, active?160:70, active?220:90, 255);
    SDL_RenderDrawRect(g_rnd,&b.r);
    txt(b.r.x+4, b.r.y+(b.r.h-12)/2, b.label);
}

// ─── рендер UI панели ──────────────────────────────────────────
static const char* RM_NAME[]={"[1] Клетки","[2] Органика","[3] Заряд","[4] Токсин","[5] Свет","[6] Кланы"};

static void render_panel(uint64_t tick, int mx, int my){
    // фон
    frect(VIEW_W,0,PANEL_W,VIEW_H,{16,16,22,255});
    SDL_SetRenderDrawColor(g_rnd,50,50,70,255);
    SDL_RenderDrawLine(g_rnd,VIEW_W,0,VIEW_W,VIEW_H);

    int cx=VIEW_W+8, cy=8;
    char buf[128];

    // кнопки управления
    Btn btns_ctrl[]={
        {{PX,cy,130,28},"|| Пауза"},
        {{PX+136,cy,70,28},"+ Быстро"},
        {{PX+212,cy,70,28},"- Медл."},
    };
    for(auto& b:btns_ctrl){
        SDL_Point p{mx,my};
        bool hov=SDL_PointInRect(&p,&b.r);
        bool act=(&b==&btns_ctrl[0]) && paused;
        draw_btn(b,act,hov);
    }
    cy+=34;

    Btn btns2[]={
        {{PX,cy,136,24},"↺ Рестарт"},
        {{PX+144,cy,136,24},"⌂ Камера"},
    };
    for(auto& b:btns2){
        SDL_Point p{mx,my};
        draw_btn(b,false,SDL_PointInRect(&p,&b.r));
    }
    cy+=30;

    snprintf(buf,sizeof(buf),"delay:%dмс  %dт/кадр  zoom:%.2f",tick_delay,ticks_per_frame,zoom);
    txt(cx,cy,buf,{130,130,150,255}); cy+=16;

    // виды
    for(int i=0;i<6;i++){
        Btn b{{PX,cy,PANEL_W-16,20},RM_NAME[i]};
        SDL_Point p{mx,my};
        draw_btn(b,rmode==i,SDL_PointInRect(&p,&b.r));
        cy+=21;
    }
    cy+=6;

    // разделитель
    SDL_SetRenderDrawColor(g_rnd,50,50,70,255);
    SDL_RenderDrawLine(g_rnd,cx,cy,VIEW_W+PANEL_W-8,cy); cy+=8;

    // свет
    static const float PH[]={4.f,2.f,3.f,2.f,3.f,2.f};
    int ph=(tick/300)%6;
    snprintf(buf,sizeof(buf),"Свет: %.0f  фаза %d/6  тик:%llu",PH[ph],ph+1,(unsigned long long)tick);
    txt(cx,cy,buf,{220,200,100,255}); cy+=16;

    // статистика
    snprintf(buf,sizeof(buf),"Клеток:   %d (макс %d)",cur_stats.cells,max_stats.cells);
    txt(cx,cy,buf); cy+=14;
    snprintf(buf,sizeof(buf),"Отростки: %d (макс %d)",cur_stats.sprouts,max_stats.sprouts);
    txt(cx,cy,buf); cy+=14;
    snprintf(buf,sizeof(buf),"Листья:   %d",cur_stats.leaves);
    txt(cx,cy,buf); cy+=14;
    snprintf(buf,sizeof(buf),"Рожд: %d  Смерт: %d",cur_stats.births,cur_stats.deaths);
    txt(cx,cy,buf,{150,200,150,255}); cy+=14;

    // гистограмма кланов
    cy+=4;
    txt(cx,cy,"Кланы:",{140,140,160,255}); cy+=13;
    int max_clan=1;
    for(int i=0;i<NUM_CLANS;i++) if(cur_stats.clan_count[i]>max_clan) max_clan=cur_stats.clan_count[i];
    int bar_w=PANEL_W-16;
    for(int i=0;i<NUM_CLANS;i++){
        if(!cur_stats.clan_count[i]) continue;
        int bw=(int)((float)cur_stats.clan_count[i]/max_clan*(bar_w-40));
        frect(cx,cy,bw,10,CLAN_COL[i]);
        snprintf(buf,sizeof(buf),"%d",cur_stats.clan_count[i]);
        txt(cx+bw+2,cy-1,buf,{160,160,160,255});
        cy+=12;
    }

    // легенда
    cy+=6;
    SDL_SetRenderDrawColor(g_rnd,50,50,70,255);
    SDL_RenderDrawLine(g_rnd,cx,cy,VIEW_W+PANEL_W-8,cy); cy+=6;
    struct{SDL_Color c;const char* n;} leg[]={
        {{255,80,80,255},"Отросток"}, {{30,160,40,255},"Лист"},
        {{160,100,40,255},"Корень"},  {{60,120,200,255},"Антенна"},
        {{180,60,200,255},"Детокс"},  {{90,55,25,255},"Древесина"},
        {{230,210,80,255},"Семечка"},
    };
    for(auto& l:leg){
        frect(cx,cy+2,10,10,l.c);
        txt(cx+14,cy,l.n,{150,150,165,255}); cy+=14;
    }

    // управление
    cy+=6;
    SDL_SetRenderDrawColor(g_rnd,50,50,70,255);
    SDL_RenderDrawLine(g_rnd,cx,cy,VIEW_W+PANEL_W-8,cy); cy+=6;
    const char* keys[]={
        "Пробел  пауза","R       рестарт",
        "СКМ     двигать камеру","Колесо  зум",
        "+/-     скорость","1-6     вид","Esc     выход"
    };
    for(auto k:keys){ txt(cx,cy,k,{100,100,120,255}); cy+=13; }
}

// ─── обработка кнопок ──────────────────────────────────────────
static bool btn_hit(int mx,int my,int bx,int by,int bw,int bh){
    return mx>=bx&&mx<bx+bw&&my>=by&&my<by+bh;
}

static void handle_panel_click(int mx,int my){
    int cy=8;
    // пауза
    if(btn_hit(mx,my,PX,cy,130,28)) paused=!paused;
    // быстро
    if(btn_hit(mx,my,PX+136,cy,70,28)){ ticks_per_frame=std::min(ticks_per_frame+5,200); tick_delay=0; }
    // медленно
    if(btn_hit(mx,my,PX+212,cy,70,28)){ ticks_per_frame=std::max(ticks_per_frame-5,1); }
    cy+=34;
    // рестарт
    if(btn_hit(mx,my,PX,cy,136,24)){
        sim_init((unsigned long long)time(nullptr));
        memset(&max_stats,0,sizeof(max_stats));
    }
    // камера
    if(btn_hit(mx,my,PX+144,cy,136,24)) reset_camera();
    cy+=30; cy+=16;
    // виды
    for(int i=0;i<6;i++){
        if(btn_hit(mx,my,PX,cy,PANEL_W-16,20)) rmode=i;
        cy+=21;
    }
}

// ─── main ──────────────────────────────────────────────────────
int main(){
    srand((unsigned)time(nullptr));
    SDL_Init(SDL_INIT_VIDEO); TTF_Init();

    SDL_Window* win=SDL_CreateWindow("EvoSim CUDA",
        SDL_WINDOWPOS_CENTERED,SDL_WINDOWPOS_CENTERED,WIN_W,WIN_H,0);
    g_rnd=SDL_CreateRenderer(win,-1,SDL_RENDERER_ACCELERATED);
    load_font();

    // стартовая камера — показать ровно 1 тайл мира
    reset_camera();

    // GPU пиксельный буфер
    cudaMalloc(&d_pixels, VIEW_W*VIEW_H*sizeof(uint32_t));

    // SDL текстура — формат ARGB8888 совпадает с 0xAARRGGBB в коде
    world_tex=SDL_CreateTexture(g_rnd,SDL_PIXELFORMAT_ARGB8888,
                                SDL_TEXTUREACCESS_STREAMING,VIEW_W,VIEW_H);

    // инициализируем симуляцию
    sim_init((unsigned long long)time(nullptr));

    bool running=true;
    SDL_Event ev;
    int mx=0,my=0;
    static uint32_t* h_pixels=(uint32_t*)malloc(VIEW_W*VIEW_H*4);

    while(running){
        while(SDL_PollEvent(&ev)){
            if(ev.type==SDL_QUIT) running=false;

            if(ev.type==SDL_MOUSEMOTION){
                mx=ev.motion.x; my=ev.motion.y;
                if(dragging&&mx<VIEW_W){
                    cam_x=drag_cx-(mx-drag_sx)/zoom;
                    cam_y=drag_cy-(my-drag_sy)/zoom;
                }
            }
            if(ev.type==SDL_MOUSEWHEEL&&mx<VIEW_W){
                float oz=zoom;
                if(ev.wheel.y>0) zoom=std::min(zoom*1.2f,64.f);
                else             zoom=std::max(zoom/1.2f,0.1f);
                cam_x+=mx/oz - mx/zoom;
                cam_y+=my/oz - my/zoom;
            }
            if(ev.type==SDL_MOUSEBUTTONDOWN){
                if(ev.button.button==SDL_BUTTON_MIDDLE&&ev.button.x<VIEW_W){
                    dragging=true; drag_sx=ev.button.x; drag_sy=ev.button.y;
                    drag_cx=cam_x; drag_cy=cam_y;
                }
                if(ev.button.button==SDL_BUTTON_LEFT){
                    mx=ev.button.x; my=ev.button.y;
                    if(mx>=VIEW_W) handle_panel_click(mx,my);
                }
            }
            if(ev.type==SDL_MOUSEBUTTONUP&&ev.button.button==SDL_BUTTON_MIDDLE)
                dragging=false;

            if(ev.type==SDL_KEYDOWN) switch(ev.key.keysym.sym){
                case SDLK_ESCAPE: running=false; break;
                case SDLK_SPACE:  paused=!paused; break;
                case SDLK_r:
                    sim_init((unsigned long long)time(nullptr));
                    memset(&max_stats,0,sizeof(max_stats));
                    break;
                case SDLK_1: rmode=0; break; case SDLK_2: rmode=1; break;
                case SDLK_3: rmode=2; break; case SDLK_4: rmode=3; break;
                case SDLK_5: rmode=4; break; case SDLK_6: rmode=5; break;
                case SDLK_PLUS: case SDLK_EQUALS: case SDLK_KP_PLUS:
                    ticks_per_frame=std::min(ticks_per_frame+5,200); tick_delay=0; break;
                case SDLK_MINUS: case SDLK_KP_MINUS:
                    ticks_per_frame=std::max(ticks_per_frame-5,1);
                    if(ticks_per_frame<=1) tick_delay=50;
                    break;
            }
        }

        // симуляция
        if(!paused){
            for(int i=0;i<ticks_per_frame;i++) sim_tick();
        }

        uint64_t tick=sim_get_tick();

        // статистика раз в 60 тиков
        if(tick>200 && tick-last_stats_tick>=60){
            sim_get_stats(&cur_stats);
            if(cur_stats.cells  >max_stats.cells)   max_stats.cells  =cur_stats.cells;
            if(cur_stats.sprouts>max_stats.sprouts)  max_stats.sprouts=cur_stats.sprouts;
            last_stats_tick=tick;
        }

        // рендер GPU -> текстура
        sim_render(d_pixels,VIEW_W,VIEW_H,cam_x,cam_y,zoom,rmode);
        cudaMemcpy(h_pixels,d_pixels,VIEW_W*VIEW_H*4,cudaMemcpyDeviceToHost);
        SDL_UpdateTexture(world_tex,nullptr,h_pixels,VIEW_W*4);

        SDL_SetRenderDrawColor(g_rnd,0,0,0,255);
        SDL_RenderClear(g_rnd);
        SDL_Rect dst{0,0,VIEW_W,VIEW_H};
        SDL_RenderCopy(g_rnd,world_tex,nullptr,&dst);
        render_panel(tick,mx,my);
        SDL_RenderPresent(g_rnd);

        if(tick_delay>0) SDL_Delay(tick_delay);
    }

    free(h_pixels);
    cudaFree(d_pixels);
    SDL_DestroyTexture(world_tex);
    if(fnt) TTF_CloseFont(fnt);
    if(fnt_sm) TTF_CloseFont(fnt_sm);
    TTF_Quit();
    SDL_DestroyRenderer(g_rnd);
    SDL_DestroyWindow(win);
    SDL_Quit();
    return 0;
}
