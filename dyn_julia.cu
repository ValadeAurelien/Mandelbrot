#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>
#include <SDL/SDL.h>
#include <unistd.h>

typedef struct 
{
  double min_re, min_im, max_re, max_im;
  int nb_pts_re, nb_pts_im;
} cplx_plan_struct;

enum order_enum {UP, DOWN, RIGHT, LEFT, PLUS, MINUS};

__global__ 
void z_funct(Uint32 *d_img, cplx_plan_struct * d_cplx_plan, double c_re, double c_im, int nb_ite)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int x = i % d_cplx_plan->nb_pts_re;
  int y = i / d_cplx_plan->nb_pts_re;

  double z_re = d_cplx_plan->min_re + 
    (d_cplx_plan->max_re-d_cplx_plan->min_re)
    /d_cplx_plan->nb_pts_re*x;

  double z_im = d_cplx_plan->max_im - 
    (d_cplx_plan->max_im-d_cplx_plan->min_im)
    /d_cplx_plan->nb_pts_im*y;

  double z_re_tmp = 0;
  int t=0;

  while ((z_re*z_re+z_im*z_im <= 4) && (t<nb_ite)) 
  {
    z_re_tmp = z_re*z_re - z_im*z_im + c_re;
    z_im = 2*z_im*z_re + c_im;
    z_re = z_re_tmp;
    t++;
  }
  int r,g,b;
  if (t == nb_ite) 
  {
    r=0; g=0; b=0;
  }
  else
  {
    if ( t < nb_ite/2) r = 0;
    else r = 2*floor(256./nb_ite*t-128);
    g = floor(256./nb_ite*(nb_ite-2*abs(nb_ite/2.-t)));
//    if ( t < nb_ite/2) b = 256-2*floor(256./nb_ite*t);
    if ( t < nb_ite/2) b = 256-2*floor(128-256./nb_ite*t);
    else b = 0;
  }
  d_img[i] = r*65536+g*256+b; 
}

void mv_cplx_plan(cplx_plan_struct * cplx_plan, enum order_enum order)
{
  double size_re, size_im;
  switch (order)
  {
    case UP :
      size_re = (cplx_plan->max_im - cplx_plan->min_im)/3;
      cplx_plan->min_im += size_re;
      cplx_plan->max_im += size_re;
      break;
    case DOWN :
      size_re = (cplx_plan->max_im - cplx_plan->min_im)/3;
      cplx_plan->min_im -= size_re;
      cplx_plan->max_im -= size_re;
      break;
    case LEFT :
      size_im = (cplx_plan->max_re - cplx_plan->min_re)/3;
      cplx_plan->min_re -= size_im;
      cplx_plan->max_re -= size_im;
      break;
    case RIGHT :
      size_im = (cplx_plan->max_re - cplx_plan->min_re)/3;
      cplx_plan->min_re += size_im;
      cplx_plan->max_re += size_im;
      break;
    case MINUS :
      size_re = (cplx_plan->max_re - cplx_plan->min_re)/4;
      size_im = (cplx_plan->max_im - cplx_plan->min_im)/4;
      cplx_plan->min_re -= size_re;
      cplx_plan->max_re += size_re;
      cplx_plan->min_im -= size_im;
      cplx_plan->max_im += size_im;
      break;
    case PLUS :
      size_re = (cplx_plan->max_re - cplx_plan->min_re)/4;
      size_im = (cplx_plan->max_im - cplx_plan->min_im)/4;
      cplx_plan->min_re += size_re;
      cplx_plan->max_re -= size_re;
      cplx_plan->min_im += size_im;
      cplx_plan->max_im -= size_im;
      break;
  }
}
      

int main(int argc, char * argv[])
{
  
  if (argc == 1)
  {
/*    printf( "Help \n 
            8 args : \n \n
            arg1 = min real part  |  arg2 = min imaginary part  \n
            arg3 = max real part  |  arg4 = max imaginary part  \n
            arg5 = number of points on the real axe  |  arg6 = number of points on the imaginary axe  \n
            arg7 = nb of iterations  |  arg8 = limit convergence 
            \n \n 
            4 args : \n \n
             arg1 = number of points on the real axe  |  arg2 = number of points on the imaginary axe  \n
            arg3 = nb of iterations  
            \n \n") ;*/
    return 1;
  }

  double max_re, max_im, min_re, min_im, c_re, c_im;
  int nb_pts_re, nb_pts_im,nb_ite;

  if (argc == 10)
  {
    try 
    {
      min_re = atof(argv[1]);
      min_im = atof(argv[2]);
      max_re = atof(argv[3]);
      max_im = atof(argv[4]);
      nb_pts_re = atoi(argv[5]);
      nb_pts_im = atoi(argv[6]);
      c_re = atof(argv[7]);
      c_im = atof(argv[8]);
      nb_ite = atoi(argv[9]);
    }
    catch (...)
    { 
      printf( "Bad Args : see help (type nameofprogram without args)\n\n");
      return 1;
    }
  }
  if (argc == 6 ) 
  {
    min_re = -2;
    min_im = -1;
    max_re = 1;
    max_im = 1;

    try 
    {
      nb_pts_re = atoi(argv[1]);
      nb_pts_im = atoi(argv[2]);
      c_re = atof(argv[3]);
      c_im = atof(argv[4]);
      nb_ite = atoi(argv[5]);
    }
    catch (...)
    { 
      printf( "Bad Args : see help (type nameofprogram without args)\n\n");
      return 1;
    }
  }


  int size_i = sizeof(Uint32)*nb_pts_re*nb_pts_im;
  Uint32 *d_img;
  cudaMalloc(&d_img,size_i);
  cplx_plan_struct cplx_plan, *d_cplx_plan;
  cudaMalloc(&d_cplx_plan, sizeof(cplx_plan_struct));

  dim3 blockDim = 1024;
  dim3 gridDim = (nb_pts_re*nb_pts_im)/1024 + 1;

  cplx_plan.min_re = min_re;
  cplx_plan.min_im = min_im;
  cplx_plan.max_re = max_re;
  cplx_plan.max_im = max_im;
  cplx_plan.nb_pts_re = nb_pts_re;
  cplx_plan.nb_pts_im = nb_pts_im;

  SDL_Init(SDL_INIT_VIDEO);
  SDL_Surface *SDL_img = SDL_SetVideoMode(nb_pts_im, nb_pts_re, 32, SDL_HWSURFACE | SDL_DOUBLEBUF);
  SDL_Event event;

  bool quit = false;
  bool recalc = true;
  bool refresh = true;

  int mouse_x, mouse_y;
  double x, y;

  while ( !quit )
  {
    while( SDL_PollEvent( &event ) )
    {
      switch( event.type )
      {
        case SDL_KEYDOWN:
          switch ( event.key.keysym.sym )
          {
            case SDLK_UP:
              mv_cplx_plan(&cplx_plan, UP);
              recalc = true;
              refresh = true;
              break;
            case SDLK_DOWN:
              mv_cplx_plan(&cplx_plan, DOWN);
              recalc = true;
              refresh = true;
              break;
            case SDLK_LEFT:
              mv_cplx_plan(&cplx_plan, LEFT);
              recalc = true;
              refresh = true;
              break;
            case SDLK_RIGHT:
              mv_cplx_plan(&cplx_plan, RIGHT);
              recalc = true;
              refresh = true;
              break;
            case SDLK_KP_PLUS:
              mv_cplx_plan(&cplx_plan, PLUS);
              recalc = true;
              refresh = true;
              break;
            case SDLK_KP_MINUS:
              mv_cplx_plan(&cplx_plan, MINUS);
              recalc = true;
              refresh = true;
              break;
            case SDLK_KP_MULTIPLY:
              nb_ite *= 2;
              recalc = true;
              refresh = true;
              break;
            case SDLK_KP_DIVIDE:
              nb_ite /= 2;
              recalc = true;
              refresh = true;
              break;
            case SDLK_q:
              quit = true;
              recalc = false;
              refresh = true;
              break;
            case SDLK_i:
              cplx_plan.min_re = -2;
              cplx_plan.min_im = -1;
              cplx_plan.max_re = 1;
              cplx_plan.max_im = 1;
              recalc = true;
              refresh = true;
              break;
            case SDLK_s :
              SDL_SaveBMP(SDL_img, "data_julia.bmp");
              break;
            default:
              break;
          }
          break;
        case SDL_MOUSEMOTION :
          SDL_GetMouseState(&mouse_x, &mouse_y);
          refresh = true;
          break;
        default:
          break;
      }
    }
    if (refresh)
    {
      if (recalc)
      {
        recalc = false;
        cudaMemcpy(d_cplx_plan, &cplx_plan, sizeof(cplx_plan_struct), cudaMemcpyHostToDevice);
        z_funct<<<gridDim,blockDim>>>(d_img, d_cplx_plan, c_re, c_im, nb_ite);
        cudaMemcpy(SDL_img->pixels, d_img, size_i, cudaMemcpyDeviceToHost);
      }
      refresh = false;
      x = cplx_plan.min_re + mouse_x*1./nb_pts_re*(cplx_plan.max_re-cplx_plan.min_re); 
      y = cplx_plan.min_im + mouse_y*1./nb_pts_im*(cplx_plan.max_im-cplx_plan.min_im); 
      printf("\rmouse position (re : im)  %.20lf : %.20lf  |  nb_ite %d      ", x, y, nb_ite);
      SDL_Flip(SDL_img);
    }
  }

  SDL_Quit();

  printf("\n\n");
}
