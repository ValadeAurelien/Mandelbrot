#include <iostream>
#include <complex>
#include <cmath>
#include <iomanip>
#include <string>
#include <fstream>


using namespace std;


__global__ 
void z_funct(double * d_mat_re, double * d_mat_im, int *d_img, int nb_ite)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  double c_re = d_mat_re[i];
  double c_im = d_mat_im[i];
  double z_re = 0, z_im =0;
  double z_re_tmp = 0;
  int t=0;

  while ((z_re*z_re+z_im*z_im <= 4) && (t<nb_ite)) 
  {
    z_re_tmp = z_re*z_re - z_im*z_im + c_re;
    z_im = 2*z_im*z_re + c_im;
    z_re = z_re_tmp;
    t++;
  }
  d_img[i] = t;
}


int main(int argc, char * argv[])
{
  
  if (argc == 1)
  {
    cout << "Help" << endl 
      << " 8 args :" << endl << endl
      << "arg1 = min real part  |  arg2 = min imaginary part " << endl
      << "arg3 = max real part  |  arg4 = max imaginary part " << endl
      << "arg5 = number of points on the real axe  |  arg6 = number of points on the imaginary axe " << endl
      << "arg7 = nb of iterations  |  arg8 = limit convergence" 
      << endl << endl 
      << " 4 args :" << endl << endl
      << "arg1 = number of points on the real axe  |  arg2 = number of points on the imaginary axe " << endl
      << "arg3 = nb of iterations " 
      << endl << endl ;
    return 1;
  }

  double max_re, max_im, min_re, min_im;
  int nb_pts_re, nb_pts_im, nb_ite;

  if (argc == 8)
  {
    try 
    {
      min_re = stod(argv[1]);
      min_im = stod(argv[2]);
      max_re = stod(argv[3]);
      max_im = stod(argv[4]);
      nb_pts_re = stoi(argv[5]);
      nb_pts_im = stoi(argv[6]);
      nb_ite = stoi(argv[7]);
    }
    catch (...)
    { 
      cout << "Bad Args : see help (type nameofprogram without args)" << endl << endl;
      return 1;
    }
  }
  if (argc == 4 ) 
  {
    min_re = -2;
    min_im = -1;
    max_re = 1;
    max_im = 1;

    try 
    {
      nb_pts_re = stoi(argv[1]);
      nb_pts_im = stoi(argv[2]);
      nb_ite = stoi(argv[3]);
    }
    catch (...)
    { 
      cout << "Bad Args : see help (type nameofprogram without args)" << endl << endl;
      return 1;
    }
  }

  cout << "Initializing..." << endl;

  int size_d = sizeof(double)*nb_pts_re*nb_pts_im;
  int size_i = sizeof(int)*nb_pts_re*nb_pts_im;
  
  double * mat_re = (double *)malloc(size_d);
  double * mat_im = (double *)malloc(size_d);

  double re, im;

  for (int i=0; i<nb_pts_im; i++)
  {
    im = max_im - (max_im-min_im)/nb_pts_im*i;

    for (int j=0; j<nb_pts_re; j++)
    {
      re = max_re - (max_re-min_re)/nb_pts_re*j;
      mat_re[i*nb_pts_re+j] = re;
      mat_im[i*nb_pts_re+j] = im;
    }
  }

  double *d_mat_re, *d_mat_im;
  int *d_img;
  cudaMalloc(&d_mat_re,size_d);
  cudaMalloc(&d_mat_im,size_d);
  cudaMalloc(&d_img,size_i);
  
  cudaMemcpy(d_mat_re, mat_re, size_d, cudaMemcpyHostToDevice);
  cudaMemcpy(d_mat_im, mat_im, size_d, cudaMemcpyHostToDevice);

  cout << "Running on GPU..." << endl;
  dim3 blockDim = 1024;
  dim3 gridDim = (nb_pts_re*nb_pts_im)/1024 + 1;
  z_funct<<<gridDim,blockDim>>>(d_mat_re, d_mat_im, d_img, nb_ite);
  
  cout << "Fetching datas..." << endl; 
  int * img = (int*) malloc(size_i);

  cudaMemcpy(mat_re, d_mat_re, size_d, cudaMemcpyDeviceToHost);
  cudaMemcpy(mat_im, d_mat_im, size_d, cudaMemcpyDeviceToHost);
  cudaMemcpy(img, d_img, size_i, cudaMemcpyDeviceToHost);


  cout << "Writing on the disk..." << endl;
  ofstream file; file.open("conv");
  for (int i=0; i<nb_pts_im; i++)
  {
    for (int j=0; j<nb_pts_re; j++)
    {
      file << setw(15) << img[i*nb_pts_re+j] ;
    }
    file << endl ;
  }

  file.close();
/*
  file.open("val");
  for (int i=0; i<nb_pts_im; i++)
  {
    for (int j=0; j<nb_pts_re; j++)
    {
      file << setw(15) << pow(mat_re[i*nb_pts_re+j],2) + pow(mat_im[i*nb_pts_re+j],2) ;
    }
    file << endl ;
  }

  file.close();
*/
  cout << "Done !" << endl;
  return 0;
}

