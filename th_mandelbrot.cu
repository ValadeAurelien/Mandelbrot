#include "thrust/device_vector.h"
#include "thrust/host_vector.h"
#include "thrust/tuple.h"
#include "thrust/complex.h"
#include <iostream>
#include <iomanip>
#include <string>
#include <fstream>

using namespace std;
typedef thrust::complex<float> th_complex;
typedef thrust::device_vector<th_complex> th_dev_cplx_vec;
typedef thrust::device_vector<int> th_dev_bool_vec;



struct z_functor
{
  int nb_ite;
  z_functor(int _nb_ite) : nb_ite(_nb_ite) {}

  __host__ __device__
    int operator () (th_complex c)
    {
      th_complex z_val = 0;
      int t=0;
      while ((abs(z_val) < 2) && (t<nb_ite)) 
      {
        z_val = z_val*z_val + c;
        t++;
      }
      return t;
    }
};



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

  float max_re, max_im, min_re, min_im;
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

  th_dev_cplx_vec mat (nb_pts_re*nb_pts_im) ;

  float re, im;

  for (int i=0; i<nb_pts_im; i++)
  {
    im = max_im - (max_im-min_im)/nb_pts_im*i;

    for (int j=0; j<nb_pts_re; j++)
    {
      re = max_re - (max_re-min_re)/nb_pts_re*j;
      th_complex cplx (re,im);
      mat[i*nb_pts_re+j] = cplx;
    }
  }

  th_dev_bool_vec img (nb_pts_re* nb_pts_im);
  thrust::transform(mat.begin(), mat.end(), img.begin(), z_functor(nb_ite));
  

  ofstream file; file.open("data");
  for (int i=0; i<nb_pts_im; i++)
  {
    for (int j=0; j<nb_pts_re; j++)
    {
      file << setw(15) << img[i*nb_pts_re+j] ;
    }
    file << endl ;
  }

  file.close();

  return 0;
}

