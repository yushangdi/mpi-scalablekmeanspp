#include <iostream>
#include <fstream>
#include <string>

#include <omp.h>

#include <Eigen/Dense>
#include "spectral.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"


//  git clone https://gitlab.com/libeigen/eigen.git
//  g++ -std=c++17 -mcx16 -I../eigen/ -I /home/ubuntu/tmfg_benchmark/par_tmfg/parlaylib/include  -DPARLAY_OPENMP spectral.cpp  -fopenmp
// OMP_NUM_THREADS=n ./a.out
// https://eigen.tuxfamily.org/dox/TopicMultiThreading.html
using namespace Eigen;

bool _debug;

// read a symmatric matrix from file
parlay::sequence<double> readDataFromFile(char const *fname, int& n, int& d) {
  std::ifstream myFile (fname, std::ios::in | std::ios::binary);
  int sizes[2] = {0,0};
  myFile.read((char*)sizes, sizeof(int) * 2);
  n = sizes[0];
  d = sizes[1];
  if(n<=0 || d <=0){
    std::cout << "bad input dims" << std::endl;
    std::cout << "input dims: " << n << " " << d << " " << std::endl;
    exit(1);
  }
  parlay::sequence<double> W = parlay::sequence<double>(n*d);
  myFile.read((char*)W.data(), sizeof(double) * n*d);
  if (W.size() == 0) {
    std::cout << "readPointsFromFile empty file" << std::endl;
    abort();
  }
  if (W.size() % n != 0) {
    std::cout << "readPointsFromFile wrong file type or wrong dimension" << std::endl;
    abort();
  }
  return W;
}

// ./a.out ./Image_data/test.dat 
int main(int argc, char *argv[]) {
  char* filename = argv[1];
  // k is the number of nearest neighbors to connect to each point
  int k = argv[2];
  int n;
  int d;

  /* read data points from file ------------------------------------------*/
  parlay::sequence<double> W = readDataFromFile(filename, n, d);
  std::cout << "input dims: " << n << " " << d << " " << std::endl;
  if(_debug){
    for(long i=0;i< std::min((long)10, ((long)n)*d);++i){
      std::cout << W[i] << " ";
    }
    std::cout << std::endl;
  }

  Eigen::initParallel();
  int num_threads = Eigen::nbThreads();
  std::cout << "num_threads: " << num_threads << std::endl;

  // Create a matrix of data points
  MatrixXd X(n, d);
  // X << 1, 2, 3;
  parlay::parallel_for(0,((long)n)*d,[&](long i){
    X(i) = W[i];
  });

  Eigen::MatrixXd embedded = SpectralEmbedding(X, k);
  std::cout << embedded << std::endl;
}