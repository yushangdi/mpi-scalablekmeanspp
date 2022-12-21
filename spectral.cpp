#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <cstdlib>

#include <omp.h>

#include <Eigen/Dense>
#include "spectral.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/internal/get_time.h"

//  git clone https://gitlab.com/libeigen/eigen.git
//  g++ -std=c++17 -mcx16 -Ieigen/ -I../par-filtered-graph-clustering/parlaylib/include  -DPARLAY_OPENMP spectral.cpp  -fopenmp
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


void writeDataBinary(char const *filename, Eigen::MatrixXd& X){
  int n = X.rows();
  int d = X.cols();
  std::ofstream fout(filename, std::ios::binary);
  fout.write(reinterpret_cast<char*>(&n), sizeof(int));
  fout.write(reinterpret_cast<char*>(&d), sizeof(int));
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < d; j++) {
    fout.write(reinterpret_cast<char*>(&X(i, j)), sizeof(double));
    }
  }
  fout.close();
}

// ./a.out ./Image_data/test.dat 2 ./Image_data/test_emb.dat 
int main(int argc, char *argv[]) {
  char* filename = argv[1];
  // k is the number of nearest neighbors to connect to each point
  int k = atoi(argv[2]);
  char* out_filename = argv[3];
  int n;
  int d;

  /* read data points from file ------------------------------------------*/
  parlay::internal::timer t; t.start();
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
  parlay::parallel_for(0,n,[&](long i){
    parlay::parallel_for(0,d,[&](long j){
      X(i, j) = W[i*d+j];
    });
  });
  
  // std::cout << "X" << X << std::endl;
  t.next("read");

  Eigen::MatrixXd embedded = SpectralEmbedding(X, k);
  // std::cout << embedded << std::endl;
  t.next("embedding");

  if(out_filename){
    std::cout << "writing output to " << out_filename << std::endl;
    writeDataBinary(out_filename, embedded);
    t.next("output");
  }
}