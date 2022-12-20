#include <iostream>
#include <Eigen/Dense>
#include <omp.h>
//  git clone https://gitlab.com/libeigen/eigen.git
//  g++ -g -I ../eigen/ spectral_embedding.cpp -fopenmp
// https://eigen.tuxfamily.org/dox/TopicMultiThreading.html
using namespace Eigen;

int main() {

  int num_threads = Eigen::nbThreads();
  // k is the number of nearest neighbors to connect to each point
  int k = 2;
  // Create a matrix of data points
  MatrixXd X(5, 3);
  X << 1, 2, 3,
          4, 5, 6,
          7, 8, 9,
          10, 11, 12,
          13, 14, 15;
  // MatrixXd X = MatrixXd::Random(100, 10);
  // Compute the affinity matrix using a Gaussian kernel
  // MatrixXd affinity = MatrixXd::Zero(5, 5);
  // double sigma = 1.0;
  // for (int i = 0; i < 5; i++) {
  //   for (int j = i + 1; j < 5; j++) {
  //     double distance = (data.row(i) - data.row(j)).norm();
  //     affinity(i, j) = exp(-distance * distance / (2 * sigma * sigma));
  //     affinity(j, i) = affinity(i, j);
  //   }
  // }


  // // Compute the degree matrix
  // VectorXd degree = affinity.rowwise().sum();
  // MatrixXd degree_matrix = degree.asDiagonal();

  // // Compute the Laplacian matrix
  // MatrixXd laplacian = degree_matrix - affinity;


 // Construct the nearest neighbor graph
  
  MatrixXd A = MatrixXd::Zero(X.rows(), X.rows());
  #pragma omp parallel for
  for (int i = 0; i < X.rows(); i++)
  {
    // Find the k-nearest neighbors of point i using a nearest neighbor search algorithm
    VectorXi indices = VectorXi::LinSpaced(X.rows(), 0, X.rows() - 1);

    //TODO: change to parallel sort
    std::sort(indices.data(), indices.data() + indices.size(), 
        [&](int i1, int i2) { return (X.row(i) - X.row(i1)).squaredNorm() < (X.row(i) - X.row(i2)).squaredNorm(); });
    indices.conservativeResize(k);

    // Connect point i to its k-nearest neighbors
    for (int j : indices)
    {
      A(i, j) = 1;
    }
  }

  // make it symmetric as here: https://github.com/scikit-learn/scikit-learn/blob/dc580a8ef5ee2a8aea80498388690e2213118efd/sklearn/manifold/_spectral_embedding.py#L642
  X = (X + X.transpose())/2;


  // To continue the computation of the graph Laplacian matrix,
  //  you can subtract the sum of the weights of the edges in the graph from the 
  // diagonal matrix of vertex degrees. Here is the code to do that:
  VectorXd degrees = A.rowwise().sum();
  MatrixXd L = MatrixXd::Zero(X.rows(), X.rows());
  #pragma omp parallel for
  for (int i = 0; i < X.rows(); i++){
    #pragma omp parallel for
    for (int j = 0; j < X.rows(); j++){
      L(i, j) -= X(i, j);
    }
    L(i, i) = degrees(i);
  }

  // Compute the eigenvectors of the Laplacian matrix using Eigen's SelfAdjointEigenSolver
  SelfAdjointEigenSolver<MatrixXd> eigensolver(L);
  if (eigensolver.info() != Success)
  {
    std::cerr << "Failed to compute eigenvectors." << std::endl;
    return 1;
  }

  // Select the eigenvectors that correspond to the k smallest eigenvalues as the dimensions of the low-dimensional embedding
  MatrixXd V = eigensolver.eigenvectors().rightCols(k);

  // Project the data points onto the selected eigenvectors to obtain the low-dimensional embedding
  MatrixXd Y = X * V;

  // Print
  std::cout << "Y = " << Y << std::endl;
}