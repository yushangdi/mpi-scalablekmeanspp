#include <iostream>
#include <Eigen/Dense>
#include <Spectra/SymEigsSolver.h>
#include <omp.h>
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/internal/get_time.h"

//  git clone https://gitlab.com/libeigen/eigen.git
//  g++ -g -I ../eigen/ spectral.cpp -fopenmp
// https://eigen.tuxfamily.org/dox/TopicMultiThreading.html
using namespace Eigen;

Eigen::MatrixXd deterministicVectorSignFlip(const Eigen::MatrixXd& u) {
  // Initialize a matrix of the same size as u to store the result
  Eigen::MatrixXd uFlipped = u;

  // Loop over the rows of the matrix
  #pragma omp parallel for
  for (int i = 0; i < u.rows(); i++) {
    // Find the index of the element with the maximum absolute value in the i-th row
    // int maxAbsIdx = (u.row(i).array().abs().maxCoeff() == u.row(i).array()) ? u.row(i).array().abs().maxCoeff() : u.row(i).array().abs().maxCoeff();
     int maxAbsVal = u.row(i).array().abs().maxCoeff();

    // Flip the sign of the elements in the i-th row depending on the sign of the element with the maximum absolute value
    if (maxAbsVal < 0) {
      uFlipped.row(i) *= -1;
    }
  }

  return uFlipped;
}

Eigen::MatrixXd getEigenVectors(MatrixXd &L, int n_components){
  // Compute the eigenvectors of the Laplacian matrix using Eigen's SelfAdjointEigenSolver
  SelfAdjointEigenSolver<MatrixXd> eigensolver(L);
  if (eigensolver.info() != Success)
  {
    std::cerr << "Failed to compute eigenvectors." << std::endl;
    exit(1);
  }

  // Select the eigenvectors that correspond to the k smallest eigenvalues as the dimensions of the low-dimensional embedding
  MatrixXd V = eigensolver.eigenvectors();
  VectorXd U = eigensolver.eigenvalues();

  //sort eigenvalues/vectors
	int n = L.cols();
	for (int i = 0; i < n - 1; ++i) {
		int kk;
		U.segment(i, n - i).maxCoeff(&kk);
		if (kk > 0) {
			std::swap(U[i], U[kk + i]);
			V.col(i).swap(V.col(kk + i));
		}
	}
  auto embedding = V.rightCols(n_components);
  return embedding;
}


Eigen::MatrixXd SpectralEmbedding(MatrixXd &X, int k, int n_components) {

  // int num_threads = Eigen::nbThreads();
  std::cout << "embedding start" << std::endl;
  parlay::internal::timer t; t.start();
  //  Construct the nearest neighbor graph
  MatrixXd A = MatrixXd::Zero(X.rows(), X.rows());
  #pragma omp parallel for
  for (int i = 0; i < X.rows(); i++)
  {
    // Find the k-nearest neighbors of point i using a nearest neighbor search algorithm
    VectorXi indices = VectorXi::LinSpaced(X.rows(), 0, X.rows() - 1);

    //parallel sort
    // auto s = parlay::make_slice(indices.data(), indices.data() + indices.size());
    // parlay::sort_inplace(s, 
    //     [&](int i1, int i2) { return (X.row(i) - X.row(i1)).squaredNorm() < (X.row(i) - X.row(i2)).squaredNorm(); });
    std::sort(indices.data(), indices.data() + indices.size(),
          [&](int i1, int i2) { return (X.row(i) - X.row(i1)).squaredNorm() < (X.row(i) - X.row(i2)).squaredNorm(); });
    indices.conservativeResize(k);

    // Connect point i to its k-nearest neighbors
    #pragma omp parallel for
    for (int j = 0; j < k; j++)
    {
      A(i, indices(j)) = 1;
    }
  }

  t.next("kNN graph built");
  // // make it symmetric as here: https://github.com/scikit-learn/scikit-learn/blob/dc580a8ef5ee2a8aea80498388690e2213118efd/sklearn/manifold/_spectral_embedding.py#L642
  // // have to assign to a different variable due to aliasing issue
  MatrixXd B = (A + A.transpose())/2;

  // std::cout <<"B: " << B << std::endl;

  VectorXd degrees = B.rowwise().sum().array() - 1; // -1 to remove the diagonal from sum

	// calc normalised laplacian 
  #pragma omp parallel for
	for (int i=0; i < B.rows(); i++) {
    double v = sqrt(degrees(i));
    if(v==0){
      degrees(i)=1;
    }else{
      degrees(i)=1/v;
    }
	}

  auto C = degrees * degrees.transpose();

  MatrixXd L = B.cwiseProduct(C);

  #pragma omp parallel for
  for (int i = 0; i < B.rows(); i++){
    L(i, i) = -1.0;
  }

  t.next("laplacian built");

  auto embedding = getEigenVectors(L, n_components);
  t.next("eigens computed");
  auto embedding_normed = embedding.array().colwise() * degrees.array();

  // std::cout << embedding_normed << std::endl;

  // _deterministic_vector_sign_flip
  Eigen::MatrixXd uFlipped = deterministicVectorSignFlip(embedding_normed.transpose());

  Eigen::MatrixXd embedded = uFlipped.transpose();

  t.next("embedding computed");
  return embedded;
}