#include <iostream>
#include <Eigen/Dense>
#include <omp.h>

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


Eigen::MatrixXd SpectralEmbedding(MatrixXd &X, int k) {

  // int num_threads = Eigen::nbThreads();

  //  Construct the nearest neighbor graph
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
    #pragma omp parallel for
    for (int j = 0; j < k; j++)
    {
      A(i, indices(j)) = 1;
    }
  }

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
	int n = X.cols();
	for (int i = 0; i < n - 1; ++i) {
		int kk;
		U.segment(i, n - i).maxCoeff(&kk);
		if (kk > 0) {
			std::swap(U[i], U[kk + i]);
			V.col(i).swap(V.col(kk + i));
		}
	}
  // std::cout << V.leftCols(k) << std::endl;

  auto embedding = V.rightCols(k);//.transpose();

  // std::cout << embedding << std::endl;
  auto embedding_normed = embedding.array().colwise() * degrees.array();

  // std::cout << embedding_normed << std::endl;

  // _deterministic_vector_sign_flip
  Eigen::MatrixXd uFlipped = deterministicVectorSignFlip(embedding_normed.transpose());

  Eigen::MatrixXd embedded = uFlipped.transpose();
  return std::move(embedded);
}