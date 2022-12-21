#include <iostream>
#include <Eigen/Dense>
#include <omp.h>
//  git clone https://gitlab.com/libeigen/eigen.git
//  g++ -g -I ../eigen/ spectral_embedding.cpp -fopenmp
// https://eigen.tuxfamily.org/dox/TopicMultiThreading.html
using namespace Eigen;



Eigen::MatrixXd deterministicVectorSignFlip(const Eigen::MatrixXd& u) {
  // Initialize a matrix of the same size as u to store the result
  Eigen::MatrixXd uFlipped = u;

  // Loop over the rows of the matrix
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


//   // Compute the degree matrix
//   VectorXd degree = affinity.rowwise().sum();
//   MatrixXd degree_matrix = degree.asDiagonal();

//   // Compute the Laplacian matrix
//   MatrixXd laplacian = degree_matrix - affinity;


//  Construct the nearest neighbor graph
  
  MatrixXd A = MatrixXd::Zero(X.rows(), X.rows());
  // #pragma omp parallel for
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

  // std::cout << A << std::endl;

  // // make it symmetric as here: https://github.com/scikit-learn/scikit-learn/blob/dc580a8ef5ee2a8aea80498388690e2213118efd/sklearn/manifold/_spectral_embedding.py#L642
  // // have to assign to a different variable due to aliasing issue
  MatrixXd B = (A + A.transpose())/2;

  std::cout <<"B: " << B << std::endl;

  VectorXd degrees = B.rowwise().sum().array() - 1; // -1 to remove the diagonal from sum

	// calc normalised laplacian 
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

  for (int i = 0; i < B.rows(); i++){
    L(i, i) = -1.0;
  }

  std::cout <<"L: " << L << std::endl;

  // To continue the computation of the graph Laplacian matrix,
  //  you can subtract the sum of the weights of the edges in the graph from the 
  // diagonal matrix of vertex degrees. Here is the code to do that:  
  // VectorXd degrees = B.rowwise().sum();
  // MatrixXd L = MatrixXd::Zero(B.rows(), B.rows());
  // // #pragma omp parallel for
  // for (int i = 0; i < B.rows(); i++){
  //   // #pragma omp parallel for
  //   for (int j = 0; j < B.rows(); j++){
  //     L(i, j) -= B(i, j);
  //   }
  //   L(i, i) = degrees(i);
  // }


  // Compute the eigenvectors of the Laplacian matrix using Eigen's SelfAdjointEigenSolver
  SelfAdjointEigenSolver<MatrixXd> eigensolver(L);
  if (eigensolver.info() != Success)
  {
    std::cerr << "Failed to compute eigenvectors." << std::endl;
    return 1;
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
  std::cout << V.leftCols(k) << std::endl;

  auto embedding = V.rightCols(k);//.transpose();

  // std::cout << embedding << std::endl;
  auto embedding_normed = embedding.array().colwise() * degrees.array();

  std::cout << embedding_normed << std::endl;

  // _deterministic_vector_sign_flip
  Eigen::MatrixXd uFlipped = deterministicVectorSignFlip(embedding_normed.transpose());

  std::cout << uFlipped.transpose() << std::endl;
  // Project the data points onto the selected eigenvectors to obtain the low-dimensional embedding
  // MatrixXd Y = V*X ;

  // // Print
  // std::cout << "Y = " << Y << std::endl;

    // https://github.com/pthimon/clustering/blob/master/lib/SpectralClustering.cpp
  // Eigen::MatrixXd Deg = Eigen::MatrixXd::Zero(A.rows(), A.cols());

	// // calc normalised laplacian 
	// for ( int i=0; i < A.cols(); i++) {
	// 	Deg(i,i)=1/(sqrt((A.row(i).sum())) );
	// }

  // std::cout << Deg << std::endl;

// std::cout << L << std::endl;


	// Eigen::MatrixXd Lapla = Deg * A * Deg;

  // std::cout << Lapla << std::endl;

  // MatrixXd B = (Lapla + Lapla.transpose())/2;

  // std::cout << B << std::endl;

	// Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> s(B);
	// Eigen::VectorXd val = s.eigenvalues();
	// Eigen::MatrixXd vec = s.eigenvectors();

  // std::cout << val << std::endl;

	// //sort eigenvalues/vectors
	// int n = X.cols();
	// for (int i = 0; i < n - 1; ++i) {
	// 	int k;
	// 	val.segment(i, n - i).maxCoeff(&k);
	// 	if (k > 0) {
	// 		std::swap(val[i], val[k + i]);
	// 		vec.col(i).swap(vec.col(k + i));
	// 	}
	// }

	// //choose the number of eigenvectors to consider
  // Eigen::MatrixXd mEigenVectors;
	// if (k < vec.cols()) {
	// 	mEigenVectors = vec.block(0,0,vec.rows(),k);
	// } else {
	// 	mEigenVectors = vec;
	// }

  //  std::cout << "Y = " << mEigenVectors << std::endl;
}