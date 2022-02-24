#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>

int main(int argc, char **argv) {
  double ar = 1.0, br = 2.0, cr = 1.0; // real param value
  double ag = 2.0, bg = 2.3, cg = 1.0; // guess param value

  int count = 100;
  double w_sigma = 1.0;                        // 噪声Sigma值
  double inv_sigma = 1.0 / w_sigma;
  cv::RNG rng;                                 // OpenCV随机数产生器

  // gen data
  std::vector<double> x_data, y_data;
  for (int i = 0; i < count; i++) {
    double x = i / 100.0;
    double y = exp(ar*x*x + br*x + cr) + rng.gaussian(w_sigma * w_sigma);
    x_data.push_back(x);
    y_data.push_back(y);
  }

  // 开始Gauss-Newton迭代
  int iterations = 100;    // 迭代次数
  double cost = 0, lastCost = 0;  // 本次迭代的cost和上一次迭代的cost
  std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
  for (int iter = 0; iter < iterations; iter++) {
    
    Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
    Eigen::Vector3d b = Eigen::Vector3d::Zero();
    cost = 0;

    for (int i = 0; i < count; i++) {
      double xi = x_data[i];
      double error = y_data[i] - exp(ag*xi*xi + bg*xi + cg);

      Eigen::Vector3d J;
      J[0] = -xi*xi*exp(ag*xi*xi + bg*xi + cg);
      J[1] = -xi*exp(ag*xi*xi + bg*xi + cg);
      J[2] = -exp(ag*xi*xi + bg*xi + cg);

      H += inv_sigma * inv_sigma * J * J.transpose();
      b += -inv_sigma * inv_sigma * error * J;

      cost += error * error;      
    }

    Eigen::Vector3d dx = H.ldlt().solve(b);
    if (std::isnan(dx[0])) {
      std::cout << "result is nan!" << std::endl;
      break;
    }

    if (iter > 0 && cost >= lastCost) {
      std::cout << "cost: " << cost << ">= last cost: " << lastCost << ", break." << std::endl;
      break;
    }

    ag += dx[0];
    bg += dx[1];
    cg += dx[2];

    lastCost = cost;

    std::cout << "total cost: " << cost << ", \t\tupdate: " << dx.transpose() <<
         "\t\testimated params: " << ag << "," << bg << "," << cg << std::endl;
  }
  std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
  std::chrono::duration<double> time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
  std::cout << "solve time cost = " << time_used.count() << " seconds. " << std::endl;

  std::cout << "estimated abc = " << ag << ", " << bg << ", " << cg << std::endl;
  return 0;
  
} 
