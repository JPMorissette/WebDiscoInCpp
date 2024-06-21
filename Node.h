#ifndef NODE_H
#define NODE_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <set>
#include <string>
#include "C:/Libraries/Eigen/eigen-3.4.0/Eigen/Dense"

using namespace Eigen;
using namespace std;

class Node {
public:
    Node(const string &filename, int i);

    void calculateTimes(const string &filename, int i);
    void calculateParams(const string &filename, int i, int nbBetas);
    void calculateBetas(const string &filename, int i, int it, int nbBetas);

private:
    vector<std::string> split(const std::string &s, char delimiter);

    std::vector<std::vector<double>> readCSV(const std::string &filename);
    std::vector<std::vector<double>> readCSVNoHeader(const std::string &filename);
    std::vector<double> readGlobalTimes(const std::string &filename);
    
    template <typename T>
    void writeCSV(const std::vector<std::vector<T>> &data, const std::string &filename);

    void eigenWriteCSV(const MatrixXd& mat, const string& filename);
    void eigenWriteCSV(const VectorXd& vec, const string& filename);

    Eigen::MatrixXd readBetasCsv(const std::string &filename);
    double forceZero(double value, double threshold = 1e-10);
};

#endif // NODE_H
