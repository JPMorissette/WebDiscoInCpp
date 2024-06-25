#include "Node.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <limits>
#include <Eigen/Dense>
#include <cmath>
#include <algorithm>

using namespace Eigen;
using namespace std;

Node::Node(const string &filename, int i)
{
}

void Node::calculateTimes(const string &filename, int i)
{
    std::string input_filename = filename;
    std::string output_filename = "Times_" + std::to_string(i) + "_output.csv";
    
    std::ifstream infile(input_filename);
    std::ofstream outfile(output_filename);
    
    if (!infile.is_open() || !outfile.is_open()) {
        std::cerr << "Error opening file!" << std::endl;
        return;
    }
    
    std::set<std::string> event_times;
    std::string line;
    bool first_line = true;
    
    while (std::getline(infile, line)) {
        if (first_line) {
            first_line = false;
            continue;
        }
        std::vector<std::string> columns = split(line, ',');
        if (columns.size() < 2) {
            continue;
        }
        std::string time = columns[0];
        std::string status = columns[1];
        if (status == "1") {
            event_times.insert(time);
        }
    }
    
    for (const auto& time : event_times) {
        outfile << time << std::endl;
    }
    
    infile.close();
    outfile.close();
}

void Node::calculateParams(const string &filename, int i, int nbBetas)
{
    std::vector<std::vector<double>> node_data = readCSV(filename);
    std::vector<double> Dlist = readGlobalTimes("Global_times_output.csv");

    std::vector<std::vector<double>> Dik(Dlist.size());
    std::vector<std::vector<double>> Rik(Dlist.size());
    std::vector<std::vector<double>> sumZrh(Dlist.size(), std::vector<double>(nbBetas, 0.0));

    for (int i = 0; i < Dlist.size(); ++i) {
        for (int j = 0; j < node_data.size(); ++j) {
            if (abs(node_data[j][0] - Dlist[i]) <= 0.00000001 && node_data[j][1] == 1) {
                Dik[i].push_back(j + 1);
            }
            if (node_data[j][0] >= Dlist[i]) {
                Rik[i].push_back(j + 1);
            }
        }
        if (Dik[i].empty()) {
            Dik[i].push_back(0);
        }
    }

    for (int i = 0; i < Dik.size(); ++i) {
        const auto &indices = Dik[i];
        for (int x = 0; x < nbBetas; ++x) {
            double current_sum = 0.0;
            for (const auto &idx : indices) {
                if (idx > 0) {
                    current_sum += forceZero(node_data[idx - 1][3 + x - 1]);
                }
            }
            sumZrh[i][x] = current_sum;
        }
    }

    writeCSV(Dik, "Dik" + std::to_string(i) + ".csv");
    writeCSV(Rik, "Rik" + std::to_string(i) + ".csv");
    writeCSV(sumZrh, "sumZrh" + std::to_string(i) + ".csv");
}

void Node::calculateBetas(const string &filename, int k, int it, int nbBetas)
{
    auto node_data = readCSV("Data_site_" + to_string(k) + ".csv");
    auto beta_data = readBetasCsv("Beta_" + to_string(it) + "_output.csv");
    auto Rik_data = readCSVNoHeader("Rik" + to_string(k) + ".csv");

    VectorXd sumExp = VectorXd::Zero(Rik_data.size());
    MatrixXd sumZqExp = MatrixXd::Zero(Rik_data.size(), nbBetas);
    vector<MatrixXd> sumZqZrExp(Rik_data.size(), MatrixXd::Zero(nbBetas, nbBetas));

    for (size_t i = 0; i < Rik_data.size(); ++i) {
        vector<int> indices;
        for (const auto &val : Rik_data[i]) {
            if (val != -1) {
                indices.push_back(static_cast<int>(val));
            }
        }

        for (const auto &idx : indices) {
            VectorXd z(nbBetas);
            for (size_t x = 2; x < node_data[0].size(); ++x) {
                double value = node_data[idx-1][x];
                z(x - 2) = value;
            }
            
            double exp_val = exp((beta_data.transpose() * z).sum());
            sumExp[i] += exp_val;
            sumZqExp.row(i) += z.transpose() * exp_val;
            sumZqZrExp[i] += z * z.transpose() * exp_val;
        }
    }

    vector<vector<double>> combined_matrix(nbBetas * nbBetas, vector<double>(Rik_data.size()));
    for (size_t i = 0; i < Rik_data.size(); ++i) {
       for (int j = 0; j < nbBetas * nbBetas; ++j) {
           combined_matrix[j][i] = sumZqZrExp[i](j / nbBetas, j % nbBetas);
       }
    }

    eigenWriteCSV(sumExp, "sumExp" + to_string(k) + "_output_" + to_string(it) + ".csv");
    eigenWriteCSV(sumZqExp, "sumZqExp" + to_string(k) + "_output_" + to_string(it) + ".csv");
    writeCSV(combined_matrix, "sumZqZrExp" + to_string(k) + "_output_" + to_string(it) + ".csv");
}

vector<std::string> Node::split(const std::string &s, char delimiter)
{
    vector<std::string> tokens;
    string token;
    istringstream tokenStream(s);
    while (std::getline(tokenStream, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}

std::vector<std::vector<double>> Node::readCSV(const std::string &filename) {
    std::ifstream infile(filename);
    std::vector<std::vector<double>> data;
    std::string line;
    bool first_line = true;

    while (std::getline(infile, line)) {
        if (first_line) {
            first_line = false;
            continue; // Skip the header
        }
        std::vector<std::string> columns = split(line, ',');
        std::vector<double> row;
        for (const std::string &val : columns) {
            row.push_back(std::stod(val));
        }
        data.push_back(row);
    }
    return data;
}

std::vector<std::vector<double>> Node::readCSVNoHeader(const std::string &filename) {
    std::ifstream infile(filename);
    std::vector<std::vector<double>> data;
    std::string line;
    bool first_line = true;

    while (std::getline(infile, line)) {
        std::vector<std::string> columns = split(line, ',');
        std::vector<double> row;
        for (const std::string &val : columns) {
            row.push_back(std::stod(val));
        }
        data.push_back(row);
    }
    return data;
}

Eigen::MatrixXd Node::readBetasCsv(const std::string &filename) {
    std::ifstream infile(filename);
    std::vector<double> values; // Vector to store all values
    double value;

    // Read all values into a single vector
    while (infile >> value) {
        values.push_back(value);
    }

    // Determine number of rows and columns
    size_t num_rows = values.size(); // Each value is a row
    size_t num_cols = 1; // Since each value is in a separate line

    // Create an Eigen MatrixXd of size num_rows x num_cols
    Eigen::MatrixXd data(num_rows, num_cols);

    // Fill the Eigen matrix
    for (size_t i = 0; i < num_rows; ++i) {
        data(i, 0) = values[i]; // Each value goes into its own row
    }

    return data;
}

std::vector<double> Node::readGlobalTimes(const std::string &filename) {
    std::ifstream infile(filename);
    std::vector<double> times;
    std::string line;

    while (std::getline(infile, line)) {
        if (!line.empty()) {
            times.push_back(std::stod(line));
        }
    }
    return times;
}

void Node::eigenWriteCSV(const MatrixXd& mat, const string& filename) {
    ofstream file(filename);

    if (file.is_open()) {
        for (int i = 0; i < mat.rows(); ++i) {
            for (int j = 0; j < mat.cols(); ++j) {
                file << mat(i, j);
                if (j < mat.cols() - 1) {
                    file << ",";
                }
            }
            file << endl;
        }
        file.close();
        cout << "MatrixXd data written to " << filename << endl;
    } else {
        cerr << "Unable to open file: " << filename << endl;
    }
}

void Node::eigenWriteCSV(const VectorXd& vec, const string& filename) {
        ofstream file(filename);

        if (file.is_open()) {
            for (int i = 0; i < vec.size(); ++i) {
                file << vec(i) << endl; // Write each element followed by newline
            }
            file.close();
            cout << "VectorXd data written to " << filename << endl;
        } else {
            cerr << "Unable to open file: " << filename << endl;
        }
    }

double Node::forceZero(double value, double threshold) {
    if (std::abs(value) < threshold) {
        return 0.0;
    } else {
        return value;
    }
}

template <typename T>
void Node::writeCSV(const std::vector<std::vector<T>> &data, const std::string &filename) {
    std::ofstream outfile(filename);
    for (const auto &row : data) {
        for (size_t i = 0; i < row.size(); ++i) {
            if (row[i] != static_cast<T>(-1)) {
                outfile << row[i];
            }
            if (i < row.size() - 1) {
                outfile << ",";
            }
        }
        outfile << "\n";
    }
}