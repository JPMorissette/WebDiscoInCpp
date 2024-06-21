/*
    Distributed Cox Model in C++
    License: https://creativecommons.org/licenses/by-nc-sa/4.0/
    Copyright: GRIIS / Université de Sherbrooke

    TODO: 
    - Clean Node.cpp (many different read and write functions, could be merged)
    - Test with big number of data
*/

#define STATS_ENABLE_EIGEN_WRAPPERS

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>
#include <set>
#include "Node.h"

using namespace std;
using namespace Eigen;

// Split values in a CSV (used in readDataFromCSV)
vector<string> split(const string &s, char delimiter) {
    vector<string> tokens;
    string token;
    istringstream tokenStream(s);
    while (getline(tokenStream, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}

// Read CSV file and store in a vector<vector>>
vector<vector<double>> readCSVVector(const string& filename) {
    ifstream file(filename);
    vector<vector<double>> data;
    string line;

    while (getline(file, line)) {
        stringstream lineStream(line);
        vector<double> row;
        string cell;

        while (getline(lineStream, cell, ',')) {
            row.push_back(stod(cell));
        }

        data.push_back(row);
    }

    return data;
}

// Read data from CSV (no header)
MatrixXd readDataFromCSV(const string &filename) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error: Could not open the file " << filename << endl;
        exit(1);
    }

    vector<vector<string>> table;
    string line;
    while (getline(file, line)) {
        vector<string> fields = split(line, ',');
        table.push_back(fields);
    }
    file.close();

    int rows = table.size();
    int cols = 0;
    for (const auto& row : table) {
        if (row.size() > cols) {
            cols = row.size();
        }
    }

    MatrixXd m(rows, cols);
    m.setZero();

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < table[i].size(); ++j) {
            try {
                if (table[i][j].empty()) {
                    m(i, j) = 0.0; 
                } else {
                    m(i, j) = std::stod(table[i][j]);
                }
            } catch (const std::invalid_argument& e) {
                cerr << "Invalid argument: " << e.what() << endl;
                exit(1);
            } catch (const std::out_of_range& e) {
                cerr << "Out of range: " << e.what() << endl;
                exit(1);
            }
        }
    }

    return m;
}

// Write data in CSV
void writeResultsToCSV(const string& filename, const MatrixXd& matrix) {
    ofstream outputFile(filename);
    outputFile << fixed << setprecision(15);
    if (outputFile.is_open()) {
        for (int i = 0; i < matrix.rows(); ++i) {
            for (int j = 0; j < matrix.cols(); ++j) {
                outputFile << matrix(i, j);
                if (j < matrix.cols() - 1) {
                    outputFile << ",";
                }
            }
            outputFile << endl;
        }
        outputFile.close();
        cout << "CSV file created successfully: " << filename << endl;
    } else {
        cerr << "Unable to create CSV file: " << filename << endl;
    }
}

// Force to zero values that are smaller then the threshold
double forceZero(double value, double threshold = 1e-10) {
    if (std::abs(value) < threshold) {
        return 0.0;
    }
    else {
        return value;
    }
}

int main() {
    const int nbOfSites = 3;
    const int nbBetas = 3;
    const int maxIt = 2;

    // Create sites (needed for data storage)
    vector<Node> sites;
    for (int i = 1; i <= nbOfSites; ++i) {
        string filename = "Data_site_" + to_string(i) + ".csv";
        sites.emplace_back(filename, i);
    }

     // LOCAL: Calculate times for each site
    for (int i = 1; i <= nbOfSites; ++i) {
        string filename = "Data_site_" + to_string(i) + ".csv";
        sites[i-1].calculateTimes(filename, i);
    }

    // GLOBAL: Read and combine times from all sites
    vector<MatrixXd> combined_times;
    for (int k = 1; k <= nbOfSites; ++k) {
        string filename = "Times_" + to_string(k) + "_output.csv";
        MatrixXd times = readDataFromCSV(filename);
        combined_times.push_back(times);
    }

    int total_elements = 0;
    for (const auto& matrix : combined_times) {
        total_elements += matrix.size();
    }

    MatrixXd ordered_times(total_elements, 1);
    int current_index = 0;
    for (const auto& matrix : combined_times) {
        ordered_times.block(current_index, 0, matrix.size(), 1) = matrix;
        current_index += matrix.size();
    }

    std::sort(ordered_times.col(0).data(), ordered_times.col(0).data() + ordered_times.col(0).size());

    // Write the sorted times to a new CSV file
    writeResultsToCSV("Global_times_output.csv", ordered_times);

    // LOCAL: Calculate params for each site
    for (int i = 1; i <= nbOfSites; ++i) {
        string filename = "Data_site_" + to_string(i) + ".csv";
        sites[i-1].calculateParams(filename, i, nbBetas);
    }

    // GLOBAL: Calculate parameters (sumZr, normDi)
    MatrixXd sumZrGlobal = MatrixXd::Zero(0, 0);
    MatrixXd normDikGlobal = MatrixXd::Zero(0, 1);

    for (int i = 1; i <= nbOfSites; ++i) {
        string Dik_filename = "Dik" + to_string(i) + ".csv";
        MatrixXd Dik_data = readDataFromCSV(Dik_filename);
        if (normDikGlobal.rows() == 0) {
            normDikGlobal = MatrixXd::Zero(Dik_data.rows(), 1);
        }
        for (int j = 0; j < Dik_data.rows(); ++j) {
            normDikGlobal(j) += (Dik_data.row(j).array() != 0).count();
        }

        string sumZrh_filename = "sumZrh" + to_string(i) + ".csv";
        MatrixXd temp = readDataFromCSV(sumZrh_filename);

        if (sumZrGlobal.rows() == 0) {
            sumZrGlobal = MatrixXd::Zero(temp.rows(), temp.cols());
        }
        sumZrGlobal += temp;
    }

    writeResultsToCSV("normDikGlobal.csv", normDikGlobal);
    writeResultsToCSV("sumZrGlobal.csv", sumZrGlobal);

    // Check if Beta_1_output.csv exists; if not, initialise first beta
    if (!ifstream("Beta_1_output.csv")) {
        MatrixXd beta = MatrixXd::Zero(nbBetas, 1);
        writeResultsToCSV("Beta_1_output.csv", beta);
    }

    // Main computation loop
    for (int it = 1; it <= maxIt; ++it){

        // LOCAL: Calculate aggregates for beta estimation
        for (int i = 1; i <= nbOfSites; ++i) {
            string filename = "Data_site_" + to_string(i) + ".csv";
            sites[i-1].calculateBetas(filename, i, it, nbBetas);
        }

        // GLOBAL: Use local data to calculate new betas
        string fileName = "sumExp1_output_" + to_string(it) + ".csv";
        ifstream file(fileName);
        if (file.good()) {
            string betaFileName = "Beta_" + to_string(it) + "_output.csv";
            MatrixXd beta = readDataFromCSV(betaFileName);

            // Initialize matrices and arrays
            MatrixXd Exp_data = readDataFromCSV("sumExp1_output_1.csv");
            MatrixXd sumExpGlobal = MatrixXd::Zero(Exp_data.rows(), 1);

            MatrixXd ExpZq_data = readDataFromCSV("sumZqExp1_output_1.csv");
            MatrixXd sumZqExpGlobal = MatrixXd::Zero(ExpZq_data.rows(), nbBetas);

            //nbBetas, nbBetas, ncol(sumZqZrExp)
            MatrixXd Rik_data = readDataFromCSV("Global_times_output.csv");
            vector<MatrixXd> sumZqZrExp(Rik_data.size(), MatrixXd::Zero(nbBetas, nbBetas));
            vector<MatrixXd> sumZqZrExpGlobal(Rik_data.size(), MatrixXd::Zero(nbBetas, nbBetas));

            // Read files and sum values
            for (int i = 1; i <= nbOfSites; ++i) {
                string Exp_filename = "sumExp" + to_string(i) + "_output_" + to_string(it) + ".csv";
                Exp_data = readDataFromCSV(Exp_filename);
                sumExpGlobal += Exp_data;

                string ExpZq_filename = "sumZqExp" + to_string(i) + "_output_" + to_string(it) + ".csv";
                ExpZq_data = readDataFromCSV(ExpZq_filename);
                sumZqExpGlobal += ExpZq_data;

                string ExpZqZr_filename = "sumZqZrExp" + to_string(i) + "_output_" + to_string(it) + ".csv";
                MatrixXd ExpZqZr_data = readDataFromCSV(ExpZqZr_filename);
                vector<vector<double>> combined_matrix_data = readCSVVector("sumZqZrExp" + to_string(i) + "_output_" + to_string(it) + ".csv"); 

                for (size_t i = 0; i < Rik_data.size(); ++i) {
                    for (int j = 0; j < nbBetas * nbBetas; ++j) {
                        sumZqZrExp[i](j / nbBetas, j % nbBetas) = combined_matrix_data[j][i];
                    }
                    sumZqZrExpGlobal[i] += sumZqZrExp[i];
                }
            }

            // Calculate parameters
            MatrixXd sumZrGlobal_int = sumZrGlobal.colwise().sum();

            // Calculate first derivative
            MatrixXd sumExpGlobalReplicated = sumExpGlobal.replicate(1, nbBetas);
            MatrixXd ZrExp_Exp = sumZqExpGlobal.array() / sumExpGlobalReplicated.array();

            MatrixXd normGlobalReplicated = normDikGlobal.replicate(1, nbBetas);
            MatrixXd Norm_ZrExp_Exp = normGlobalReplicated.array() * ZrExp_Exp.array();

            MatrixXd sumDi_Norm_ZrExp_Exp = Norm_ZrExp_Exp.colwise().sum(); 
            MatrixXd lr_beta = sumZrGlobal_int.array() - sumDi_Norm_ZrExp_Exp.array();

            // Calculate second derivative
            MatrixXd lrq_beta = MatrixXd::Zero(nbBetas, nbBetas);
            for (int i = 0; i < nbBetas; ++i) {
                for (int j = 0; j < nbBetas; ++j) {

                    MatrixXd elements = MatrixXd::Zero(Rik_data.size(), 1);
                    for (int k = 0; k < Rik_data.size(); ++k) {
                        elements(k, 0) = sumZqZrExpGlobal[k](i, j);
                    }

                    MatrixXd a = elements.array() / sumExpGlobal.array();
                    MatrixXd b = sumZqExpGlobal.col(i).array() / sumExpGlobal.array();
                    MatrixXd c = sumZqExpGlobal.col(j).array() / sumExpGlobal.array();

                    MatrixXd value_ij = a.array() - b.array() * c.array();
                    MatrixXd Norm_value_ij = normDikGlobal.array() * value_ij.array();
                    lrq_beta(i, j) = -Norm_value_ij.sum();
                }
            }

            // Calculate new beta
            MatrixXd lrq_beta_inv = lrq_beta.inverse();
            MatrixXd betaT = lr_beta.transpose();
            MatrixXd temp = lrq_beta_inv * betaT;
            MatrixXd new_beta = beta.array() - temp.array();

            // Write new_beta to CSV file Beta_(ite+1)_output.csv
            string newBetaFileName = "Beta_" + to_string(it + 1) + "_output.csv";
            writeResultsToCSV(newBetaFileName, new_beta);
        }
    }
    
    return 0;
}
