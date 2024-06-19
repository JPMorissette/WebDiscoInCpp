#define STATS_ENABLE_EIGEN_WRAPPERS

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <stats.hpp>
#include <iomanip>
#include <set>
#include "Node.h"

using namespace std;
using namespace Eigen;

vector<long double> readTimesFromFile(const string& filename) {
    vector<long double> times;
    ifstream file(filename);
    string line;

    while (getline(file, line)) {
        stringstream ss(line);
        string value;
        while (getline(ss, value, ',')) {
            try {
                times.push_back(stold(value));
            } catch (const invalid_argument& e) {
                cerr << "Invalid number in file: " << filename << endl;
            }
        }
    }

    file.close();
    return times;
}

vector<vector<double>> readCSV(const string& filename) {
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

template <typename T>
void writeVectorToFile(const vector<T>& vec, const string& filename, const string& delimiter = "\n") {
    ofstream file(filename);
    file << fixed << setprecision(15);
    for (size_t i = 0; i < vec.size(); ++i) {
        file << vec[i];
        if (i < vec.size() - 1) {
            file << delimiter;
        }
    }
    file.close();
}

vector<string> split(const string &s, char delimiter) {
    vector<string> tokens;
    string token;
    istringstream tokenStream(s);
    while (getline(tokenStream, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}

double forceZero(double value, double threshold = 1e-10) {
    if (std::abs(value) < threshold) {
        return 0.0;
    } else {
        return value;
    }
}


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
    int cols = table.empty() ? 0 : table[0].size();
    MatrixXd m(rows, cols);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
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

void writeResultsToCSV(const string& filename, const MatrixXd& matrix) {
    ofstream outputFile(filename);
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

int main() {
    const int nbOfSites = 3;
    const int nbBetas = 7;
    const int maxIt = 3;

    // Create sites
    vector<Node> sites;
    for (int i = 1; i <= nbOfSites; ++i) {
        string filename = "Data_site_" + to_string(i) + ".csv";
        sites.emplace_back(filename, i);
    }

    // Calculate times for each site
    for (int i = 1; i <= nbOfSites; ++i) {
        string filename = "Data_site_" + to_string(i) + ".csv";
        sites[i-1].calculateTimes(filename, i);
    }

    // Read and combine times from all sites
    vector<long double> combined_times;
    for (int k = 1; k <= nbOfSites; ++k) {
        string filename = "Times_" + to_string(k) + "_output.csv";
        vector<long double> times = readTimesFromFile(filename);
        combined_times.insert(combined_times.end(), times.begin(), times.end());
    }

    // Remove duplicates and sort the times
    set<long double> unique_times(combined_times.begin(), combined_times.end());
    vector<long double> sorted_times(unique_times.begin(), unique_times.end());

    // Write the sorted times to a new CSV file
    writeVectorToFile(sorted_times, "Global_times_output.csv");

    // Calculate params for each site
    for (int i = 1; i <= nbOfSites; ++i) {
        string filename = "Data_site_" + to_string(i) + ".csv";
        sites[i-1].calculateParams(filename, i, nbBetas);
    }

    // Calculate parameters (sumZr, normDi)
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

    // Check if Beta_1_output.csv exists; if not, create and write zeros
    if (!ifstream("Beta_1_output.csv")) {
        MatrixXd beta = MatrixXd::Zero(nbBetas, 1);
        writeResultsToCSV("Beta_1_output.csv", beta);
    }

    // Main computation loop would go here
    for (int it = 1; it <= maxIt; ++it){

        // Calculate data for beta estimation for each site
        for (int i = 1; i <= nbOfSites; ++i) {
            string filename = "Data_site_" + to_string(i) + ".csv";
            cout << i << endl;
            sites[i-1].calculateBetas(filename, i, it, nbBetas);
        }

        // Use local estimates to calculate new betas
        string fileName = "sumExp1_output_" + to_string(it) + ".csv";
        ifstream file(fileName);
        if (file.good()) {
            string betaFileName = "Beta_" + to_string(it) + "_output.csv";
            MatrixXd beta = readDataFromCSV(betaFileName);

            // Initialize matrices and arrays
            string Exp_filename = "sumExp1_output_1.csv";
            MatrixXd Exp_data = readDataFromCSV(Exp_filename);
            MatrixXd sumExpGlobal = MatrixXd::Zero(Exp_data.rows(), 1);

            string ExpZq_filename = "sumZqExp1_output_1.csv";
            MatrixXd ExpZq_data = readDataFromCSV(ExpZq_filename);
            MatrixXd sumZqExpGlobal = MatrixXd::Zero(ExpZq_data.rows(), nbBetas);

            //nbBetas, nbBetas, ncol(sumZqZrExp)
            MatrixXd Rik_data = readDataFromCSV("Global_times_output.csv");
            vector<MatrixXd> sumZqZrExp(Rik_data.size(), MatrixXd::Zero(nbBetas, nbBetas));
            vector<MatrixXd> sumZqZrExpGlobal(Rik_data.size(), MatrixXd::Zero(nbBetas, nbBetas));

            // Read files and sum values
            for (int i = 1; i <= nbOfSites; ++i) {
                Exp_filename = "sumExp" + to_string(i) + "_output_" + to_string(it) + ".csv";
                Exp_data = readDataFromCSV(Exp_filename);
                sumExpGlobal += Exp_data;

                ExpZq_filename = "sumZqExp" + to_string(i) + "_output_" + to_string(it) + ".csv";
                ExpZq_data = readDataFromCSV(ExpZq_filename);
                sumZqExpGlobal += ExpZq_data;

                string ExpZqZr_filename = "sumZqZrExp" + to_string(i) + "_output_" + to_string(it) + ".csv";
                MatrixXd ExpZqZr_data = readDataFromCSV(ExpZqZr_filename);

                vector<vector<double>> combined_matrix_data = readCSV("sumZqZrExp" + to_string(i) + "_output_" + to_string(it) + ".csv"); 

                // Restore data from combined_matrix_data to sumZqZrExp
                for (size_t i = 0; i < Rik_data.size(); ++i) {
                    for (int j = 0; j < nbBetas * nbBetas; ++j) {
                        sumZqZrExp[i](j / nbBetas, j % nbBetas) = combined_matrix_data[j][i];
                    }
                }

                for (size_t i = 0; i < sumZqZrExpGlobal.size(); ++i) {
                    sumZqZrExpGlobal[i] += sumZqZrExp[i];
                }

            }

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
