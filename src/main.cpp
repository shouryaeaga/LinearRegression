#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <math.h>
#include <iomanip>
#include "machine_learning_algos/LinearRegression.h"

void standardise(std::vector<std::vector<double>> &X) {
    int n = X.size();
    int m = X[0].size();
    for (int j = 0; j < m; j++) {
        // calculate mean
        double tot = 0;
        for (int i = 0; i < n; i++) tot+=X[i][j];
        double mean = tot/n;

        double stdev = 0;
        for (int i = 0; i < n; i++) stdev += pow(X[i][j] - mean, 2);
        stdev = sqrt(stdev / n);

        for (int i = 0; i < n; i ++) {
            if (stdev != 0)
                X[i][j] = (X[i][j] - mean)/stdev;
        }
    }
}

std::vector<std::vector<double>> loadCSV(const std::string &filename) {
    std::vector<std::vector<double>> data;
    std::ifstream file(filename);

    if (!file.is_open()) {
        throw std::runtime_error("Could not open file " + filename);
    }

    std::string line;
    std::vector<double> temp;
    while (getline(file, line)) {
        std::stringstream ss(line);
        while (ss.good()) {
            std::string substr;
            getline(ss, substr, ',');
            if (isdigit(substr.at(0))) {
                temp.push_back(std::stoi(substr));
            }
        }
        data.push_back(temp);
        temp.clear();
    }

    file.close();
    return data;
}

int main(int, char**){
    LinearRegression regression(0.001);
    std::vector<std::vector<double>> X = {{1.0, 2.0, 3.0, 4.0}, {2.0, 4.0, 6.0, 8.0}, {3.0, 6.0, 9.0, 12.0}};
    std::vector<double> y = {1, 2, 3};

    regression.fit(X, y);

    std::vector<double> result = regression.predict(X);
    for (auto& elem : result) {
        std::cout << elem << std::endl;
    }

    std::cout << "----- HOUSING DATASET ------\n";

    std::vector<std::vector<double>> data = loadCSV("../Housing.csv");
    LinearRegression housingDatasetRegression;
    X.clear();
    y.clear();
    for (int i = 0; i < data.size(); i++) {
        y.push_back(data[i][0]);
        data[i].erase(data[i].begin());
        X.push_back(data[i]);
    }

    std::vector<std::vector<double>> X_train = std::vector<std::vector<double>>(X.begin(), X.begin() + 300);
    std::vector<std::vector<double>> X_test = std::vector<std::vector<double>>(X.begin() + 300, X.end());
    std::vector<double> y_train = std::vector<double>(y.begin(), y.begin() + 300);
    std::vector<double> y_test = std::vector<double>(y.begin() + 300, y.end());

    standardise(X_train);
    standardise(X_test);

    std::cout << "loaded data\n";

    housingDatasetRegression.fit(X_train, y_train);

    std::cout << std::fixed << std::setprecision(2) << "R^2: " << housingDatasetRegression.evaluate(X_train, y_train) << std::endl;
}
