#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <math.h>
#include <iomanip>
#include <ctime>
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

    std::vector<std::vector<double>> data = loadCSV("../Housing_Cleaned.csv");
    LinearRegression housingDatasetRegression;
    X.clear();
    y.clear();
    for (int i = 0; i < data.size(); i++) {
        y.push_back(data[i][0]);
        data[i].erase(data[i].begin());
        X.push_back(data[i]);
    }

    

    //random shuffle
    std::srand(std::time(0));
    for (int i = 0; i < X.size(); i ++) {
        int j = std::rand() % X.size();
        std::swap(X[i], X[j]);
        std::swap(y[i], y[j]);
    }

    std::vector<std::vector<double>> X_train = {X.begin(), X.begin() + (int)(0.8*X.size())};
    std::vector<std::vector<double>> X_test = {X.begin() + (int)(0.8*X.size()), X.end()};
    std::vector<double> y_train = {y.begin(), y.begin() + (int)(0.8*y.size())};
    std::vector<double> y_test = {y.begin() + (int)(0.8*y.size()), y.end()};


    standardise(X_train);
    standardise(X_test);

    std::cout << "loaded data\n";

    housingDatasetRegression.fit(X_train, y_train);

    std::cout << std::fixed << std::setprecision(2) << "R^2: " << housingDatasetRegression.evaluate(X_train, y_train) << std::endl;
}
