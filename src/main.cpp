#include <iostream>
#include <vector>
#include "machine_learning_algos/LinearRegression.h"

int main(int, char**){
    LinearRegression regression;
    std::vector<std::vector<double>> X = {{1.0, 3.0}, {4.0, 12.2}};
    std::vector<int> y = {143, 234};

    regression.fit(X, y);
}
