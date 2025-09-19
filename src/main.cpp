#include <iostream>
#include <vector>
#include "machine_learning_algos/LinearRegression.h"

int main(int, char**){
    LinearRegression regression;
    std::vector<std::vector<double>> X = {{1.0}, {2.0}};
    std::vector<double> y = {79, 158};

    regression.fit(X, y);

    std::vector<double> result = regression.predict(X);
    std::cout << "Predicted " << result[0] << " " << result[1] << std::endl;
}
