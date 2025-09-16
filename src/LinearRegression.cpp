#include "machine_learning_algos/LinearRegression.h"
#include <math.h>
#include <stdexcept>


void LinearRegression::fit(const std::vector<std::vector<double>> &X, const std::vector<int> &y){
    if (X.size() != y.size() || X.size() == 0) {
        throw std::invalid_argument("Incorrect dimensions provided");
    }
    if (X[0].size() == 0) {
        throw std::invalid_argument("No data");
    }

    LinearRegression::Weight intercept;
    intercept.type = WeightType::BIAS;
    LinearRegression::weights.push_back(intercept);
    for (int i = 0; i < X[0].size(); i++) {
        LinearRegression::Weight slope;
        slope.type = WeightType::SLOPE;

        LinearRegression::weights.push_back(slope);
    }
}

double LinearRegression::MSE_loss(const std::vector<int> &y_true, const std::vector<int> &y_pred)
{
    double loss = 0.0;
    int n = y_true.size();
    for (int i = 0; i < n; ++i) {
        loss += pow((y_pred[i] - y_true[i]), 2);
    }

    return loss/n;
}