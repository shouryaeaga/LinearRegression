#include "machine_learning_algos/LinearRegression.h"
#include <math.h>
#include <stdexcept>
#include <iostream>

void LinearRegression::fit(const std::vector<std::vector<double>> &X, const std::vector<int> &y){
    if (X.size() != y.size() || X.size() == 0) {
        throw std::invalid_argument("Incorrect dimensions provided");
    }
    if (X[0].size() == 0) {
        throw std::invalid_argument("No data");
    }

    int n = X.size();

    LinearRegression::Weight intercept;
    intercept.type = WeightType::BIAS;
    LinearRegression::weights.push_back(intercept);
    for (int i = 0; i < X[0].size(); i++) {
        LinearRegression::Weight slope;
        slope.type = WeightType::SLOPE;

        LinearRegression::weights.push_back(slope);
    }

    std::vector<int> predicted = LinearRegression::predict(X);


    double previous_loss = INFINITY;
    double loss = LinearRegression::MSE_loss(y, predicted);

    while (abs(loss-previous_loss) > threshold) {
        // Update bias
        int total_loss = 0;
        for (int i = 0; i < predicted.size(); i++) {
            total_loss += predicted[i] - y[i];
        }
        double bias_gradient = (2/n)*total_loss;
        weights[0].weight -= learning_rate*bias_gradient;
        
        // Update weights
        
        for (int feature = 0; feature < X[0].size(); feature++) {
            int e = 0;
            for (int i = 0; i < X.size(); i++) {
                e += (predicted[i] - y[i])*X[i][feature];
            }
        }

        predicted = LinearRegression::predict(X);
        previous_loss = loss;
        loss = LinearRegression::MSE_loss(y, predicted);

    }

    std::cout << weights[0].weight << std::endl;
    std::cout << predicted[0] << " " << predicted[1] << std::endl;

    
}

std::vector<int> LinearRegression::predict(const std::vector<std::vector<double>> &X) {
    if (X.size() == 0) {
        throw std::invalid_argument("Input data has improper dimensions.");
    }
    if (X[0].size() != weights.size() - 1) {
        throw std::invalid_argument("Input data has improper dimensions.");
    }

    std::vector<int> y;

    for (int h = 0; h < X.size(); h++) {
        double result = 0;
        for (int i = 0; i < weights.size(); i++) {
            if (weights[i].type == BIAS) {
                result += weights[i].weight;
            } else {
                result += X[h][i-1] * weights[i].weight;
            }
        }
        y.push_back(round(result));
    }

    return y;

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