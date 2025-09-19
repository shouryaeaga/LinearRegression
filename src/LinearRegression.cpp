#include "machine_learning_algos/LinearRegression.h"
#include <math.h>
#include <stdexcept>
#include <iostream>

void LinearRegression::fit(const std::vector<std::vector<double>> &X, const std::vector<double> &y){
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

    std::vector<double> predicted = LinearRegression::predict(X);


    double previous_loss = INFINITY;
    double loss = LinearRegression::MSE_loss(y, predicted);

    while (abs(loss-previous_loss) > threshold) {
        // Update bias
        double total_loss = 0;
        for (int i = 0; i < predicted.size(); i++) {
            total_loss += (2.0/n)*(predicted[i] - y[i]);
        }
        weights[0].weight -= learning_rate*total_loss;
        
        // Update weights
        
        for (int feature = 0; feature < X[0].size(); feature++) {
            double e = 0;
            for (int i = 0; i < X.size(); i++) {
                e += (2.0/n)*(predicted[i] - y[i])*X[i][feature];
            }
            weights[feature+1].weight -= learning_rate*e;
        }

        predicted = LinearRegression::predict(X);
        previous_loss = loss;
        loss = LinearRegression::MSE_loss(y, predicted);

    }
    
}

std::vector<double> LinearRegression::predict(const std::vector<std::vector<double>> &X) {
    if (X.size() == 0) {
        throw std::invalid_argument("Input data has improper dimensions.");
    }
    if (X[0].size() != weights.size() - 1) {
        throw std::invalid_argument("Input data has improper dimensions.");
    }

    std::vector<double> y;

    for (int h = 0; h < X.size(); h++) {
        double result = 0;
        for (int i = 0; i < weights.size(); i++) {
            if (weights[i].type == BIAS) {
                result += weights[i].weight;
            } else {
                result += X[h][i-1] * weights[i].weight;
            }
        }
        y.push_back(result);
    }

    return y;

}


double LinearRegression::MSE_loss(const std::vector<double> &y_true, const std::vector<double> &y_pred)
{
    double loss = 0.0;
    int n = y_true.size();
    for (int i = 0; i < n; ++i) {
        loss += pow((y_pred[i] - y_true[i]), 2)/n;
    }

    return loss;
}

float LinearRegression::evaluate(std::vector<std::vector<double>> X, std::vector<double> y) {
    double mean = 0;
    for (int i = 0; i < y.size(); i++) {
        mean += y[i];
    }
    mean /= y.size();
    double sum_predict_diff = 0;
    double sum_mean_diff = 0;
    for (int i = 0; i < y.size(); i++) {
        double prediction = predict({X[i]})[0];
        sum_predict_diff += pow((y[i] - prediction), 2);
        sum_mean_diff += pow((y[i] - mean), 2);
    }
    return 1 - (sum_predict_diff/sum_mean_diff);
}