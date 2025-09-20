# Linear Regression in C++

This repository contains a simple implementation of **Linear Regression** in C++ as a learning project.  
It includes `src/main.cpp` as example code you can use, and `include/machine_learning_algos/LinearRegression.h` and `src/LinearRegression.cpp` which contains the class for the implementation which you can copy and paste into your project.

---

## Api

```cpp
LinearRegression regression(double learning_rate, double threshold)
// threshold is after how small of a difference in MSE (mean squared error) should the program stop.
.fit(std::vector<std::vector<double>> X, std::vector<double> y);
// Fits the model to the training data

.predict(std::vector<std::vector<double>> X);
// Predicts for each input in X, returns a std::vector<double>

.evaluate(std::vector<std::vector<double>> X_test, std::vector<double> y_test);
// Calculates and returns the r^2, to evaluate the model based on the test data provided
```
---

## Goals
 - This project helped me understand how linear regression works internally, and more about how machine learning algorithms work generally including:
    - Cost functions
    - Optimisation (gradient descent in this case)
    - Requirements for scaling (standardisation in this case)
    - Metrics (R^2 metric here)

## Performance
 - Using the housing dataset, which I cleaned a bit using python, I managed to achieve an R^2 score of around 0.68, which is similar to scikit-learn's score.
