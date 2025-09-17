#include <vector>

class LinearRegression {
    public:
        LinearRegression(double learning_rate = 0.01, double threshold = 0.00001) : learning_rate(learning_rate), threshold(threshold) {};

        void fit(const std::vector<std::vector<double>>& X, const std::vector<int>& y);

        std::vector<int> predict(const std::vector<std::vector<double>>& X);
    private:
        enum WeightType {
            BIAS,
            SLOPE
        };
        struct Weight {
            WeightType type;
            double weight = 0.0;
        };

        std::vector<Weight> weights;

        double learning_rate;
        double threshold;

        double MSE_loss(const std::vector<int>& y_true, const std::vector<int>& y_pred);

        
};