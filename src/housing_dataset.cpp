#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

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
    }

    file.close();
    return data;
}

int main() {
    std::vector<std::vector<double>> data = loadCSV("../Housing.csv");
    
}