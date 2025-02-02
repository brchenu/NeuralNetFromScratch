#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <bitset>
#include <format>
#include <sstream>
#include <cassert>
#include <cmath>

class Value {
    std::vector<Value> parents;
    float data;
public:
    Value(float data, std::vector<Value> parents) : data(data), parents(parents) {}
    Value(float data): data(data) {}

    friend std::ostream& operator<<(std::ostream &os, const Value v) {
        return os << "Value(data=" <<  v.data << ")";
    }
};

std::vector<char> parseIdxFile(const std::string &filepath) {
    std::ifstream file(filepath, std::ios::binary);

    if (!file.is_open()) {
        std::cerr << "Error while opening the file";
        std::exit(EXIT_FAILURE);
    }

    file.seekg(0, std::ios::end);
    std::streamsize length = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buff(static_cast<size_t>(length));
    file.read(buff.data(), length);

    // Check if we read every expected bytes
    if (file.gcount() != length) {
        std::cerr << "Error | File was not read entirely";
        std::exit(EXIT_FAILURE);
    }

    file.close();

    return buff;
}

void printFirstBytes(int size, const std::vector<char> buffer) {
    assert(size <= buffer.size());
    for (int i = 0; i < size; i++) {
        for (int j = 7; j >= 0; j--) {
            std::cout << ((buffer[i] >> j) & 1);
        }
        std::cout << " ";
    }
    std::cout << "\n";
}

float sigmoid(float z) {
    return 1 / (1 + std::exp(-z));
}

float rootMeanSquared(std::vector<float> truth, std::vector<float> prediction) {
    assert(truth.size() == prediction.size());
    float sum = 0;
    for (int i = 0; i < truth.size(); i++) {
        sum += std::pow(truth[i] - prediction[i], 2);
    }
    return sum / truth.size();
}
 
int main() {
    std::cout << "Start\n";
    auto buffer = parseIdxFile("input/train-images.idx3-ubyte");

    printFirstBytes(4, buffer);

    return 0;
}