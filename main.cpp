#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <bitset>
#include <format>
#include <sstream>
#include <cassert>

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

int main() {
    std::cout << "Start\n";
    auto buffer = parseIdxFile("input/train-images.idx3-ubyte");

    printFirstBytes(4, buffer);

    return 0;
}