#include <iostream>
#include <string>
#include <fstream>

void parseIdxFile(const std::string &filepath) {
    std::ifstream file(filepath);

    if (!file.is_open()) {
        std::cerr << "Error while opening the file";
        std::exit(EXIT_FAILURE);
    }
}

int main() {
    std::cout << "\nStart";
    return 0;
}