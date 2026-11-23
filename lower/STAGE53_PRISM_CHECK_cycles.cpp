// Count even and two-source bond sets by total degree and weak-bond degree.
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using Word = std::uint64_t;
using Counts = std::vector<std::vector<std::uint32_t>>;

int main() {
    int edges, rank, num_patterns;
    Word source_path;
    if (!(std::cin >> edges >> rank >> source_path >> num_patterns)) return 2;
    if (rank != 23 || edges != 50 || num_patterns <= 0) return 2;
    std::vector<Word> cycles(rank), weak(num_patterns);
    for (auto& cycle : cycles) std::cin >> cycle;
    for (auto& mask : weak) std::cin >> mask;
    if (!std::cin) return 2;

    std::vector<std::vector<Word>> masks[2];
    for (auto& by_degree : masks) by_degree.resize(edges+1);
    Word even = 0;
    for (std::uint32_t j = 0; j < (1u << rank); ++j) {
        if (j > 0) even ^= cycles[__builtin_ctz(j)];
        const Word odd = even ^ source_path;
        masks[0][__builtin_popcountll(even)].push_back(even);
        masks[1][__builtin_popcountll(odd)].push_back(odd);
    }

    std::vector<std::string> output(num_patterns);
    std::atomic<int> next{0};
    auto worker = [&]() {
        for (;;) {
            const int row = next.fetch_add(1);
            if (row >= num_patterns) return;
            std::ostringstream data;
            for (int parity = 0; parity <= 1; ++parity) {
                Counts counts(edges+1,std::vector<std::uint32_t>(edges+1,0));
                for (int degree = parity; degree <= edges; degree += 2) {
                    for (Word mask : masks[parity][degree])
                        ++counts[degree][__builtin_popcountll(mask & weak[row])];
                }
                data << row << ' ' << parity;
                std::uint64_t sum = 0;
                for (int degree = 0; degree <= edges; ++degree) {
                    for (int k = 0; k <= degree; ++k) {
                        const auto count = counts[degree][k];
                        if (count == 0) continue;
                        data << ' ' << degree << ' ' << k << ' ' << count;
                        sum += count;
                    }
                }
                if (sum != (1u << rank)) std::terminate();
                data << '\n';
            }
            output[row] = data.str();
        }
    };
    std::vector<std::thread> jobs;
    for (int j = 0; j < 3; ++j) jobs.emplace_back(worker);
    for (auto& job : jobs) job.join();
    for (const auto& line : output) std::cout << line;
}
