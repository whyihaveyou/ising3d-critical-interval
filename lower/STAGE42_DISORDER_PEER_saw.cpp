// A fresh positive vector for the unweighted 16-step walk matrix.
// Reuse the geometry builder with independent all-direction row sums.
#define main archived_weighted_main
#include "verify_saw16_direct_straight_20261121.cpp"
#undef main

int main() {
    const Inputs input{16, 1, 1, 12994574, 53221763};
    const auto graph = build(input);
    std::size_t edges = 0;
    for (const auto& row : graph) {
        for (int next : row.next) {
            if (next < -1 || next >= int(graph.size()))
                throw std::runtime_error("out-of-range row index");
            edges += next >= 0;
        }
    }
    if (graph.size() != input.states || edges != input.edges)
        throw std::runtime_error("wrong number of states or transitions");
    std::cout << "unweighted graph " << graph.size() << " rows, " << edges
              << " transitions" << std::endl;
    check_published_short_counts(graph);
    const Integer scale_floor = Integer{1} << 48;
    std::vector<Integer> h(graph.size(), scale_floor), next(graph.size());
    for (int iteration = 0; iteration <= 90; ++iteration) {
        Integer high = 0;
        std::size_t worst = 0;
        for (std::size_t i = 0; i < graph.size(); ++i) {
            Integer sum = 0;
            for (int j : graph[i].next)
                if (j >= 0) sum += h[j];
            next[i] = sum;
            high = std::max(high, sum);
            if (Wide(sum)*h[worst] > Wide(next[worst])*h[i])
                worst = i;
        }
        if (Wide(next[worst])*100000 <= Wide(473195)*h[worst]) {
            auto divisor = std::gcd(next[worst], h[worst]);
            std::cout << "power-of-two vector iteration=" << iteration
                      << " all-row upper=" << next[worst]/divisor << "/"
                      << h[worst]/divisor << " <= 473195/100000" << std::endl;
            return 0;
        }
        unsigned shift = 0;
        while ((high >> shift) > 2*scale_floor) ++shift;
        for (auto& value : next) value = (value >> shift) + 1;
        h.swap(next);
    }
    throw std::runtime_error("positive vector did not pass required rational bound");
}
