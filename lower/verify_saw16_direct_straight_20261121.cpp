// Direct memory-16 weighted row bound. Adapted from verify_saw14_turn_weighted_20261121.cpp.
// The canonical suffix rule is that of verify_saw_trimmed_memory_20261121.cpp.
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

using Integer = std::uint64_t;
using Wide = unsigned __int128;
using Point = std::array<int, 3>;

constexpr int maximum_memory = 16;
constexpr int small_length = 8;
constexpr Integer scale = Integer{1} << 40;
constexpr Integer weight_num = 352018;
constexpr Integer weight_den = 352275;
static_assert(0 < weight_num && weight_num <= weight_den);
static_assert(6 * (scale + 16) < UINT64_MAX / weight_den);

struct Inputs {
    int memory;
    Integer weight_num, weight_den;
    std::size_t states, edges;
};

struct Row {
    std::array<int, 6> next;
    int previous_direction;
};

Point moved(Point p, int direction) {
    p[direction / 2] += (direction % 2 == 0) ? 1 : -1;
    return p;
}

Integer canonical(const int* steps, int begin, int end) {
    int axes[3] = {-1, -1, -1}, signs[3] = {}, used = 0;
    Integer result = 1;
    for (int j = begin; j < end; ++j) {
        int axis = steps[j] / 2, sign = steps[j] % 2;
        if (axes[axis] < 0) {
            axes[axis] = used++;
            signs[axis] = sign;
        }
        result = (result << 3) | (2 * axes[axis] + (sign ^ signs[axis]));
    }
    return result;
}

// Only slot IDs live in the open-address table. Full keys stay in states and
// are compared on every collision. The IDs follow the original BFS order.
std::vector<Row> build(const Inputs& inputs) {
    const int memory = inputs.memory;
    constexpr std::uint32_t slots_count = 1u << 24;
    constexpr auto empty = UINT32_MAX;
    if (inputs.states >= slots_count || inputs.states > INT32_MAX)
        throw std::invalid_argument("hash table or row ID too small");
    auto slot = [](Integer key) -> std::uint32_t {
        // SplitMix64 finalizer. Integer wrap is only used to choose the slot.
        key = (key ^ (key >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
        key = (key ^ (key >> 27)) * UINT64_C(0x94d049bb133111eb);
        return (key ^ (key >> 31)) & (slots_count - 1);
    };
    std::vector<Integer> states{1};
    states.reserve(inputs.states);
    std::vector<std::uint32_t> index(slots_count, empty);
    index[slot(1)] = 0;
    std::vector<Row> graph;
    graph.reserve(inputs.states);
    for (std::size_t i = 0; i < states.size(); ++i) {
        int steps[maximum_memory + 1], length = 0;
        for (Integer encoded = states[i]; encoded > 1; encoded >>= 3) {
            if (length >= memory - 1 || (encoded & 7) >= 6)
                throw std::runtime_error("invalid suffix");
            steps[length++] = static_cast<int>(encoded & 7);
        }
        std::reverse(steps, steps + length);
        Point vertices[maximum_memory + 2] = {};
        for (int j = 0; j < length; ++j)
            vertices[j+1] = moved(vertices[j], steps[j]);
        Row row;
        row.next.fill(-1);
        row.previous_direction = length ? steps[length-1] : -1;
        for (int d = 0; d < 6; ++d) {
            Point next = moved(vertices[length], d);
            if (std::find(vertices, vertices + length + 1, next) != vertices + length + 1)
                continue;
            int begin = 0;
            while (begin <= length) {
                int distance = 0;
                for (int a = 0; a < 3; ++a)
                    distance += std::abs(next[a] - vertices[begin][a]);
                if (length + 1 - begin + distance <= memory) break;
                ++begin;
            }
            if (begin > length) throw std::logic_error("previous vertex must be retained");
            steps[length] = d;
            Integer code = canonical(steps, begin, length+1);
            auto pos = slot(code);
            while (index[pos] != empty && states[index[pos]] != code)
                pos = (pos + 1) & (slots_count - 1);
            if (index[pos] == empty) {
                if (states.size() == inputs.states) throw std::runtime_error("unexpected extra state");
                index[pos] = static_cast<std::uint32_t>(states.size());
                states.push_back(code);
            }
            row.next[d] = static_cast<int>(index[pos]);
        }
        graph.push_back(row);
    }
    return graph;
}

using StraightCounts = std::array<std::array<Integer, small_length+1>, small_length+1>;

void direct_walks(std::vector<Point>& path, int last, int straight, StraightCounts& totals) {
    int length = static_cast<int>(path.size()) - 1;
    ++totals[length][straight];
    if (length == small_length) return;
    for (int d = 0; d < 6; ++d) {
        Point next = moved(path.back(), d);
        if (std::find(path.begin(), path.end(), next) != path.end()) continue;
        path.push_back(next);
        direct_walks(path, d, straight + (d == last), totals);
        path.pop_back();
    }
}

void check_straight_polynomials(const std::vector<Row>& graph) {
    StraightCounts direct{}, grouped{};
    std::vector<Point> path{{0, 0, 0}};
    direct_walks(path, -1, 0, direct);
    using State = std::pair<int, int>;  // row, number of straight steps
    std::map<State, Integer> counts{{{0, 0}, 1}};
    for (int n = 0; n <= small_length; ++n) {
        std::map<State, Integer> following;
        for (const auto& item : counts) {
            const auto [id, straight] = item.first;
            grouped[n][straight] += item.second;
            if (n == small_length) continue;
            for (int d = 0; d < 6; ++d) {
                int id_next = graph[id].next[d];
                if (id_next >= 0)
                    following[{id_next, straight + (d == graph[id].previous_direction)}]
                        += item.second;
            }
        }
        counts.swap(following);
    }
    if (direct != grouped) throw std::runtime_error("straight-count polynomial mismatch");
    std::cout << "checked straight-count polynomials through n=" << small_length << "\n";
}

std::vector<Integer> weighted_image(const std::vector<Row>& graph,
                                    const std::vector<Integer>& vector,
                                    const Inputs& inputs) {
    std::vector<Integer> result(graph.size());
    for (std::size_t i = 0; i < graph.size(); ++i) {
        const Row& row = graph[i];
        Integer total = 0;
        for (int j : row.next)
            if (j >= 0) total += vector[j];
        total *= inputs.weight_den;
        if (row.previous_direction >= 0) {
            int j = row.next[row.previous_direction];
            if (j >= 0) total -= (inputs.weight_den-inputs.weight_num) * vector[j];
        }
        result[i] = total;
    }
    return result;
}

void check_published_short_counts(const std::vector<Row>& graph) {
    constexpr Integer counts[] = {1, 6, 30, 150, 726, 3534, 16926, 81390, 387966,
        1853886, 8809878, 41934150, 198842742, 943974510, 4468911678, 21175146054,
        100121875974, 473730252102};
    std::vector<Integer> current(graph.size(), 1), next(graph.size());
    // Six choices per step give 6^17 < 2^44, including every non-root state.
    for (int n = 1; n <= 17; ++n) {
        for (std::size_t i = 0; i < graph.size(); ++i) {
            Integer sum = 0;
            for (int j : graph[i].next) if (j >= 0) sum += current[j];
            next[i] = sum;
        }
        current.swap(next);
        if (current[0] != counts[n]) throw std::runtime_error("unweighted count mismatch");
    }
    std::cout << "unweighted counts 0..17 verified, c17=" << current[0] << std::endl;
}

void row_certificate(const std::vector<Integer>& image, const std::vector<Integer>& h, int iteration) {
    std::size_t worst = 0;
    for (std::size_t i = 0; i < h.size(); ++i) {
        if (!h[i] || h[i] > scale+16) throw std::runtime_error("invalid positive vector");
        if (static_cast<Wide>(image[i])*h[worst] > static_cast<Wide>(image[worst])*h[i])
            worst = i;
    }
    const Integer numerator = image[worst], denominator = weight_den*h[worst];
    const Integer divisor = std::gcd(numerator, denominator);
    std::cout << "iteration=" << iteration << " checked_rows=" << h.size() << " worst=" << worst
              << " row_upper=" << numerator/divisor << "/" << denominator/divisor << std::endl;
}

int main(int argc, char** argv) {
    const int iterations = argc == 2 ? std::stoi(argv[1]) : 90;
    if (argc > 2 || iterations < 1 || iterations > 90) throw std::invalid_argument("iterations 1..90");
    const Inputs inputs{16, weight_num, weight_den, 12994574, 53221763};
    std::cout << "building one memory-16 graph" << std::endl;
    auto graph = build(inputs);
    std::size_t edges = 0;
    for (const Row& row : graph)
        edges += std::count_if(row.next.begin(), row.next.end(), [](int j) { return j >= 0; });
    if (graph.size() != inputs.states || edges != inputs.edges) throw std::runtime_error("graph size");
    std::cout << "states=" << graph.size() << " labelled_transitions=" << edges
              << " straight_weight=" << weight_num << "/" << weight_den << std::endl;
    check_straight_polynomials(graph);
    check_published_short_counts(graph);
    std::vector<Integer> h(graph.size(), scale);
    for (int i = 0; i <= iterations; ++i) {
        auto image = weighted_image(graph, h, inputs);
        if (i == 12 || i == 24 || i == 40 || i == 60 || i == iterations)
            row_certificate(image, h, i);
        if (i == iterations) break;
        const Integer largest = *std::max_element(image.begin(), image.end());
        if (!largest) throw std::runtime_error("empty transfer");
        for (std::size_t j = 0; j < h.size(); ++j)
            h[j] = static_cast<Wide>(image[j])*scale/largest+16;
    }
    return 0;
}
