// Finite-memory walk bound with unobserved older contacts allowed to remove
// any subset of the locally admissible successors on the physical torus.
#define main archived_walk_main
#include "verify_saw16_direct_straight_20261121.cpp"
#undef main

constexpr Integer denominator54 = 100000000;
constexpr Integer level54 = Integer{1} << 30;
using ContactPowers = std::array<Integer, 6>;

ContactPowers contact_powers(Integer numerator, Integer denominator) {
    ContactPowers powers{denominator54};
    for (std::size_t i = 1; i < powers.size(); ++i)
        powers[i] = (powers[i-1]*numerator + denominator-1)/denominator;
    return powers;
}

// The output divided by denominator54 dominates the exact contact Bellman row.
Integer contact_row(const Row& row, const std::vector<Integer>& v,
                    const ContactPowers& powers, Integer wn, Integer wd) {
    std::array<Integer, 6> successor{};
    int m = 0;
    Integer sum = 0, smallest = UINT64_MAX;
    for (int j : row.next) {
        if (j < 0) continue;
        successor[m++] = v[j];
        sum += v[j];
        smallest = std::min(smallest, v[j]);
    }
    if (!m || row.previous_direction < 0) return denominator54*sum;
    if (m > 5) throw std::logic_error("noninitial state has six successors");
    Integer best = powers[5-m]*sum;
    if (wn <= wd) return best;
    // (1-j*(1-1/w))*w^j <= 1 for integer j>=0, by Bernoulli.
    if (smallest*wn >= sum*(wn-wd)) return best;
    std::sort(successor.begin(), successor.begin()+m);
    for (int j = 1; j < m; ++j) {
        sum -= successor[j-1];
        best = std::max(best, powers[5-m+j]*sum);
    }
    return best;
}

int main(int argc, char** argv) {
    const int memory = argc>1 ? std::stoi(argv[1]) : 16;
    const int iterations = argc>2 ? std::stoi(argv[2]) : 43;
    const Integer wn = argc>3 ? std::stoull(argv[3]) : 10000;
    const Integer wd = argc>4 ? std::stoull(argv[4]) : 9991;
    if (argc>5 || iterations<1 || !wd || !wn || wd>1000000 || wn>1000000 ||
        wn*4>wd*5) throw std::invalid_argument("memory iterations w_num w_den, 0 < w <= 5/4");
    const std::map<int, std::pair<std::size_t, std::size_t>> sizes = {
        {4,{5,25}}, {8,{237,1015}}, {12,{46562,192576}},
        {14,{759774,3122796}}, {16,{12994574,53221763}}
    };
    const auto [states, transitions] = sizes.at(memory);
    Inputs input{memory, 1, 1, states, transitions};
    const auto graph = build(input);
    std::size_t labels=0;
    for (const auto& row : graph)
        for (int j : row.next) labels += j>=0;
    if (graph.size()!=states || labels!=transitions) throw std::logic_error("wrong walk graph");
    const auto powers=contact_powers(wn,wd);
    std::cout<<"memory "<<memory<<" states "<<states<<" labels "<<labels
             <<" contact "<<wn<<"/"<<wd<<" upper-power-unit "<<denominator54<<std::endl;
    // Each component <= 2*level54+1, six summands, w^5 <= (5/4)^5.
    // Integer products for image, normalization and cutoff remain below UINT64_MAX.
    std::vector<Integer> h(states,level54), next(states);
    for (int i=0; i<=iterations; ++i) {
        Integer peak=0;
        std::size_t worst=0;
        for (std::size_t s=0; s<states; ++s) {
            next[s]=contact_row(graph[s],h,powers,wn,wd);
            peak=std::max(peak,next[s]);
            if (Wide(next[s])*h[worst]>Wide(next[worst])*h[s]) worst=s;
        }
        if (i==iterations || i==14 || i==20 || i==26 || i==32 || i==38) {
            Integer a=next[worst], b=h[worst]*denominator54;
            Integer common=std::gcd(a,b);
            std::cout<<"iteration "<<i<<" highest certified row "<<a/common<<"/"<<b/common
                     <<" ceil x1e8 "<<Integer((Wide(a)*100000000+b-1)/b)
                     <<" maximizer "<<worst<<std::endl;
        }
        unsigned shift=0;
        while ((peak>>shift)>2*level54) ++shift;
        for (auto& v : next) v=(v>>shift)+1;
        h.swap(next);
    }
}
