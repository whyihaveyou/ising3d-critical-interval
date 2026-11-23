// Exact tensor contraction for one source class and one output prefix at a time.
#include <gmpxx.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <sys/resource.h>

using Z = mpz_class;

void require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
}

struct Ratio { Z num, den; };
struct Cut { Z num, den; };
struct Source {
    int id;
    Cut upper, low;
    std::vector<int> high;
};
struct Model {
    int nbits, corners, orbits;
    Z scale, trace[4];
    std::vector<Ratio> ratios;
    std::vector<unsigned> masks;
    std::vector<Z> weights, values;
    std::vector<unsigned> owner;
    std::vector<Source> sources;
};

Model read_model(const char* data, const char* owners) {
    std::ifstream in(data);
    std::string tag;
    Model m;
    int nclasses;
    require(bool(in >> tag >> m.nbits >> m.corners >> m.orbits >> nclasses >> m.scale)
            && tag == "STAGE34v1", "bad model header");
    require(m.nbits >= 3 && m.nbits <= 18 && m.corners >= 1 && m.corners <= 8
            && m.orbits >= 1 && m.orbits <= 65536 && nclasses >= 1 && nclasses <= 256
            && m.scale > 0, "invalid model dimensions or scale");
    const int n = 1 << m.nbits;
    m.ratios.resize(m.nbits);
    for (auto& r : m.ratios)
        require(bool(in >> r.num >> r.den) && r.num > 0 && r.num <= r.den, "bad ratio");
    for (Z& t : m.trace) require(bool(in >> t) && t > 0, "bad trace");
    m.masks.resize(m.corners);
    for (auto& mask : m.masks)
        require(bool(in >> mask) && mask < unsigned(n) && __builtin_popcount(mask) == 3,
                "bad corner mask");
    m.weights.resize(m.orbits);
    m.values.resize(m.orbits);
    for (int i = 0; i < m.orbits; ++i)
        require(bool(in >> m.weights[i] >> m.values[i]) && m.weights[i] > 0,
                "bad orbit row");
    std::vector<bool> seen(1 << m.corners, false);
    for (int i = 0; i < nclasses; ++i) {
        Source c;
        int count;
        require(bool(in >> c.id >> c.upper.num >> c.upper.den >> c.low.num >> c.low.den >> count)
                && c.id >= 0 && c.id < int(seen.size()) && !seen[c.id]
                && c.upper.den > 0 && c.low.den > 0 && count >= 0 && count < n,
                "bad class bound");
        require(c.low.num*c.upper.den <= c.upper.num*c.low.den, "reversed bounds");
        seen[c.id] = true;
        c.high.resize(count);
        std::vector<bool> used(n, false);
        for (int& f : c.high) {
            require(bool(in >> f) && f >= 0 && f < n && !used[f], "invalid high word");
            used[f] = true;
        }
        m.sources.push_back(std::move(c));
    }
    require(!(in >> tag), "trailing model data");
    std::ifstream bits(owners, std::ios::binary);
    m.owner.resize(n);
    for (auto& id : m.owner) {
        const int lo = bits.get(), hi = bits.get();
        require(lo >= 0 && hi >= 0, "truncated owner table");
        id = unsigned(lo + 256*hi);
        require(id < unsigned(m.orbits), "owner outside orbit array");
    }
    require(bits.get() == std::ifstream::traits_type::eof(), "trailing owner data");
    return m;
}

struct Geometry {
    std::vector<unsigned> histogram;
    std::vector<std::vector<Z>> factors;
};

Geometry make_geometry(const Model& m, const Source& c, int prefix_bits) {
    const int radix = m.corners + 1;
    const unsigned step[] = {1u, unsigned(radix), unsigned(radix*radix), 0u};
    const int prefixes = 1 << prefix_bits;
    Geometry g;
    g.histogram.resize(m.owner.size());
    for (unsigned b = 0; b < m.owner.size(); ++b) {
        unsigned code = 0;
        for (int j = 0; j < m.corners; ++j) {
            int count = __builtin_popcount(b & m.masks[j]);
            if ((c.id >> j) & 1) count = 3-count;
            code += step[count];
        }
        g.histogram[b] = code;
    }
    std::vector<Z> prefix_factor(prefixes, Z(1));
    for (int x = 0; x < prefixes; ++x)
        for (int j = 0; j < prefix_bits; ++j)
            prefix_factor[x] *= ((x >> j) & 1) ? m.ratios[j].num : m.ratios[j].den;
    g.factors.resize(radix*radix*radix);
    for (int a = 0; a <= m.corners; ++a) {
        for (int b = 0; a+b <= m.corners; ++b) {
            for (int d = 0; a+b+d <= m.corners; ++d) {
                const int counts[] = {a, b, d, m.corners-a-b-d};
                Z factor = 1;
                for (int j = 0; j < 4; ++j)
                    for (int t = 0; t < counts[j]; ++t) factor *= m.trace[j];
                auto& row = g.factors[a + radix*b + radix*radix*d];
                for (const auto& k : prefix_factor) row.emplace_back(factor*k);
            }
        }
    }
    return g;
}

void transform_suffix(std::vector<Z>& v, const Model& m, int first_bit) {
    Z a, b;
    for (int j = first_bit; j < m.nbits; ++j) {
        const int stride = 1 << (j-first_bit);
        const auto& r = m.ratios[j];
        for (int start = 0; start < int(v.size()); start += 2*stride) {
            for (int i = start; i < start+stride; ++i) {
                auto left = v[i].get_mpz_t();
                auto right = v[i+stride].get_mpz_t();
                mpz_mul(a.get_mpz_t(), r.den.get_mpz_t(), left);
                mpz_addmul(a.get_mpz_t(), r.num.get_mpz_t(), right);
                mpz_mul(b.get_mpz_t(), r.den.get_mpz_t(), right);
                mpz_addmul(b.get_mpz_t(), r.num.get_mpz_t(), left);
                mpz_swap(left, a.get_mpz_t());
                mpz_swap(right, b.get_mpz_t());
            }
        }
    }
}

void verify_cut(const Model& m, const Source& c, const Geometry& g,
                int prefix_bits, bool low, bool dump, bool check) {
    const auto& cut = low ? c.low : c.upper;
    const std::string name = low ? "low" : "upper";
    const int n = int(m.owner.size()), prefixes = 1 << prefix_bits;
    const int suffixes = n/prefixes;
    std::vector<Z> signed_weights(m.orbits);
    for (int i = 0; i < m.orbits; ++i)
        signed_weights[i] = m.weights[i]*(cut.num*m.scale-cut.den*m.values[i]);
    std::vector<bool> high(n, false);
    for (int f : c.high) high[f] = true;
    int outside_sample = (c.id*2053+n/7) % n;
    while (high[outside_sample]) outside_sample = (outside_sample+1) % n;
    int inside_sample = c.high.empty() ? 0 : c.high.front();
    std::vector<Z> slab(suffixes);
    int checked = 0, skipped = 0, negatives_skipped = 0, min_field = -1;
    Z minimum;
    for (int p = 0; p < prefixes; ++p) {
        for (int s = 0; s < suffixes; ++s) {
            auto value = slab[s].get_mpz_t();
            mpz_set_ui(value, 0);
            for (int x = 0; x < prefixes; ++x) {
                const int b = (s << prefix_bits) | x;
                const auto& coefficient = g.factors[g.histogram[b]][p ^ x];
                mpz_addmul(value, signed_weights[m.owner[b]].get_mpz_t(), coefficient.get_mpz_t());
            }
        }
        transform_suffix(slab, m, prefix_bits);
        for (int s = 0; s < suffixes; ++s) {
            const int f = (s << prefix_bits) | p;
            const auto& z = slab[s];
            if (dump || f == inside_sample || f == outside_sample)
                std::cout << "value " << name << " " << f << " " << z << "\n";
            if (low && high[f]) {
                ++skipped;
                if (z < 0) ++negatives_skipped;
            } else {
                ++checked;
                if (check && z < 0) throw std::runtime_error("negative " + name +
                    " numerator at class " + std::to_string(c.id) + " field " + std::to_string(f));
                if (min_field < 0 || z < minimum) { min_field = f; minimum = z; }
            }
        }
        std::cout << "prefix " << name << " " << p << " " << suffixes << std::endl;
    }
    require(checked+skipped == n && skipped == (low ? int(c.high.size()) : 0),
            "incomplete bound coverage");
    std::cout << "bound " << name << " " << checked << " " << skipped << " "
              << negatives_skipped << " " << min_field << " " << sgn(minimum) << std::endl;
}

int main(int argc, char** argv) {
    try {
        require(argc == 5 || argc == 6, "usage: kernel MODEL OWNERS CLASS PREFIX_BITS [test-dump]");
        const bool dump = argc == 6;
        require(!dump || std::string(argv[5]) == "test-dump", "unknown mode");
        auto start = std::chrono::steady_clock::now();
        const Model m = read_model(argv[1], argv[2]);
        const int c = std::stoi(argv[3]), k = std::stoi(argv[4]);
        require(0 <= k && k <= m.nbits && k <= 6, "unsupported block size");
        require(!dump || m.nbits <= 8, "test dumps only support small tensors");
        auto source = std::find_if(m.sources.begin(), m.sources.end(), [=](const Source& s){return s.id == c;});
        require(source != m.sources.end(), "unknown class");
        const auto geometry = make_geometry(m, *source, k);
        verify_cut(m, *source, geometry, k, false, dump, !dump);
        verify_cut(m, *source, geometry, k, true, dump, !dump);
        struct rusage usage{};
        require(getrusage(RUSAGE_SELF, &usage) == 0, "missing resource usage");
        long long bytes = usage.ru_maxrss;
#ifndef __APPLE__
        bytes *= 1024;
#endif
        std::chrono::duration<double> elapsed = std::chrono::steady_clock::now()-start;
        std::cout << (dump ? "DUMP" : "PASS") << " class " << c << " seconds " << elapsed.count()
                  << " peak_rss_bytes " << bytes << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "FAIL " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
