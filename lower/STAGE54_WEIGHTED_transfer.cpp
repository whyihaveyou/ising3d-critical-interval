// Integer comparison for the physically marked three-bond path weights.
// Reuse the archived state construction. Explicitly recover its two last
// directions in its own canonical frame, then check physical short walks.
#define main archived_straight_main
#include "verify_saw16_direct_straight_20261121.cpp"
#undef main

#include <functional>

using Weights = std::array<Integer, 6>;
constexpr Integer denominator = 1000000;
constexpr Weights lower_weights{988855, 990091, 990720, 991321, 991419, denominator};
constexpr Weights upper_weights{988928, 990150, 990771, 991368, 991464, denominator};
constexpr Integer magnitude = Integer{1} << 35;
static_assert(6*denominator*(magnitude+1) < UINT64_MAX);

int local_motif(int first, int middle, int last) {
    if (first < 0 || middle < 0) return 5;
    const bool first_straight = first == middle;
    const bool last_straight = last == middle;
    if (first_straight && last_straight) return 0;
    if (first_straight || last_straight) return 1;
    if ((first ^ 1) == last) return 2;
    if (first == last) return 4;
    return 3;
}

Inputs geometry(int memory) {
    if (memory == 8) return {8,1,1,237,0};
    if (memory == 12) return {12,1,1,46562,192576};
    if (memory == 14) return {14,1,1,759774,3122796};
    if (memory == 16) return {16,1,1,12994574,53221763};
    throw std::invalid_argument("memory must be 8, 12, 14 or 16");
}

void mark_rows(std::vector<Row>& graph, int memory) {
    // build() keeps one canonical trimmed word per state but returns only its
    // destinations. Replay the first discovery of each word on the same graph.
    std::vector<Integer> keys(graph.size());
    keys[0] = 1;
    for (size_t i=0;i<graph.size();++i) {
        if (!keys[i]) throw std::runtime_error("unreachable state");
        int steps[maximum_memory+1], n=0;
        for (auto code=keys[i];code>1;code>>=3)
            steps[n++]=code&7;
        std::reverse(steps,steps+n);
        std::array<Point,maximum_memory+1> vertices{};
        for (int k=0;k<n;++k) vertices[k+1]=moved(vertices[k],steps[k]);
        const int last = n ? steps[n-1] : -1;
        const int penultimate = n>1 ? steps[n-2] : -1;
        if (last != graph[i].previous_direction)
            throw std::runtime_error("canonical reconstruction disagrees");
        int classes = 0;
        for (int d=0;d<6;++d) {
            classes |= local_motif(penultimate,last,d) << (3*d);
            const int child = graph[i].next[d];
            const Point target = moved(vertices[n],d);
            const bool occupied=std::find(vertices.begin(),vertices.begin()+n+1,target)!=vertices.begin()+n+1;
            if ((child<0)!=occupied) throw std::runtime_error("physical suffix rejection mismatch");
            if (child<0) continue;
            int cut=0;
            while (cut<=n) {
                int distance=0;
                for (int xyz=0;xyz<3;++xyz)
                    distance += std::abs(target[xyz]-vertices[cut][xyz]);
                if (n+1-cut+distance<=memory) break;
                ++cut;
            }
            if (cut>n) throw std::runtime_error("missing predecessor");
            steps[n]=d;
            const auto expected=canonical(steps,cut,n+1);
            if (keys[child] && keys[child]!=expected)
                throw std::runtime_error("canonical key or transition was inconsistent");
            keys[child]=expected;
        }
        graph[i].previous_direction=classes;
    }
}

void check_words(const std::vector<Row>& graph, int maximum_length) {
    // The first memory/2 steps always retain the entire prefix. Walk in
    // physical axes, mapping each outgoing direction into the state's frame.
    std::vector<Point> walk(1,Point{0,0,0});
    std::vector<int> word;
    std::array<unsigned long long, 10> counts{};
    std::function<void(int)> visit = [&](int row) {
        const size_t n=word.size();
        ++counts[n];
        if (n==static_cast<size_t>(maximum_length)) return;
        int axes[3]={-1,-1,-1}, signs[3]={}, used=0;
        for(int step:word) {
            const int axis=step/2;
            if(axes[axis]<0) {
                axes[axis]=used++;
                signs[axis]=step%2;
            }
        }
        for (int d=0;d<6;++d) {
            const auto position=moved(walk.back(),d);
            if (std::find(walk.begin(),walk.end(),position)!=walk.end()) continue;
            const int axis=d/2;
            const int label=axes[axis]<0 ? 2*used : 2*axes[axis]+(signs[axis]^(d%2));
            const int first=n>1?word[n-2]:-1, middle=n?word.back():-1;
            const int mark=(graph[row].previous_direction>>(3*label))&7;
            const int child=graph[row].next[label];
            if (child<0 || mark!=local_motif(first,middle,d))
                throw std::runtime_error("direct walk or motif lost");
            word.push_back(d);
            walk.push_back(position);
            visit(child);
            walk.pop_back();
            word.pop_back();
        }
    };
    visit(0);
    const std::array<unsigned long long,9> expected={1,6,30,150,726,3534,16926,81390,387966};
    for (int j=0;j<=maximum_length;++j)
        if(counts[j]!=expected[j]) throw std::runtime_error("short count mismatch");
    std::cout << "direct physical path/motif check through " << maximum_length
              << " count " << counts[maximum_length] << std::endl;
}

struct Bound {
    Integer lower,upper;
};

Bound check_matrix(const std::vector<Row>& graph, const Weights& weights, int iterations) {
    std::vector<Integer> h(graph.size(),magnitude), later(graph.size());
    Bound result{};
    for(int k=0;k<=iterations;++k) {
        Integer max_image=0;
        for(size_t i=0;i<graph.size();++i) {
            const auto& row=graph[i];
            Integer value=0;
            for(int d=0;d<6;++d) {
                const int j=row.next[d];
                if(j<0) continue;
                const int mark=(row.previous_direction>>(3*d))&7;
                value+=weights[mark]*h[j];
            }
            later[i]=value;
            max_image=std::max(max_image,value);
        }
        if(k==iterations) {
            // A strictly positive right vector provides the upper bound.
            Integer upper=0;
            for(size_t i=0;i<graph.size();++i)
                upper=std::max(upper,(later[i]+h[i]-1)/h[i]);
            // A clipped nonnegative vector suffices for a lower bound; even
            // dead states on the finite-memory graph cause no obstruction.
            constexpr Integer clip=4096;
            Integer lower=UINT64_MAX;
            for(size_t i=0;i<graph.size();++i) {
                if(h[i]<=clip) continue;
                Integer image=0;
                for(int d=0;d<6;++d) {
                    int j=graph[i].next[d];
                    if(j<0 || h[j]<=clip) continue;
                    int mark=(graph[i].previous_direction>>(3*d))&7;
                    image+=weights[mark]*(h[j]-clip);
                }
                lower=std::min(lower,image/(h[i]-clip));
            }
            if(lower==UINT64_MAX || !h[0] || h[0]<=clip)
                throw std::runtime_error("empty reachable lower certificate");
            result={lower,upper};
            break;
        }
        for(size_t i=0;i<h.size();++i)
            h[i]=1+Wide(later[i])*magnitude/max_image;
    }
    return result;
}

int main(int argc, char** argv) {
    const int memory=argc>=2?std::stoi(argv[1]):12;
    const int iterations=argc>=3?std::stoi(argv[2]):28;
    if(iterations<1 || iterations>80) throw std::invalid_argument("iterations 1..80");
    const auto input=geometry(memory);
    auto graph=build(input);
    size_t actual_edges=0;
    for (const auto& row:graph)
        actual_edges+=std::count_if(row.next.begin(),row.next.end(),[](int j){return j>=0;});
    if(graph.size()!=input.states || (input.edges && actual_edges!=input.edges))
        throw std::runtime_error("geometric counts");
    mark_rows(graph,memory);
    std::cout<<"memory "<<memory<<" rows "<<graph.size()<<" edges "<<actual_edges<<std::endl;
    check_words(graph,std::min(7,memory/2));
    if(memory==16) {
        const Weights balanced{998874,999457,999457,999906,1000000,denominator};
        const auto test=check_matrix(graph,balanced,iterations);
        std::cout<<"r=1927/1250 priced shape matrix upper in units 1e-6 "<<test.upper<<std::endl;
        if(test.upper>4730332) throw std::runtime_error("frozen physical upper not reached");
    }
    const auto optimistic=check_matrix(graph,lower_weights,iterations);
    const auto true_upper=check_matrix(graph,upper_weights,iterations);
    std::cout<<"Perron lower and upper, units 1e-6 "<<optimistic.lower<<" "<<true_upper.upper<<std::endl;
    const Wide lhs=Wide{109}*optimistic.lower;
    const Wide rhs=Wide{509}*denominator;
    if(lhs<=rhs) throw std::runtime_error("no optimistic-matrix obstruction at r=1.545");
    std::cout<<"u times optimistic spectral radius >1 by at least (units 1e-9) "
             <<static_cast<Integer>((lhs-rhs)*1000000000/rhs)<<std::endl;
}
