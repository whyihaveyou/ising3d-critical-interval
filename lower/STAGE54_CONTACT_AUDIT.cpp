// Reuse the coordinate suffix graph. Generate a positive vector, then audit
// each possible number of actual children using unrounded rational powers.
#define main archived_saw_main
#include "verify_saw16_direct_straight_20261121.cpp"
#undef main

int main(int argc, char** argv) {
    if (argc != 2) throw std::invalid_argument("choose bulk, sparse or dense");
    std::string which = argv[1];
    const Integer unit = 100000000, level = Integer{1} << 30;
    Integer num, den, ceiling;
    int iterations;
    if (which == "bulk") {
        num=10000; den=9991; ceiling=473271745; iterations=40;
    } else if (which == "sparse") {
        num=4; den=5; ceiling=456604381; iterations=32;
    } else if (which == "dense") {
        num=6; den=5; ceiling=492941352; iterations=32;
    } else throw std::invalid_argument("invalid audit case");

    const auto graph = build({16, 1, 1, 12994574, 53221763});
    std::vector<Integer> v(graph.size(), level), next(graph.size());
    std::array<Integer, 6> rounded{unit};
    std::array<Wide, 6> np{1}, dp{1};
    for (int a=1; a<=5; ++a) {
        rounded[a]=(rounded[a-1]*num+den-1)/den;
        np[a]=np[a-1]*num;
        dp[a]=dp[a-1]*den;
    }
    std::uint64_t final_comparisons=0, labels=0;
    for (int it=0; it<=iterations; ++it) {
        Integer peak=0;
        for (std::size_t s=0; s<graph.size(); ++s) {
            std::array<Integer, 6> values{};
            int m=0;
            Integer sum=0;
            for (int dest:graph[s].next) {
                if (dest < 0) continue;
                values[m++]=v[dest];
                sum+=v[dest];
            }
            const bool root=graph[s].previous_direction < 0;
            if ((root && (s != 0 || m != 6)) || (!root && m>5))
                throw std::runtime_error("inconsistent degree");
            if (it == iterations) {
                labels+=m;
                std::sort(values.begin(), values.begin()+m, std::greater<Integer>());
                Integer largest_sum=0;
                for (int k=1; k<=m; ++k) {
                    largest_sum+=values[k-1];
                    if (root && k<m) continue;
                    int a=root ? 0 : 5-k;
                    if (np[a]*unit*largest_sum > dp[a]*ceiling*v[s])
                        throw std::runtime_error("exact row failed: "+std::to_string(s));
                    ++final_comparisons;
                }
                continue;
            }
            // This pass only proposes v. Rounding directions here do not
            // enter the exact final inequality.
            Integer image=root ? unit*sum : rounded[5-m]*sum;
            if (!root && num>den && m>1) {
                auto smallest=*std::min_element(values.begin(),values.begin()+m);
                if (smallest*num<sum*(num-den)) {
                    std::sort(values.begin(),values.begin()+m);
                    for (int j=0; j<m-1; ++j) {
                        sum-=values[j];
                        image=std::max(image, rounded[6-m+j]*sum);
                    }
                }
            }
            next[s]=image;
            peak=std::max(image, peak);
        }
        if (it==iterations) break;
        unsigned shift=0;
        while ((peak>>shift)>2*level) ++shift;
        for (auto& y:next) y=(y>>shift)+1;
        v.swap(next);
    }
    if (graph.size()!=12994574 || labels!=53221763 || final_comparisons!=labels-5)
        throw std::runtime_error("incomplete audit");
    std::cout<<which<<" exact rational weight "<<num<<"/"<<den
             <<" states "<<graph.size()<<" labels "<<labels
             <<" inequalities "<<final_comparisons<<" ceiling "<<ceiling<<"/"<<unit
             <<" maximum vector component "<<*std::max_element(v.begin(),v.end())<<"\n";
}
