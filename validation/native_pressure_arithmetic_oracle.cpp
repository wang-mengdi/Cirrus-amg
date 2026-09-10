// Offline arithmetic oracle for actual device pressure face weights/stencils.
// Compile without fast-math using a compiler with >53 long-double mantissa bits.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

template<class T>T read(std::istream& in) {
    T v;in.read(reinterpret_cast<char*>(&v),sizeof(v));
    if(!in)throw std::runtime_error("Truncated binary input");return v;
}
std::vector<double> vectorFile(const std::filesystem::path& p,size_t n) {
    if(std::filesystem::file_size(p)!=n*sizeof(double))throw std::runtime_error("Wrong vector length");
    std::vector<double> v(n);std::ifstream in(p,std::ios::binary);
    in.read(reinterpret_cast<char*>(v.data()),std::streamsize(n*sizeof(double)));
    if(!in||!std::all_of(v.begin(),v.end(),[](double x){return std::isfinite(x);}))
        throw std::runtime_error("Invalid vector input");return v;
}
template<class T>void add(T& sum,T& error,T value) {
    const T next=sum+value;
    error+=std::abs(sum)>=std::abs(value)?(sum-next)+value:(value-next)+sum;
    sum=next;
}
struct Face {std::int32_t owner,neighbor,begin,end;};
struct Entry {std::int32_t cell;double weight;};
int main(int argc,char** argv) {
    try {
        if(argc!=8&&argc!=9)throw std::runtime_error("Usage: oracle operator.bin rhs.bin iterate.bin ax.bin scale output-directory label [iterate_low.bin]");
        if(std::numeric_limits<long double>::digits<64)throw std::runtime_error("Oracle requires at least 64 mantissa bits");
        const std::uint16_t endian=1;
        if(*reinterpret_cast<const unsigned char*>(&endian)!=1||sizeof(double)!=8)
            throw std::runtime_error("Oracle input is little-endian float64");
        const auto out=std::filesystem::path(argv[6]);
        if(std::filesystem::exists(out))throw std::runtime_error("Preserve previous oracle output");
        std::ifstream in(argv[1],std::ios::binary);char magic[8];in.read(magic,8);
        if(std::string(magic,8)!="CIRRUSP1")throw std::runtime_error("Unknown face export format");
        const auto n=read<std::uint64_t>(in),regular=read<std::uint64_t>(in),cf=read<std::uint64_t>(in),entries=read<std::uint64_t>(in);
        if(!n||n>100000000||regular>10*n||cf>10*n||entries>1000*n)throw std::runtime_error("Invalid operator dimensions");
        auto rhs=vectorFile(argv[2],n),x=vectorFile(argv[3],n),gpu=vectorFile(argv[4],n);
        const auto low=argc==9?vectorFile(argv[8],n):std::vector<double>(n,0.);
        const double scale=std::stod(argv[5]);if(!(scale>0&&std::isfinite(scale)))throw std::runtime_error("Invalid RHS scale");
        for(double& b:rhs)b/=scale; // Exactly the normalized double RHS used on device.
        std::vector<long double> accurate(n,0),accurateError(n,0);
        std::vector<double> naive(n,0),compensated(n,0),compensatedError(n,0);
        std::vector<unsigned char> interfaceCell(n,0);
        auto checkCell=[&](int c){if(c<0||std::uint64_t(c)>=n)throw std::runtime_error("Invalid cell index");};
        for(std::uint64_t f=0;f<regular;++f) {
            const int p=read<std::int32_t>(in),q=read<std::int32_t>(in);const double w=read<double>(in);
            checkCell(p);checkCell(q);if(!(w>0&&std::isfinite(w)))throw std::runtime_error("Invalid regular coefficient");
            const long double exact=static_cast<long double>(w)*((static_cast<long double>(x[p])-x[q])+(static_cast<long double>(low[p])-low[q]));
            add(accurate[p],accurateError[p],exact);add(accurate[q],accurateError[q],-exact);
            const double value=w*((x[p]-x[q])+(low[p]-low[q]));naive[p]+=value;naive[q]-=value;
            add(compensated[p],compensatedError[p],value);add(compensated[q],compensatedError[q],-value);
        }
        std::vector<Face> faces(cf);
        for(auto& f:faces) {
            f={read<std::int32_t>(in),read<std::int32_t>(in),read<std::int32_t>(in),read<std::int32_t>(in)};
            checkCell(f.owner);checkCell(f.neighbor);
            if(f.begin<0||f.end<=f.begin||std::uint64_t(f.end)>entries)throw std::runtime_error("Invalid stencil range");
            interfaceCell[f.owner]=interfaceCell[f.neighbor]=1;
        }
        std::vector<Entry> terms(entries);
        for(auto& e:terms) {e.cell=read<std::int32_t>(in);e.weight=read<double>(in);checkCell(e.cell);if(!std::isfinite(e.weight))throw std::runtime_error("Invalid stencil weight");}
        if(in.peek()!=std::char_traits<char>::eof())throw std::runtime_error("Unexpected trailing operator data");
        for(const auto& f:faces) {
            long double value=0,error=0;double plain=0,sum=0,comp=0;
            for(int j=f.begin;j<f.end;++j) {
                const auto e=terms[j];
                add(value,error,static_cast<long double>(e.weight)*((static_cast<long double>(x[e.cell])-x[f.owner])+(static_cast<long double>(low[e.cell])-low[f.owner])));
                const double term=e.weight*((x[e.cell]-x[f.owner])+(low[e.cell]-low[f.owner]));plain+=term;add(sum,comp,term);
            }
            value+=error;sum+=comp;
            add(accurate[f.owner],accurateError[f.owner],value);add(accurate[f.neighbor],accurateError[f.neighbor],-value);
            naive[f.owner]+=plain;naive[f.neighbor]-=plain;
            add(compensated[f.owner],compensatedError[f.owner],sum);add(compensated[f.neighbor],compensatedError[f.neighbor],-sum);
        }
        long double b2=0,rg2=0,ra2=0,rn2=0,rc2=0,round2=0,roundCf2=0,roundReg2=0;
        std::filesystem::create_directories(out);
        std::ofstream ax(out/"accurate_ax_hi_lo.bin",std::ios::binary),residual(out/"accurate_residual.bin",std::ios::binary);
        std::vector<std::size_t> order(n);std::iota(order.begin(),order.end(),0);
        for(size_t c=0;c<n;++c) {
            accurate[c]+=accurateError[c];compensated[c]+=compensatedError[c];
            const long double b=rhs[c],rg=b-gpu[c],ra=b-accurate[c],rn=b-naive[c],rc=b-compensated[c],round=gpu[c]-accurate[c];
            b2+=b*b;rg2+=rg*rg;ra2+=ra*ra;rn2+=rn*rn;rc2+=rc*rc;round2+=round*round;
            (interfaceCell[c]?roundCf2:roundReg2)+=round*round;
            const double hi=double(accurate[c]),lo=double(accurate[c]-hi),r=double(ra);
            ax.write(reinterpret_cast<const char*>(&hi),8);ax.write(reinterpret_cast<const char*>(&lo),8);
            residual.write(reinterpret_cast<const char*>(&r),8);
        }
        if(!(b2>0))throw std::runtime_error("Zero RHS");
        ax.flush();residual.flush();if(!ax.good()||!residual.good())throw std::runtime_error("Oracle binary write failed");
        std::partial_sort(order.begin(),order.begin()+std::min<size_t>(64,n),order.end(),[&](size_t a,size_t b){return std::abs(gpu[a]-accurate[a])>std::abs(gpu[b]-accurate[b]);});
        std::ofstream worst(out/"worst_roundoff.csv");worst<<std::setprecision(21)<<"id,interface_cell,rhs,iterate,gpu_ax,accurate_ax,gpu_roundoff,accurate_residual\n";
        for(size_t j=0;j<std::min<size_t>(64,n);++j) {const auto c=order[j];worst<<c<<','<<int(interfaceCell[c])<<','<<rhs[c]<<','<<x[c]<<','<<gpu[c]<<','<<accurate[c]<<','<<gpu[c]-accurate[c]<<','<<rhs[c]-accurate[c]<<'\n';}
        const auto relative=[&](long double value){return std::sqrt(value/b2);};
        std::ofstream report(out/"result.json");report<<std::setprecision(21)
            <<"{\n  \"scope\": \"Offline higher-precision arithmetic evaluation of actual device face coefficients, not a flow solve\",\n"
            <<"  \"label\": \""<<argv[7]<<"\",\n  \"long_double_mantissa_bits\": "<<std::numeric_limits<long double>::digits
            <<",\n  \"twofold_input\": "<<(argc==9?"true":"false")<<",\n  \"cells\": "<<n<<",\n  \"regular_faces\": "<<regular<<",\n  \"interface_faces\": "<<cf<<",\n  \"stencil_entries\": "<<entries
            <<",\n  \"gpu_residual\": "<<relative(rg2)<<",\n  \"accurate_residual\": "<<relative(ra2)
            <<",\n  \"naive_double_residual\": "<<relative(rn2)<<",\n  \"compensated_double_residual\": "<<relative(rc2)
            <<",\n  \"gpu_ax_roundoff_over_rhs\": "<<relative(round2)<<",\n  \"gpu_interface_roundoff_over_rhs\": "<<relative(roundCf2)
            <<",\n  \"gpu_regular_roundoff_over_rhs\": "<<relative(roundReg2)<<",\n  \"rhs_scale\": "<<scale<<"\n}\n";
        report.flush();worst.flush();if(!report.good()||!worst.good())throw std::runtime_error("Oracle text write failed");
        std::cout<<"GPU residual "<<relative(rg2)<<", higher-precision residual "<<relative(ra2)<<", compensated-double residual "<<relative(rc2)<<'\n';
    } catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
