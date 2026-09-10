// Offline coupled pressure replay from captured ORIGINAL Aphros face expressions.
// Does not advance or replace an Aphros flow state. Geometry and coefficients
// enter as original doubles; only pressure and face arithmetic are extended.
#define AMGCL_NO_BOOST
#include <amgcl/adapter/crs_tuple.hpp>
#include <amgcl/amg.hpp>
#include <amgcl/coarsening/smoothed_aggregation.hpp>
#include <amgcl/relaxation/spai0.hpp>
#include <amgcl/solver/cg.hpp>
#include <amgcl/make_solver.hpp>
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <vector>

template<class T> T read(std::ifstream& in) {
    T v;in.read(reinterpret_cast<char*>(&v),sizeof(v));
    if(!in)throw std::runtime_error("Truncated pressure replay input");return v;
}
struct Face {int a,b;double e0,e1,constant;};
int main(int argc,char** argv) {
    try {
        if(argc!=3)throw std::runtime_error("Usage: pressure_probe input.bin output-prefix");
        if(std::numeric_limits<long double>::digits<=53)throw std::runtime_error("Extended arithmetic unavailable");
        std::ifstream input(argv[1],std::ios::binary);
        const int64_t n=read<int64_t>(input),nf=read<int64_t>(input);const double rate=read<double>(input);
        if(n<2||n>5000000||nf<1||nf>30000000||!(rate>0))throw std::runtime_error("Invalid graph size or scale");
        std::vector<long double> pressure(n),original(n),volume(n),net(n);
        for(int c=0;c<n;++c) {pressure[c]=original[c]=read<double>(input);volume[c]=read<double>(input);}
        std::vector<Face> faces(nf);std::vector<std::map<int,double>> rows(n);
        for(auto& f:faces) {
            f.a=read<int32_t>(input);f.b=read<int32_t>(input);
            f.e0=read<double>(input);f.e1=read<double>(input);f.constant=read<double>(input);
            if(f.a<0||f.a>=n||f.b<0||f.b>=n||f.a==f.b||!(f.e0>0)||f.e1!=-f.e0)
                throw std::runtime_error("Invalid symmetric captured face");
            rows[f.a][f.a]+=f.e0;rows[f.a][f.b]+=f.e1;
            rows[f.b][f.a]-=f.e0;rows[f.b][f.b]-=f.e1;
        }
        if(input.peek()!=std::ifstream::traits_type::eof())throw std::runtime_error("Unexpected trailing bytes");
        int gauge=0;for(int c=0;c<n;++c)if(rows[c][c]>rows[gauge][gauge])gauge=c;
        // A double AMG solve only proposes corrections. Acceptance below always
        // re-evaluates the original full face expressions in extended precision.
        std::vector<int> ptr(1,0),col;std::vector<double> val;
        for(int c=0;c<n;++c) {
            if(c==gauge) {col.push_back(c);val.push_back(1.);}
            else for(const auto& e:rows[c])if(e.first!=gauge) {col.push_back(e.first);val.push_back(e.second);}
            ptr.push_back(int(col.size()));
        }
        rows.clear();rows.shrink_to_fit();
        using Backend=amgcl::backend::builtin<double>;
        using Solver=amgcl::make_solver<amgcl::amg<Backend,amgcl::coarsening::smoothed_aggregation,
                  amgcl::relaxation::spai0>,amgcl::solver::cg<Backend>>;
        Solver::params prm;prm.solver.tol=1e-14;prm.solver.maxiter=1000;
        Solver solve(std::tie(n,ptr,col,val),prm);
        std::ofstream history(std::string(argv[2])+"_history.csv");
        history<<std::setprecision(21)<<"refinement,relative_divergence_linf,global_net,correction_iterations,correction_relative_residual\n";
        bool passed=false;int iterations=0;double linearResidual=0;long double maximum=0;
        for(int pass=0;pass<=20;++pass) {
            std::fill(net.begin(),net.end(),0.L);
            for(const auto& f:faces) {
                // Preserve the original expression and grouping; no pressure-
                // difference substitution and no post hoc flux redistribution.
                const long double q=(pressure[f.a]*f.e0+pressure[f.b]*f.e1)+f.constant;
                net[f.a]+=q;net[f.b]-=q;
            }
            maximum=0;long double total=0;
            for(int c=0;c<n;++c) {maximum=std::max(maximum,std::abs(net[c])/volume[c]/rate);total+=net[c];}
            history<<pass<<','<<maximum<<','<<total<<','<<iterations<<','<<linearResidual<<'\n';history.flush();
            std::cout<<"refinement="<<pass<<" relative_divergence="<<std::setprecision(18)<<maximum<<std::endl;
            if(maximum<1e-7L) {passed=true;break;}
            if(pass==20)break;
            double scale=0;for(auto v:net)scale=std::max(scale,double(std::abs(v)));
            std::vector<double> rhs(n),delta(n,0.);
            for(int c=0;c<n;++c)rhs[c]=-double(net[c])/scale;rhs[gauge]=0;
            std::tie(iterations,linearResidual)=solve(rhs,delta);
            if(!std::isfinite(linearResidual))throw std::runtime_error("Nonfinite pressure correction solve");
            for(int c=0;c<n;++c)pressure[c]+=static_cast<long double>(delta[c])*scale;
        }
        std::ofstream field(std::string(argv[2])+"_pressure.csv");
        field<<"id,pressure_hex,net_flux,relative_divergence\n";
        for(int c=0;c<n;++c)field<<c<<','<<std::hexfloat<<pressure[c]<<std::defaultfloat<<std::setprecision(21)
            <<','<<net[c]<<','<<std::abs(net[c])/volume[c]/rate<<'\n';
        if(!history||!field)throw std::runtime_error("Pressure probe output failed");
        std::ofstream report(std::string(argv[2])+"_report.json");
        report<<std::setprecision(21)<<"{\"passed\":"<<(passed?"true":"false")<<",\"cells\":"<<n
            <<",\"faces\":"<<nf<<",\"mantissa_bits\":"<<std::numeric_limits<long double>::digits
            <<",\"relative_divergence_linf\":"<<maximum
            <<",\"scope\":\"Offline coupled original face equations with extended pressure; not a new Aphros flow trajectory\"}\n";
        return passed?0:1;
    } catch(const std::exception& e) {std::cerr<<e.what()<<std::endl;return 2;}
}
