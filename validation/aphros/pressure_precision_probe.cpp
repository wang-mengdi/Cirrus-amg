// Local arithmetic feasibility probe from captured original face expressions.
// Neighbor pressures are frozen: this is NOT a coupled reference flow solve.
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

struct Face { int cell,sign; double volume,rate,e0,e1,b,p0,p1; };
int main(int argc,char** argv) {
  try {
    if(argc!=2)throw std::runtime_error("Expected captured-face CSV path");
    std::ifstream input(argv[1]);if(!input)throw std::runtime_error("Cannot open input");
    std::string line;std::getline(input,line);std::vector<std::vector<Face>> groups;
    if(line!="cell,sign,volume,rate,e0,e1,b,p0,p1")throw std::runtime_error("Unexpected input schema");
    while(std::getline(input,line)) {
      std::replace(line.begin(),line.end(),',',' ');std::istringstream stream(line);Face f{};
      if(!(stream>>f.cell>>f.sign>>f.volume>>f.rate>>f.e0>>f.e1>>f.b>>f.p0>>f.p1) ||
         f.cell<0 || (f.sign!=1&&f.sign!=-1) || !(f.volume>0&&f.rate>0))
        throw std::runtime_error("Invalid captured face");
      groups.resize(std::max(groups.size(),size_t(f.cell+1)));groups[f.cell].push_back(f);
    }
    std::cout<<std::setprecision(std::numeric_limits<long double>::max_digits10)
             <<"cell,double_digits,extended_digits,current_pressure_extended_eval,local_extended_target,local_extended_relative_divergence\n";
    for(size_t i=0;i<groups.size();++i) {
      const auto& fs=groups[i];if(fs.empty())throw std::runtime_error("Empty pressure probe");
      long double diagonal=0,constant=0,actual=0;
      for(const auto& f:fs) {
        const long double s=f.sign,a=f.sign==1?f.e0:f.e1,n=f.sign==1?f.e1:f.e0;
        const long double pn=f.sign==1?f.p1:f.p0;
        diagonal+=s*a;constant+=s*(n*pn+static_cast<long double>(f.b));
        actual+=s*(static_cast<long double>(f.e0)*f.p0+static_cast<long double>(f.e1)*f.p1+static_cast<long double>(f.b));
      }
      const long double target=-constant/diagonal;long double corrected=0;
      for(const auto& f:fs) {
        const long double p0=f.sign==1?target:f.p0,p1=f.sign==-1?target:f.p1;
        corrected+=static_cast<long double>(f.sign)*(p0*f.e0+p1*f.e1+static_cast<long double>(f.b));
      }
      const long double scale=static_cast<long double>(fs[0].volume)*fs[0].rate;
      std::cout<<i<<','<<std::numeric_limits<double>::digits<<','<<std::numeric_limits<long double>::digits
               <<','<<std::abs(actual)/scale<<','<<target<<','<<std::abs(corrected)/scale<<'\n';
    }
    if(std::numeric_limits<long double>::digits<=std::numeric_limits<double>::digits)
      throw std::runtime_error("This compiler does not provide extended long double precision");
    return 0;
  } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
