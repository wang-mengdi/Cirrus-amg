#include "simple/PressureRoundoffCycle.h"
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using Guard=simple::PressureRoundoffCycle;
int checks=0;
void require(bool value,const std::string& name) {
    if(!value)throw std::runtime_error(name);
    ++checks;
}
Guard::Observation sample(double residual) {
    return {residual,1e-18,0.026,4e-21,0.000625};
}
int sequence(const std::vector<Guard::Observation>& values) {
    Guard guard;int period=0;
    for(const auto& value:values)period=guard.observe(value);
    return period;
}
int main() {
    try {
        for(int period=2;period<=4;++period) {
            Guard guard;
            for(int i=0;i<2*period;++i) {
                const int result=guard.observe(sample((1+i%period)*1e-9));
                require(result==(i==2*period-1?period:0),"Two complete periods required");
            }
        }
        std::vector<Guard::Observation> constant(12,sample(1e-9));
        require(sequence(constant)==0,"Existing constant-repeat rule is separate");
        std::vector<Guard::Observation> decreasing;
        for(int i=8;i>0;--i)decreasing.push_back(sample(i*1e-9));
        require(sequence(decreasing)==0,"No cycle in monotone decrease");
        std::vector<Guard::Observation> longPeriod;
        for(int i=0;i<10;++i)longPeriod.push_back(sample((1+i%5)*1e-9));
        require(sequence(longPeriod)==0,"Do not infer an unobserved longer cycle");
        const std::vector<Guard::Observation> valid={sample(1e-9),sample(2e-9),sample(1e-9),sample(2e-9)};
        for(int position=0;position<4;++position) {
            for(int fault=0;fault<8;++fault) {
                auto values=valid;
                auto& value=values[position];
                if(fault==0)value.velocityChange=1e-15;
                if(fault==1)value.impulseChange=1e-17;
                if(fault==2)value.velocityScale=0;
                if(fault==3)value.impulseScale=0;
                if(fault==4)value.divergence=1.01e-8;
                if(fault==5)value.velocityChange=std::numeric_limits<double>::quiet_NaN();
                if(fault==6)value.impulseScale=std::numeric_limits<double>::infinity();
                if(fault==7)value.impulseChange=-1;
                require(sequence(values)==0,"Every update in both periods must qualify");
            }
        }
        auto boundary=sample(1e-9);
        const double halfEpsilon=0.5*std::numeric_limits<double>::epsilon();
        boundary.velocityChange=halfEpsilon*boundary.velocityScale;
        boundary.impulseChange=halfEpsilon*boundary.impulseScale;
        require(Guard::small(boundary),"Include exact half-epsilon bound");
        boundary.velocityChange=std::nextafter(boundary.velocityChange,std::numeric_limits<double>::infinity());
        require(!Guard::small(boundary),"Reject above the half-epsilon bound");
        Guard recovering;
        for(const auto& value:valid)recovering.observe(value);
        recovering.observe({1e-9,std::numeric_limits<double>::infinity(),0,0,0});
        for(int i=0;i<4;++i)
            require(recovering.observe(valid[i])==(i==3?2:0),"A large pass breaks the old window");
        require(recovering.size()==8,"Bounded history after many observations");
        require(Guard{}.observe(sample(2e-9))==0,"A new projection has no previous history");
        // The caller rejects a candidate when compensated divergence is above
        // its bound. This detector deliberately does not claim to check that
        // field, the full vector, mass conservation, or flow accuracy.
        std::cout<<"{\"passed\":true,\"checks\":"<<checks
                 <<",\"scope\":\"Actual C++ bounded cycle detector only; no CFD accuracy claim\"}\n";
        return 0;
    } catch(const std::exception& error) {
        std::cerr<<error.what()<<'\n';return 1;
    }
}
