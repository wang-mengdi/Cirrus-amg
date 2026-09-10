#pragma once
#include <array>
#include <cmath>
#include <limits>

namespace simple {

// A bounded history for one pressure projection. This only identifies an
// additional stopping candidate; the caller must still check compensated
// divergence and all final physical acceptance conditions.
class PressureRoundoffCycle {
public:
    struct Observation {
        double divergence,velocityChange,velocityScale,impulseChange,impulseScale;
    };
    static bool small(const Observation& value) {
        const double halfEpsilon=0.5*std::numeric_limits<double>::epsilon();
        return std::isfinite(value.divergence) && value.divergence>=1e-10 && value.divergence<=1e-8 &&
            std::isfinite(value.velocityChange) && std::isfinite(value.velocityScale) &&
            std::isfinite(value.impulseChange) && std::isfinite(value.impulseScale) &&
            value.velocityScale>0 && value.impulseScale>0 &&
            value.velocityChange>=0 && value.velocityChange<=halfEpsilon*value.velocityScale &&
            value.impulseChange>=0 && value.impulseChange<=halfEpsilon*value.impulseScale;
    }
    int observe(Observation value) {
        if(count_==int(history_.size())) {
            for(int i=1;i<count_;++i)history_[i-1]=history_[i];
        } else ++count_;
        history_[count_-1]=value;
        for(int period=2;period<=4;++period) {
            if(count_<2*period)continue;
            const int first=count_-2*period;
            bool repeated=true,nonconstant=false;
            for(int i=first;i<count_;++i) {
                repeated=repeated && small(history_[i]);
                nonconstant=nonconstant || history_[i].divergence!=history_[first].divergence;
            }
            for(int i=0;i<period;++i)
                repeated=repeated && history_[first+i].divergence==history_[first+i+period].divergence;
            if(repeated && nonconstant)return period;
        }
        return 0;
    }
    int size() const {return count_;}
    const Observation& at(int index) const {return history_[index];}
private:
    std::array<Observation,8> history_{};
    int count_=0;
};

} // namespace simple
