#pragma once

#include "SimpleMesh.h"
#include <Eigen/SparseCore>
#include <cmath>
#include <utility>

namespace simple {

// A normalized unevaluated sum. The low part carries face corrections that
// would otherwise disappear when added to a much larger stored flux.
struct FluxPair {
    double high=0.,low=0.;
    static FluxPair sum(double a,double b) {
        const double s=a+b,v=s-a;
        return {s,(a-(s-v))+(b-v)};
    }
    static FluxPair product(double a,double b) {
        const double p=a*b;return {p,std::fma(a,b,-p)};
    }
    double rounded() const {return high+low;}
};
inline FluxPair operator-(FluxPair a) {return {-a.high,-a.low};}
inline FluxPair operator+(FluxPair a,FluxPair b) {
    auto s=FluxPair::sum(a.high,b.high);const auto t=FluxPair::sum(a.low,b.low);
    s=FluxPair::sum(s.high,s.low+t.high);
    return FluxPair::sum(s.high,s.low+t.low);
}
inline FluxPair operator-(FluxPair a,FluxPair b) {return a+(-b);}
inline FluxPair operator*(FluxPair a,double b) {
    const auto p=FluxPair::product(a.high,b);
    return FluxPair::sum(p.high,p.low+a.low*b);
}
inline FluxPair operator/(FluxPair a,double b) {
    const double q=a.high/b;const auto r=a-FluxPair::product(q,b);
    return FluxPair::sum(q,r.rounded()/b);
}

class ConservativeFlux {
public:
    using Vector=Eigen::VectorXd;
    Vector high,low;
    ConservativeFlux()=default;
    explicit ConservativeFlux(Vector values,bool compensated=false):high(std::move(values)) {
        if(compensated)low=Vector::Zero(high.size());
    }
    bool compensated() const {return low.size()!=0;}
    Eigen::Index size() const {return high.size();}
    FluxPair get(Eigen::Index i) const {return {high[i],compensated()?low[i]:0.};}
    double operator[](Eigen::Index i) const {return get(i).rounded();}
    bool allFinite() const {return high.allFinite()&&low.allFinite();}
    void set(Eigen::Index i,FluxPair value) {
        if(compensated()) {high[i]=value.high;low[i]=value.low;}
        else high[i]=value.rounded();
    }
    void setDouble(Eigen::Index i,double value) {
        high[i]=value;if(compensated())low[i]=0.;
    }
    void add(Eigen::Index i,FluxPair value) {
        if(compensated())set(i,get(i)+value);
        else high[i]+=value.rounded();
    }
    void subtractProduct(const Vector& a,const Vector& b) {
        if(!compensated()) {high.array()-=a.array()*b.array();return;}
        for(Eigen::Index i=0;i<size();++i)add(i,-FluxPair::product(a[i],b[i]));
    }
    double quotient(Eigen::Index i,double denominator) const {
        return compensated()?(get(i)/denominator).rounded():high[i]/denominator;
    }
    double maximumVelocity(const Vector& areas) const {
        if(!compensated())return (high.array().abs()/areas.array()).maxCoeff();
        double result=0.;for(Eigen::Index i=0;i<size();++i)result=std::max(result,std::abs(quotient(i,areas[i])));
        return result;
    }
    double maximumVelocityDifference(const ConservativeFlux& other,const Vector& areas) const {
        if(!compensated()&&!other.compensated())return ((high-other.high).array().abs()/areas.array()).maxCoeff();
        double result=0.;for(Eigen::Index i=0;i<size();++i)
            result=std::max(result,std::abs(((get(i)-other.get(i))/areas[i]).rounded()));
        return result;
    }
    Vector rounded() const {return compensated()?Vector(high+low):high;}
    // Oriented shared-face balance: a single stored pair is used for both
    // neighboring cells. Only the final balance is rounded to a linear RHS.
    ConservativeFlux negativeDivergence(const Mesh& mesh) const {
        ConservativeFlux result(Vector::Zero(mesh.cells.size()),true);
        for(size_t j=0;j<mesh.faces.size();++j) {
            const auto& f=mesh.faces[j];if(f.neighbor<0)continue;
            const auto value=get(j);result.add(f.owner,-value);result.add(f.neighbor,value);
        }
        return result;
    }
    template<class Field> ConservativeFlux negativeWeightedDivergence(const Mesh& mesh,const Field& face,int component) const {
        ConservativeFlux result(Vector::Zero(mesh.cells.size()),true);
        for(size_t j=0;j<mesh.faces.size();++j) {
            const auto& f=mesh.faces[j];if(f.neighbor<0)continue;
            const auto value=get(j)*face(j,component);result.add(f.owner,-value);result.add(f.neighbor,value);
        }
        return result;
    }
    template<class Sparse> ConservativeFlux multiplied(const Sparse& matrix) const {
        ConservativeFlux result(Vector::Zero(matrix.rows()),true);
        for(int outer=0;outer<matrix.outerSize();++outer)
            for(typename Sparse::InnerIterator it(matrix,outer);it;++it)result.add(it.row(),get(it.col())*it.value());
        return result;
    }
};
} // namespace simple
