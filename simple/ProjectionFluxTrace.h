#pragma once

#include "ConservativeFlux.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace simple {
// Read-only preview of precisely the existing twofold face update. Keep only
// the faces incident to the eventual worst cell instead of copying all q.
// Actual physical updates and residual acceptance still run independently.
struct ProjectionFluxTrace {
    struct FaceBefore {int face;FluxPair before,after;double delta;};
    std::vector<FaceBefore> worstFaces;
    Eigen::Index worst=0;
    double worstDefect=0.,compensated=0.,volumeL2=0.;
    double velocityChange=0.,velocityScale=0.;
    int changedFlux=0,unrepresented=0;

    void capture(const Mesh& mesh,const ConservativeFlux& q,const Eigen::VectorXd& areas,
                 const Eigen::VectorXd& pressureAreas,const Eigen::VectorXd& gradient,
                 const Eigen::VectorXd& volumes,double volume) {
        if(!q.compensated())throw std::invalid_argument("Compact pressure trace requires twofold flux");
        *this=ProjectionFluxTrace{};
        Eigen::VectorXd defect;
        {
            ConservativeFlux balance(Eigen::VectorXd::Zero(mesh.cells.size()),true);
            for(std::size_t j=0;j<mesh.faces.size();++j) {
                const auto& face=mesh.faces[j];if(face.neighbor<0)continue;
                const auto value=q.get(j)+(-FluxPair::product(pressureAreas[j],gradient[j]));
                balance.add(face.owner,-value);balance.add(face.neighbor,value);
            }
            defect=balance.rounded();
        }
        compensated=(defect.array().abs()/volumes.array()).maxCoeff(&worst);
        volumeL2=std::sqrt((defect.array().square()/volumes.array()).sum()/volume);
        worstDefect=defect[worst];
        for(std::size_t j=0;j<mesh.faces.size();++j) {
            const auto& face=mesh.faces[j];if(face.neighbor<0)continue;
            const auto before=q.get(j),after=before+(-FluxPair::product(pressureAreas[j],gradient[j]));
            const double delta=-(pressureAreas[j]*gradient[j]);
            velocityChange=std::max(velocityChange,std::abs(delta)/areas[j]);
            velocityScale=std::max(velocityScale,std::abs(before.rounded())/areas[j]);
            const bool unchanged=after.high==before.high&&after.low==before.low;
            changedFlux+=!unchanged;unrepresented+=delta!=0.&&unchanged;
            if(face.owner==worst||face.neighbor==worst)worstFaces.push_back({int(j),before,after,delta});
        }
    }
    void verify(const ConservativeFlux& q,Eigen::Index actualWorst,double actualDefect,
                double actualCompensated,double actualVolumeL2) const {
        if(actualWorst!=worst||actualDefect!=worstDefect||actualCompensated!=compensated||actualVolumeL2!=volumeL2)
            throw std::runtime_error("Pressure trace preview differs from actual face balance");
        for(const auto& face:worstFaces) {
            const auto actual=q.get(face.face);
            if(actual.high!=face.after.high||actual.low!=face.after.low)
                throw std::runtime_error("Pressure trace preview differs from actual stored flux");
        }
    }
};
} // namespace simple
