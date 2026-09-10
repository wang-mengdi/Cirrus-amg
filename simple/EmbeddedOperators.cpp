// Embedded-wall/face closures and deferred-term redistribution follow Aphros
// b60ce3da52c19935fa24c778f62f02141eaf7f80 (Copyright (c) 2021 ETH Zurich).
// MIT license: ../validation/aphros/LICENSE.aphros. The native octree
// coarse/fine extension and sparse operator assembly are implemented here.
#include "EmbeddedOperators.h"
#include "QuadraticReconstruction.h"
#include "ConstructionMemory.h"
#include "OpenFaceLookup.h"
#include <Eigen/QR>
#include <map>
#include <cmath>
#include <stdexcept>
#include <fstream>
#include <sstream>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <limits>

namespace simple {
namespace {
using Key=std::array<int,3>;
using FaceKey=std::array<int,4>;
using Triplet=Eigen::Triplet<double>;
// Assembly rows usually contain two entries; wall and coarse/fine rows are
// wider. Keep the same sorted-key traversal and accumulation order as std::map
// without allocating a tree node for every coefficient.
struct Row {
    using Term=std::pair<int,double>;
    std::vector<Term> terms;
    double& operator[](int column) {
        auto it=std::lower_bound(terms.begin(),terms.end(),column,
                               [](const Term& term,int key){return term.first<key;});
        if(it==terms.end() || it->first!=column)it=terms.insert(it,Term{column,0.});
        return it->second;
    }
    bool empty() const {return terms.empty();}
    auto begin() {return terms.begin();}
    auto end() {return terms.end();}
    auto begin() const {return terms.begin();}
    auto end() const {return terms.end();}
};
void add(Row& to,const Row& from,double factor) {for(auto p:from) to[p.first]+=factor*p.second;}
size_t nonzeros(const Row& row) {
    size_t count=0;for(auto term:row)if(term.second!=0)++count;return count;
}
void reserveRows(EmbeddedOperators::Sparse& result,int rows,int columns,size_t count) {
    using Sparse=EmbeddedOperators::Sparse;
    static_assert(Sparse::IsRowMajor,"Ordered row insertion requires row-major storage");
    if(count>size_t(std::numeric_limits<Sparse::StorageIndex>::max()))
        throw std::length_error("Embedded sparse operator exceeds index capacity");
    result.resize(rows,columns);result.reserve(Eigen::Index(count));
}
void appendRow(EmbeddedOperators::Sparse& result,int index,const Row& row,
               size_t expected,size_t& written) {
    result.startVec(index);
    for(auto term:row)if(term.second!=0) {
        if(written==expected)throw std::runtime_error("Embedded face coefficient count changed during construction");
        result.insertBack(index,term.first)=term.second;++written;
    }
}
EmbeddedOperators::Sparse matrix(std::vector<Row>& rows,int columns) {
    using Sparse=EmbeddedOperators::Sparse;
    static_assert(Sparse::IsRowMajor,"Ordered row insertion requires row-major storage");
    size_t nonzeros=0;
    for(const auto& row:rows)for(auto e:row)if(e.second!=0)++nonzeros;
    if(nonzeros>size_t(std::numeric_limits<Sparse::StorageIndex>::max()))
        throw std::length_error("Embedded sparse operator exceeds index capacity");
    // Row already owns sorted, unique coefficients with the original summation
    // order. Copy directly to compressed storage instead of keeping triplets
    // and Eigen's intermediate transpose alongside both assembly and output.
    Sparse result(rows.size(),columns);result.reserve(Eigen::Index(nonzeros));
    for(int i=0;i<int(rows.size());++i) {
        result.startVec(i);
        for(auto e:rows[i])if(e.second!=0)result.insertBack(i,e.first)=e.second;
    }
    result.finalize();
    std::vector<Row>().swap(rows);
    return result;
}
template<class MakeRow>
EmbeddedOperators::Sparse streamedMatrix(int rows,int columns,MakeRow makeRow) {
    size_t count=0,written=0;
    for(int i=0;i<rows;++i)count+=nonzeros(makeRow(i));
    EmbeddedOperators::Sparse result;reserveRows(result,rows,columns,count);
    for(int i=0;i<rows;++i)appendRow(result,i,makeRow(i),count,written);
    if(count!=written)throw std::runtime_error("Embedded cell coefficient count changed during construction");
    result.finalize();return result;
}
struct CellFaces {
    std::vector<size_t> offsets;
    // +(face+1) for its owner, -(face+1) for its neighbor. Each cell sees the
    // original face traversal order, including owner before neighbor if equal.
    std::vector<int> faces;
    explicit CellFaces(const Mesh& mesh):offsets(mesh.cells.size()+1,0) {
        for(const auto& face:mesh.faces) {
            ++offsets.at(size_t(face.owner)+1);
            if(face.neighbor>=0)++offsets.at(size_t(face.neighbor)+1);
        }
        for(size_t i=1;i<offsets.size();++i)offsets[i]+=offsets[i-1];
        faces.resize(offsets.back());auto cursor=offsets;
        for(int j=0;j<int(mesh.faces.size());++j) {
            const auto& face=mesh.faces[j];faces[cursor[face.owner]++]=j+1;
            if(face.neighbor>=0)faces[cursor[face.neighbor]++]=-(j+1);
        }
        for(size_t i=0;i<mesh.cells.size();++i)
            if(cursor[i]!=offsets[i+1])throw std::runtime_error("Embedded cell incidence count changed");
    }
};
} // namespace

void EmbeddedOperators::check(const Mesh& mesh,const std::string& output) const {
    double constantInterpolation=0,constantGradient=0,quadraticInterpolation=0,
        quadraticGradient=0,wallLinear=0;
    int interfaces=0;
    auto local=[&](int cell,const Face& f,double h) {
        Vec3 d=mesh.cells[cell].center-f.center;
        d[0]-=std::round(d[0]/mesh.extent[0])*mesh.extent[0];
        return Vec3(d/h);
    };
    for(int j=0;j<int(mesh.faces.size());++j) {
        const auto& f=mesh.faces[j];const double h=mesh.cells[f.owner].h;
        if(f.neighbor<0) {
            Vec3 derivative=Vec3::Zero();
            for(Sparse::InnerIterator it(wallGradient,j);it;++it)
                derivative+=it.value()*local(it.col(),f,h)*h;
            wallLinear=std::max(wallLinear,(derivative-f.embeddedNormal).norm());
            continue;
        }
        double si=0,sg=0;
        for(Sparse::InnerIterator it(interpolation,j);it;++it)si+=it.value();
        for(Sparse::InnerIterator it(faceGradient,j);it;++it)sg+=it.value()*h;
        constantInterpolation=std::max(constantInterpolation,std::abs(si+interpolationBoundaryWeight[j]-1));
        constantGradient=std::max(constantGradient,std::abs(sg));
        if(mesh.cells[f.owner].level==mesh.cells[f.neighbor].level)continue;
        ++interfaces;
        // Analytic degree <= 2 monomials in a local periodic chart. Evaluate
        // the actual assembled rows; no reconstruction coefficients are assumed.
        for(int a=0;a<3;++a)for(int b=-1;b<3;++b) {
            double qi=0,qg=0;
            auto value=[&](int cell) {Vec3 x=local(cell,f,h);return b<0?x[a]:x[a]*x[b];};
            for(Sparse::InnerIterator it(interpolation,j);it;++it)qi+=it.value()*value(it.col());
            for(Sparse::InnerIterator it(faceGradient,j);it;++it)qg+=h*it.value()*value(it.col());
            const double exact=b<0&&a==f.axis?f.sign:0.;
            quadraticInterpolation=std::max(quadraticInterpolation,std::abs(qi));
            quadraticGradient=std::max(quadraticGradient,std::abs(qg-exact));
        }
    }
    Eigen::VectorXd values(mesh.cells.size());
    for(int i=0;i<values.size();++i)values[i]=std::sin(double(i)*.137)+.3*std::cos(double(i)*.173);
    const Eigen::VectorXd viscous=compactDiffusion*values+deferredDiffusion*values;
    const Eigen::VectorXd wall=wallGradient*values;
    double wallSum=0,scale=0;
    for(int j=0;j<int(mesh.faces.size());++j)if(mesh.faces[j].neighbor<0) {
        double v=-mesh.faces[j].area*wall[j];wallSum+=v;scale+=std::abs(v);
    }
    const double balance=std::abs(viscous.sum()-wallSum)/std::max(scale,1e-30);
    double advectionBalance=0;
    if(cartesianFaceInterpolation.rows()) {
        Eigen::VectorXd testFlux(mesh.faces.size());
        for(int j=0;j<testFlux.size();++j)testFlux[j]=mesh.faces[j].neighbor<0?0:std::sin(j*.731)*mesh.faces[j].area;
        const auto advection=upwindAdvection(mesh,testFlux,1.);
        const Eigen::VectorXd net=(advection.first+advection.second)*values;
        advectionBalance=std::abs(net.sum())/std::max(net.cwiseAbs().sum(),1e-30);
    }
    const bool passed=constantInterpolation<1e-10 && constantGradient<1e-10 &&
        quadraticInterpolation<1e-10 && quadraticGradient<1e-10 && wallLinear<1e-10 && balance<1e-10 && advectionBalance<1e-10;
    nlohmann::json report{{"passed",passed},{"scope","Assembled cut operators: constant preservation, interface monomials, wall affine derivative, global viscous flux balance"},
        {"coarse_fine_faces_tested",interfaces},{"constant_interpolation_max",constantInterpolation},
        {"constant_gradient_max_scaled",constantGradient},{"interface_quadratic_interpolation_max",quadraticInterpolation},
        {"interface_quadratic_gradient_max_scaled",quadraticGradient},{"wall_affine_gradient_max",wallLinear},
        {"global_viscous_flux_balance",balance},{"maximum_zero_extension_weight",interpolationBoundaryWeight.maxCoeff()},
        {"advection_checked",bool(cartesianFaceInterpolation.rows())},{"global_advective_flux_balance",advectionBalance},
        {"zero_extension_face_count",(interpolationBoundaryWeight.array()>0).count()},
        {"interpolation_constant_policy","Includes coefficient of fixed zero data on excluded Cartesian stencil faces, as in Aphros"}};
    std::ofstream(output)<<report.dump(2)<<'\n';
    if(!passed)throw std::runtime_error("Embedded operator consistency checks failed; see "+output);
}

EmbeddedOperators::EmbeddedOperators(const Mesh& mesh,const QuadraticReconstruction* quadratic,bool convection,
                                      bool explicitUpdate,bool redistributeDiffusion) : explicitMomentum(explicitUpdate) {
    constructionMemory("embedded.begin");
    if(!mesh.embedded) throw std::runtime_error("Embedded operators require cut geometry");
    if(mesh.coarseFineFaces && !quadratic) throw std::runtime_error("Embedded coarse/fine faces require quadratic reconstruction");
    const int nc=int(mesh.cells.size()),nf=int(mesh.faces.size());
    double h=mesh.cells.front().h;
    for(const auto& c:mesh.cells)h=std::min(h,c.h);
    const int nx=int(std::llround(mesh.extent[0]/h));
    const int ny=int(std::llround(mesh.extent[1]/h)),nz=int(std::llround(mesh.extent[2]/h));
    std::map<Key,int> lookup,coarseLookup;
    auto keyOf=[&](const Vec3& x) {Key k{};for(int d=0;d<3;++d)k[d]=int(std::llround(x[d]/h-.5));return k;};
    auto wrap=[&](Key k) {k[0]=(k[0]%nx+nx)%nx;return k;};
    for(int i=0;i<nc;++i) {
        if(std::abs(mesh.cells[i].h-h)<h*1e-12)lookup[keyOf(mesh.cells[i].center)]=i;
        else {
            Key k{};for(int d=0;d<3;++d)k[d]=int(std::llround(mesh.cells[i].center[d]/(2*h)-.5));
            coarseLookup[k]=i;
        }
    }
    // Distinguish an excluded source face (Aphros initializes its sampled value
    // to zero) from a fluid source hidden inside a coarse leaf (invalid padding).
    const size_t openFaceSize=size_t(3)*nx*(ny+1)*(nz+1);
    auto faceIndex=[&](FaceKey key)->size_t {
        key[1]=(key[1]%nx+nx)%nx;
        if(key[2]<0||key[2]>ny||key[3]<0||key[3]>nz)return openFaceSize;
        return ((size_t(key[0])*nx+key[1])*(ny+1)+key[2])*(nz+1)+key[3];
    };
    auto visitOpenFaces=[&](auto visit) {for(int j=0;j<nf;++j)if(mesh.faces[j].neighbor>=0) {
        const auto& f=mesh.faces[j];
        int fine=mesh.cells[f.owner].h<mesh.cells[f.neighbor].h?f.owner:f.neighbor;
        if(std::abs(mesh.cells[fine].h-h)>h*1e-12)continue;
        FaceKey key{f.axis,0,0,0};
        for(int d=0;d<3;++d)key[d+1]=int(std::llround(d==f.axis?f.center[d]/h:mesh.cells[fine].center[d]/h-.5));
        const auto index=faceIndex(key);
        if(index==openFaceSize)throw std::runtime_error("Open face key outside embedded box");
        visit(index,j);
    }};
    OpenFaceLookup openFace(size_t(3)*nx*(ny+1),nz+1,visitOpenFaces);
    constructionMemory("embedded.open_faces_ready");
    interpolationBoundaryWeight=Eigen::VectorXd::Zero(nf);
    std::vector<std::vector<std::pair<int,Vec3>>> stencil(nc);
    for(int i=0;i<nc;++i) {
        // Only embedded-wall fits and cut-cell deferred-source redistribution
        // use this neighborhood. Full fluid cells need neither stencil.
        if(!mesh.cells[i].cut)continue;
        auto k=keyOf(mesh.cells[i].center);
        for(int a=-1;a<=1;++a)for(int b=-1;b<=1;++b)for(int c=-1;c<=1;++c) {
            auto it=lookup.find(wrap(Key{k[0]+a,k[1]+b,k[2]+c}));
            if(it!=lookup.end())stencil[i].push_back({it->second,Vec3(a,b,c)});
        }
    }
    std::vector<Vec3> wallNormal(nc,Vec3::Zero());
    for(const auto& f:mesh.faces)if(f.neighbor<0)wallNormal[f.owner]=f.embeddedNormal;
    auto baseRow=[&](FaceKey fk,bool derivative) {
        int axis=fk[0];Key positive{fk[1],fk[2],fk[3]},negative=positive;--negative[axis];
        auto p=lookup.find(wrap(negative)),n=lookup.find(wrap(positive));
        auto inCoarse=[&](Key k) {
            k=wrap(k);for(auto& q:k)q=int(std::floor(q*.5));
            return coarseLookup.count(k)>0;
        };
        if((p==lookup.end()&&inCoarse(negative))||(n==lookup.end()&&inCoarse(positive)))
            throw std::runtime_error("Cut-face interpolation reaches a coarse cell: insufficient wall padding");
        const auto index=faceIndex(fk);
        if(index==openFace.size()||openFace[index]<0)return Row{};
        if(p==lookup.end()||n==lookup.end()) {
            std::ostringstream message;message<<"Embedded face interpolation samples missing fine cells: axis="<<axis
                <<" face="<<fk[1]<<','<<fk[2]<<','<<fk[3]<<" negative_present="<<(p!=lookup.end())
                <<" positive_present="<<(n!=lookup.end());
            throw std::runtime_error(message.str());
        }
        Row row;row[p->second]=derivative?-1/h:.5;row[n->second]+=derivative?1/h:.5;return row;
    };
    auto derivative=[&](int cell,int d) {
        Row row;for(auto term:quadratic->derivativeRow(cell,d))row[term.first]+=term.second;return row;
    };
    auto taylor=[&](int cell,const Vec3& r) {
        Row row;row[cell]=1;
        for(int d=0;d<3;++d)add(row,derivative(cell,d),r[d]);
        const double weights[6]={.5*r[0]*r[0],.5*r[1]*r[1],.5*r[2]*r[2],r[0]*r[1],r[0]*r[2],r[1]*r[2]};
        for(int d=0;d<6;++d)add(row,derivative(cell,d+3),weights[d]);
        return row;
    };
    struct FaceRows {Row interpolation,gradient,wall,mix;double boundaryWeight=0;};
    auto faceRows=[&](int j) {
        FaceRows rows;
        const auto& f=mesh.faces[j];
        if(f.neighbor>=0) {
            const double localh=mesh.cells[f.owner].h;
            if(mesh.cells[f.owner].level!=mesh.cells[f.neighbor].level) {
                if(convection)rows.mix[j]=1.;
                const double weight=mesh.cells[f.neighbor].h/(localh+mesh.cells[f.neighbor].h);
                add(rows.interpolation,taylor(f.owner,f.ownerOffset),weight);
                add(rows.interpolation,taylor(f.neighbor,f.neighborOffset),1-weight);
                rows.gradient[f.owner]-=1/f.distance;rows.gradient[f.neighbor]+=1/f.distance;
                const Vec3 normal=Vec3::Unit(f.axis)*f.sign;
                const Vec3 tangent=f.delta-f.distance*normal,skew=.5*(f.ownerOffset+f.neighborOffset);
                for(int c:{f.owner,f.neighbor}) {
                    for(int d=0;d<3;++d)add(rows.gradient,derivative(c,d),-.5*tangent[d]/f.distance);
                    const int hid[3][3]={{3,6,7},{6,4,8},{7,8,5}};
                    for(int d=0;d<3;++d)add(rows.gradient,derivative(c,hid[f.axis][d]),.5*f.sign*skew[d]);
                }
            } else if(std::abs(f.area-localh*localh)<1e-12*localh*localh) {
                if(convection)rows.mix[j]=1.;
                rows.interpolation[f.owner]=.5;rows.interpolation[f.neighbor]=.5;
                rows.gradient[f.owner]=-1/f.distance;rows.gradient[f.neighbor]=1/f.distance;
            } else {
            if(std::abs(localh-h)>h*1e-12)throw std::runtime_error("Cut face is not at the finest wall resolution");
            FaceKey fk{f.axis,0,0,0};Vec3 gridCenter=mesh.cells[f.owner].center;
            gridCenter[f.axis]+=.5*h;
            for(int d=0;d<3;++d)fk[d+1]=int(std::llround(gridCenter[d]/h-(d==f.axis?0.:.5)));
            const int a=(f.axis+1)%3,b=(f.axis+2)%3;
            const double da=(f.center[a]-gridCenter[a])/h,db=(f.center[b]-gridCenter[b])/h;
            for(int u=0;u<2;++u)for(int v=0;v<2;++v) {
                const double weight=(u?std::abs(da):1-std::abs(da))*(v?std::abs(db):1-std::abs(db));
                if(weight<1e-15)continue;
                // Aphros chooses the inward tangent from the negative-side
                // cell normal, even when curvature reverses a tiny centroid shift.
                auto q=fk;q[a+1]+=u*(wallNormal[f.owner][a]<0?1:-1);q[b+1]+=v*(wallNormal[f.owner][b]<0?1:-1);
                const Row valueRow=baseRow(q,false);
                if(convection && !valueRow.empty())rows.mix[openFace[faceIndex(q)]]+=weight;
                if(valueRow.empty())rows.boundaryWeight+=weight;
                add(rows.interpolation,valueRow,weight);add(rows.gradient,baseRow(q,true),weight);
            }
            }
        } else {
            const int p=f.owner,ns=int(stencil[p].size());
            Eigen::MatrixXd design(ns,4);
            for(int r=0;r<ns;++r)design.row(r)<<stencil[p][r].second.transpose(),1.;
            Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(design);
            qr.setThreshold(1e-12);
            if(qr.rank()!=4)throw std::runtime_error("Rank-deficient embedded-wall fit");
            Eigen::Vector4d evaluation;evaluation<<((f.center-f.embeddedNormal*h-mesh.cells[p].center)/h),1.;
            Eigen::RowVectorXd coeff=evaluation.transpose()*qr.solve(Eigen::MatrixXd::Identity(ns,ns));
            for(int r=0;r<ns;++r)rows.wall[stencil[p][r].first]-=coeff[r]/h;
        }
        return rows;
    };
    constructionMemory("embedded.lookup_stencils_ready");
    // Count the same face formulas before allocating exact compressed storage.
    // Only one face's editable rows remain live, rather than several vectors
    // and heap allocations for every face alongside the final sparse arrays.
    std::array<size_t,4> counts{},written{};
    for(int j=0;j<nf;++j) {
        const auto rows=faceRows(j);
        counts[0]+=nonzeros(rows.interpolation);counts[1]+=nonzeros(rows.gradient);
        counts[2]+=nonzeros(rows.wall);counts[3]+=nonzeros(rows.mix);
    }
    constructionMemory("embedded.face_nonzeros_counted");
    reserveRows(interpolation,nf,nc,counts[0]);reserveRows(faceGradient,nf,nc,counts[1]);
    reserveRows(wallGradient,nf,nc,counts[2]);
    if(convection)reserveRows(cartesianFaceInterpolation,nf,nf,counts[3]);
    for(int j=0;j<nf;++j) {
        const auto rows=faceRows(j);
        interpolationBoundaryWeight[j]=rows.boundaryWeight;
        appendRow(interpolation,j,rows.interpolation,counts[0],written[0]);
        appendRow(faceGradient,j,rows.gradient,counts[1],written[1]);
        appendRow(wallGradient,j,rows.wall,counts[2],written[2]);
        if(convection)appendRow(cartesianFaceInterpolation,j,rows.mix,counts[3],written[3]);
    }
    if(counts!=written)throw std::runtime_error("Embedded face coefficient count changed during construction");
    interpolation.finalize();faceGradient.finalize();wallGradient.finalize();
    if(convection)cartesianFaceInterpolation.finalize();
    constructionMemory("embedded.face_matrices_ready");
    // Match Aphros redistribution of the deferred (constant) viscous terms.
    // The compact implicit stencil and the body/pressure sources stay separate.
    auto redistributionColumn=[&](int i) {
        Row column;
        const double fraction=mesh.cells[i].volume/std::pow(mesh.cells[i].h,3);
        column[i]+=fraction;
        if(fraction>=1.)return column;
        double sum=0;for(auto s:stencil[i])if(s.first!=i)sum+=mesh.cells[s.first].volume;
        if(!(sum>0))throw std::runtime_error("Isolated cut cell has no redistribution stencil");
        for(auto s:stencil[i])if(s.first!=i)column[s.first]+=(1-fraction)*mesh.cells[s.first].volume/sum;
        return column;
    };
    {
        // Original sources arrive in ascending cell order, hence each target
        // row receives ascending column IDs. Merge repeated targets locally
        // before counting/writing, retaining their original accumulation order.
        Eigen::VectorXi remaining=Eigen::VectorXi::Zero(nc);size_t count=0;
        for(int i=0;i<nc;++i)for(auto term:redistributionColumn(i))if(term.second!=0) {
            ++remaining[term.first];++count;
        }
        if(count>size_t(std::numeric_limits<Sparse::StorageIndex>::max()))
            throw std::length_error("Embedded redistribution exceeds index capacity");
        constructionMemory("embedded.redistribution_nonzeros_counted");
        redistribution.resize(nc,nc);redistribution.reserve(remaining);
        for(int i=0;i<nc;++i)for(auto term:redistributionColumn(i))if(term.second!=0) {
            if(remaining[term.first]<=0)throw std::runtime_error("Embedded redistribution count changed");
            --remaining[term.first];redistribution.insertBackUncompressed(term.first,i)=term.second;
        }
        if(remaining.any())throw std::runtime_error("Embedded redistribution count changed");
        redistribution.makeCompressed();
    }
    constructionMemory("embedded.redistribution_ready");
    // All geometry lookup work is done before constructing cell operators.
    lookup.clear();coarseLookup.clear();
    decltype(openFace){}.swap(openFace);decltype(stencil){}.swap(stencil);
    decltype(wallNormal){}.swap(wallNormal);
    constructionMemory("embedded.face_workspace_released");
    const CellFaces incident(mesh);
    constructionMemory("embedded.cell_faces_ready");
    struct CellRows {Row compact,full;};
    auto diffusionRows=[&](int cell) {
        CellRows rows;
        for(size_t q=incident.offsets[cell];q<incident.offsets[cell+1];++q) {
            const int side=incident.faces[q],j=std::abs(side)-1;const auto& f=mesh.faces[j];
            if(f.neighbor>=0) {
                const double d=f.area/f.distance;
                rows.compact[cell]+=d;rows.compact[side>0?f.neighbor:f.owner]-=d;
                const double factor=side>0?-f.area:f.area;
                for(Sparse::InnerIterator it(faceGradient,j);it;++it)rows.full[it.col()]+=factor*it.value();
            } else {
                rows.compact[cell]+=2*f.area/h;
                for(Sparse::InnerIterator it(wallGradient,j);it;++it)rows.full[it.col()]+=-f.area*it.value();
            }
        }
        return rows;
    };
    // Upstream conv=exp keeps only the time derivative in the implicit system;
    // it redistributes the complete diffusion flux residual at the old iterate.
    {
        std::array<size_t,2> counts{},written{};
        for(int i=0;i<nc;++i) {
            const auto rows=diffusionRows(i);
            if(!explicitMomentum)counts[0]+=nonzeros(rows.compact);
            counts[1]+=nonzeros(rows.full);
        }
        constructionMemory("embedded.cell_nonzeros_counted");
        Sparse fullDiffusion;reserveRows(fullDiffusion,nc,nc,counts[1]);
        if(explicitMomentum)compactDiffusion=Sparse(nc,nc);
        else reserveRows(compactDiffusion,nc,nc,counts[0]);
        for(int i=0;i<nc;++i) {
            const auto rows=diffusionRows(i);
            if(!explicitMomentum)appendRow(compactDiffusion,i,rows.compact,counts[0],written[0]);
            appendRow(fullDiffusion,i,rows.full,counts[1],written[1]);
        }
        if(counts!=written)throw std::runtime_error("Embedded diffusion coefficient count changed during construction");
        if(!explicitMomentum)compactDiffusion.finalize();
        fullDiffusion.finalize();
        if(redistributeDiffusion)deferredDiffusion=redistribution*(fullDiffusion-compactDiffusion);
        else deferredDiffusion=fullDiffusion-compactDiffusion;
    }
    constructionMemory("embedded.diffusion_matrices_ready");
    for(int d=0;d<3;++d) {
        cellGradient[d]=streamedMatrix(nc,nc,[&](int i) {
            Row row;double area=0;
            for(size_t q=incident.offsets[i];q<incident.offsets[i+1];++q) {
                const auto j=std::abs(incident.faces[q])-1;const auto& f=mesh.faces[j];
                if(f.neighbor<0||f.axis!=d)continue;
                for(Sparse::InnerIterator it(faceGradient,j);it;++it)row[it.col()]+=f.area*it.value();
                area+=f.area;
            }
            if(!(area>0))throw std::runtime_error("Missing directional aperture for pressure gradient");
            for(auto& term:row)term.second/=area;
            return row;
        });
        constructionMemory(d==0?"embedded.cell_gradient_x":d==1?"embedded.cell_gradient_y":"embedded.cell_gradient_z");
    }
    constructionMemory("embedded.before_workspace_release");
}

std::pair<EmbeddedOperators::Sparse,EmbeddedOperators::Sparse>
EmbeddedOperators::upwindAdvection(const Mesh& mesh,const Eigen::VectorXd& flux,double rho) const {
    const int nc=int(mesh.cells.size()),nf=int(mesh.faces.size());
    if(cartesianFaceInterpolation.rows()!=nf)throw std::runtime_error("Embedded convection operators were not initialized");
    std::vector<Triplet> upwind,incidence;
    upwind.reserve(2*nf);incidence.reserve(2*nf);
    for(int j=0;j<nf;++j) {
        const auto& f=mesh.faces[j];if(f.neighbor<0)continue;
        if(flux[j]>0)upwind.emplace_back(j,f.owner,1.);
        else if(flux[j]<0)upwind.emplace_back(j,f.neighbor,1.);
        else {upwind.emplace_back(j,f.owner,.5);upwind.emplace_back(j,f.neighbor,.5);}
        incidence.emplace_back(f.owner,j,rho*flux[j]);incidence.emplace_back(f.neighbor,j,-rho*flux[j]);
    }
    Sparse pick(nf,nc),balance(nc,nf);
    pick.setFromTriplets(upwind.begin(),upwind.end());balance.setFromTriplets(incidence.begin(),incidence.end());
    Sparse compact=balance*pick;
    const Sparse full=balance*(cartesianFaceInterpolation*pick);
    // Match either the implicit FOU/deferred split or original conv=exp, which
    // redistributes the complete advective residual. Both share the same faces.
    if(explicitMomentum)compact.setZero();
    Sparse deferred=redistribution*(full-compact);
    return {std::move(compact),std::move(deferred)};
}
}
