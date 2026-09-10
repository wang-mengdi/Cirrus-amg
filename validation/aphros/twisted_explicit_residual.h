// Diagnostic/opt-in repair for ConvDiffScalExp's unfilled residual halo before
// RedistributeCutCells. Only one block covering the whole domain is supported.
// The original flux assembly and redistribution helper are unchanged.
#pragma once
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

template<class M, class EB>
void TwistedExplicitResidual(FieldCell<typename M::Scal>& residual, const EB& eb,
                            M& m, int call) {
  const bool fix=std::getenv("APHROS_TWISTED_FIX_EXPLICIT_RESIDUAL_HALO")!=nullptr;
  const char* dump=std::getenv("APHROS_TWISTED_EXPLICIT_RESIDUAL_DUMP");
  if(!fix && !dump)return;
  const auto size=m.GetGlobalSize();
  const auto& block=m.GetInBlockCells();
  for(size_t d=0;d<M::dim;++d)
    if(block.GetBegin()[d]!=0 || block.GetSize()[d]!=size[d])
      throw std::runtime_error("Explicit residual halo diagnostic requires a single full-domain block");
  auto wrapped=residual;
  const auto& index=m.GetIndexCells();
  for(auto c:m.AllCells()) {
    auto key=index.GetMIdx(c);
    bool outside=false, valid=true;
    for(size_t d=0;d<M::dim;++d)if(key[d]<0 || key[d]>=size[d]) {
      outside=true;
      if(m.flags.is_periodic[d])key[d]=(key[d]%size[d]+size[d])%size[d];
      else valid=false;
    }
    if(outside && valid)wrapped[c]=residual[index.GetIdx(key)];
  }
  wrapped.SetHalo(1);
  // Three scalar components per outer iteration; keep the first two iterations
  // so instrumentation does not grow with a long physical time sequence.
  if(dump && call<6) {
    const auto before=UEmbed<M>::RedistributeCutCells(residual,eb);
    const auto after=UEmbed<M>::RedistributeCutCells(wrapped,eb);
    std::ostringstream name;name<<dump<<"_"<<call<<"_b"<<m.GetId()<<".csv";
    std::ofstream out(name.str());
    out<<std::setprecision(17)<<"x,y,z,volume,raw_residual,redistributed_original,redistributed_periodic_halo\n";
    for(auto c:eb.Cells()) {
      const auto x=m.GetCenter(c);
      out<<x[0]<<','<<x[1]<<','<<(M::dim>2?x[2]:0.)<<','<<eb.GetVolume(c)<<','
         <<residual[c]<<','<<before[c]<<','<<after[c]<<'\n';
    }
    out.flush();if(!out.good())throw std::runtime_error("Explicit residual diagnostic write failed");
  }
  if(fix)residual=std::move(wrapped);
}
