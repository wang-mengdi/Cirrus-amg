// Optional read-only capture shared by the independent driver and its linear
// backend. The original Aphros Proj/Embed implementation is not modified.
#pragma once
#include <cstddef>
#include "geom/field.h"

template<class M> struct TwistedPressureSnapshot {
  const M* mesh=nullptr;
  FieldCell<typename M::Expr> equations;
  FieldCell<typename M::Scal> returned_pressure;
  size_t sequence=0,gauge_raw=0;
  double tolerance=0,reported_residual=0;
  int refinements=0;
};

template<class M> inline TwistedPressureSnapshot<M>& TwistedLastPressure() {
  static TwistedPressureSnapshot<M> snapshot;
  return snapshot;
}
