// Actual Embed volumes supplied by the case driver to the optional validation
// linear backend. This does not modify the original Proj/Embed implementation.
#pragma once
#include "geom/field.h"

template<class M> struct TwistedPressureGeometry {
  const M* mesh=nullptr;
  FieldCell<typename M::Scal> volume;
};

template<class M> inline TwistedPressureGeometry<M>& TwistedPressureVolumes() {
  static TwistedPressureGeometry<M> geometry;
  return geometry;
}
