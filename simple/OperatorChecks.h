#pragma once

#include "SimpleMesh.h"
#include <string>

namespace simple {

// Independent geometry/operator formula checks. These deliberately do not call
// Solver's private operators and must not be presented as end-to-end solver tests.
// `output` is the JSON filename. Affine, geometric, and conservation identities
// throw on failure; quadratic truncation errors are measurements, not pass claims.
void runOperatorChecks(const Mesh& mesh, const std::string& output);

} // namespace simple
