"""Read and independently balance exported binary64 high/low shared fluxes.

Decimal converts each round-tripped binary64 part exactly. Accumulation uses
120 decimal digits; the acceptance bounds are the existing physical gates.
"""
import csv
from decimal import Decimal, localcontext
from itertools import zip_longest
import math
from pathlib import Path
import numpy as np


def balance_fields(folder, cells, height, sections):
    folder = Path(folder)
    count = len(cells)
    high_net = np.zeros(count)
    rates = np.linalg.norm(cells[:, 2:], axis=1)
    rate = float(rates.max()) / height
    if not rate > 0:
        raise ValueError('Expected a nonzero physical velocity scale')
    with localcontext() as context:
        context.prec = 120
        net = [Decimal(0)] * count
        nonzero_low = 0
        with (folder/'mesh_faces.csv').open() as f, (folder/'flux.csv').open() as q:
            faces, fluxes = csv.DictReader(f), csv.DictReader(q)
            if fluxes.fieldnames != ['id', 'flux', 'flux_low']:
                raise ValueError('Twofold flux requires ordered high and low columns')
            for index, pair in enumerate(zip_longest(faces, fluxes)):
                face, flux = pair
                if face is None or flux is None or int(face['id']) != index or int(flux['id']) != index:
                    raise ValueError('Missing or unordered face/flux row')
                owner, neighbor = int(face['owner']), int(face['neighbor'])
                if not 0 <= owner < count or not -1 <= neighbor < count:
                    raise ValueError('Invalid shared-face incidence')
                high, low = float(flux['flux']), float(flux['flux_low'])
                if not math.isfinite(high) or not math.isfinite(low):
                    raise ValueError('Nonfinite conservative flux part')
                if math.fsum((high, low)) != high or abs(low) > .5*math.ulp(high):
                    raise ValueError('Non-normalized conservative flux pair')
                nonzero_low += low != 0
                if neighbor < 0:
                    if high != 0 or low != 0:
                        raise ValueError('Nonzero stationary wall flux part')
                    continue
                value = Decimal.from_float(high) + Decimal.from_float(low)
                net[owner] += value
                net[neighbor] -= value
                high_net[owner] += high
                high_net[neighbor] -= high
        through = float(sections.mean())
        if not math.isfinite(through) or through == 0:
            raise ValueError('Invalid throughflow')
        maximum, worst, absolute = Decimal(0), -1, Decimal(0)
        for cell, value in enumerate(net):
            volume = float(cells[cell, 1])
            if not math.isfinite(volume) or volume <= 0:
                raise ValueError('Invalid actual cut-cell volume')
            absolute += abs(value)
            residual = abs(value) / Decimal.from_float(volume)
            if residual > maximum:
                maximum, worst = residual, cell
        result = {
            'flux_storage': 'twofold', 'accumulation_decimal_digits': 120,
            'faces': index+1, 'nonzero_low_parts': nonzero_low,
            'worst_cell': worst, 'divergence_absolute_linf_decimal': str(maximum),
            'divergence_relative_linf': float(maximum/Decimal.from_float(rate)),
            'global_absolute_cell_flux_over_throughflow': float(absolute/abs(Decimal.from_float(through))),
            'section_flux_relative_spread': float(np.ptp(sections)/abs(through)),
            'rounded_high_only_naive_divergence_relative_linf': float(np.max(abs(high_net)/cells[:, 1])/rate),
            'rounded_high_only_value_is_diagnostic_not_acceptance': True,
        }
        result['passed'] = (result['divergence_relative_linf'] < 1e-7 and
                            result['global_absolute_cell_flux_over_throughflow'] < 1e-8 and
                            result['section_flux_relative_spread'] < 1e-8)
        return result
