"""Zufall der Demo: SplitMix64 (Steele, Lea, Flood 2014), ganzzahlig und plattformunabhängig.

Herkunft: Kopie des Stroms aus `entrollte-netze-demo/inverse_generator.py` (Klasse `SplitMix64`), auf das Nötige gekürzt und unter dem Präfix
`nvd_` geführt. Ausgabe k hängt nur von (Seed, k) ab. Python-`hash()` ist kein Zufalls-Seed (pro Prozess gesalzen) und kommt nirgends vor.
Plattformhinweis: Normalverteilte laufen über `log` und `cos` der C-Bibliothek und können sich zwischen Plattformen in der letzten Stelle
unterscheiden; Tests prüfen Float-Werte deshalb nur mit Toleranz oder als Bänder."""

import numpy as np

_GAMMA = 0x9E3779B97F4A7C15
_MASK = (1 << 64) - 1
_M1 = 0xBF58476D1CE4E5B9
_M2 = 0x94D049BB133111EB


class SplitMix64:
    """Strom gleichverteilter 64-Bit-Werte; `uniform` und `normal` leiten daraus Gleitkommazahlen ab."""

    def __init__(self, seed: int):
        self.state = int(seed) & _MASK

    def u64(self, size: int) -> np.ndarray:
        """Die nächsten `size` 64-Bit-Werte; der Strom rückt um `size` Schritte weiter."""
        idx = np.arange(1, size + 1, dtype=np.uint64)
        with np.errstate(over="ignore"):
            z = np.uint64(self.state) + idx * np.uint64(_GAMMA)
            z = (z ^ (z >> np.uint64(30))) * np.uint64(_M1)
            z = (z ^ (z >> np.uint64(27))) * np.uint64(_M2)
            z = z ^ (z >> np.uint64(31))
        self.state = (self.state + size * _GAMMA) & _MASK
        return z

    def uniform(self, size: int) -> np.ndarray:
        """Gleichverteilte Werte in [0, 1) mit 53 Bit."""
        return (self.u64(size) >> np.uint64(11)).astype(np.float64) * (1.0 / (1 << 53))

    def normal(self, size: int) -> np.ndarray:
        """Standardnormalverteilte Werte (Box-Muller, nur der Kosinus-Zweig); verbraucht 2 * size Gleichverteilte."""
        u1 = self.uniform(size)
        u2 = self.uniform(size)
        return np.sqrt(-2.0 * np.log1p(-u1)) * np.cos(2.0 * np.pi * u2)
