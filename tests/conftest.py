"""Gemeinsame Fixtures: das Handbeispiel, an dem jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def hand():
    """Ein Merkmal x in {0, 1}; x = 0: Nachfragen 4, 5, 6; x = 1: 8, 12, 16; b = 4, h = 1, tau = 0,8.

    Von Hand: KQ ergibt 5 + 7x (Residuen -1, 0, 1, -4, 0, 4); das Residuum der Rangzahl ceil(0,8 * 6) = 5 ist 1, also KQ + Quantil 6 + 7x (Mengen 6 und 13);
    direkt (Pinball-LP): Achsenabschnitt 6, Steigung 10 (Mengen 6 und 16), Pinball-Summe 3,0."""
    X = np.array([[0.0], [0.0], [0.0], [1.0], [1.0], [1.0]])
    y = np.array([4.0, 5.0, 6.0, 8.0, 12.0, 16.0])
    return X, y, 0.8
