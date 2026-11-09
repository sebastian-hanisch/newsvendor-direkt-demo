"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen; Permalink rastet ein und begrenzt."""

import nvd_constants as C
import nvd_evaluation as ev
import nvd_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_allowed_levels():
    for name, preset in C.PRESETS.items():
        for key, state_key in P.PRESET_KEYS.items():
            spec = P.SETTING_SPECS[state_key]
            if spec.options is not None:
                assert preset[key] in spec.options, (name, key)
            else:
                assert spec.lo <= preset[key] <= spec.hi, (name, key)


def test_default_preset_equals_the_default_settings():
    first = next(iter(C.PRESETS.values()))
    default = ev.Settings()
    assert (first["welt"], first["tau"], first["n"], first["p"], first["lam"], first["n_strafe"], first["tau_hand"], first["seed"]) == (
        default.welt, default.tau, default.n, default.p, default.lam, default.n_strafe, default.tau_hand, default.seed)
    for key, state_key in P.PRESET_KEYS.items():
        assert P.SETTING_SPECS[state_key].default == first[key], key


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


def test_all_presets_differ_from_each_other():
    values = [tuple(sorted(p.items())) for p in C.PRESETS.values()]
    assert len(set(values)) == len(values)


def test_the_handbook_tau_is_one_of_the_levels():
    assert C.DEFAULT_TAU_HAND in C.TAU_OPTIONS and C.DEFAULT_TAU in C.TAU_OPTIONS
    assert C.DEFAULT_LAM in C.LAM_OPTIONS


def test_the_penalty_levels_are_the_cross_validation_grid():
    import nvd_algorithm as A
    assert C.LAM_OPTIONS == A.LAMS


def test_snap_rounds_to_the_nearest_level_and_rejects_unknown_names():
    assert P.snap(P.SETTING_SPECS["n_select"], 250) == 300 and P.snap(P.SETTING_SPECS["n_select"], 1) == 30 and P.snap(P.SETTING_SPECS["n_select"], 10 ** 9) == 1000
    assert P.snap(P.SETTING_SPECS["tau_select"], 0.93) == 0.95 and P.snap(P.SETTING_SPECS["tau_select"], 0.1) == 0.5
    assert P.snap(P.SETTING_SPECS["welt_select"], "linear") == "linear" and P.snap(P.SETTING_SPECS["welt_select"], "gibt-es-nicht") is None
    assert P.snap(P.SETTING_SPECS["seed_input"], -5) == 0 and P.snap(P.SETTING_SPECS["seed_input"], 10 ** 12) == C.SEED_MAX
