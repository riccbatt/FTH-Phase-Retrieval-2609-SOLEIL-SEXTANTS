import pytest

import library.kramers_kronig as kk
from library.phase_retrieval_universal import default_general_phase_retrieval_recipe


EXPECTED_PUBLIC_API = {
    "absorption_to_beta",
    "scattering_factor_f2_to_beta",
    "extend_beta_with_reference",
    "refractive_index_from_beta_with_reference",
    "kramers_kronig_real_from_imaginary",
    "beta_to_delta",
}


def test_public_api_is_explicit_and_legacy_aliases_are_removed():
    assert hasattr(kk, "__all__")
    assert set(kk.__all__) == EXPECTED_PUBLIC_API
    assert not hasattr(kk, "discrete_kramers_kronig_dispersion")


def test_known_spectrum_paths_are_removed_from_recipe_and_validation():
    recipe = default_general_phase_retrieval_recipe()
    for key in (
        "known_charge_beta_spectrum",
        "known_charge_delta_spectrum",
        "known_magnetic_beta_spectrum",
        "known_magnetic_delta_spectrum",
        "known_spectrum_normalization",
        "fit_known_spectrum_scale",
        "fit_known_spectrum_offset",
    ):
        assert key not in recipe
    from library.phase_retrieval_universal import _verify_recipe
    for kind in ("charge", "magnetic"):
        bad_recipe = {**recipe, **{f"{kind}_spectral_constraint": "known_beta"}}
        with pytest.raises(ValueError, match="unsupported value"):
            _verify_recipe(bad_recipe, n_observations=1, n_energies=1)
