import pytest

import library.kramers_kronig as kk
from library.phase_retrieval_universal import default_universal_phase_retrieval_recipe


EXPECTED_PUBLIC_API = {
    "kramers_kronig_real_from_imaginary",
}


def test_public_api_is_explicit_and_legacy_aliases_are_removed():
    assert hasattr(kk, "__all__")
    assert set(kk.__all__) == EXPECTED_PUBLIC_API
    assert not hasattr(kk, "discrete_kramers_kronig_dispersion")


def test_known_spectrum_paths_are_removed_from_recipe_and_validation():
    recipe = default_universal_phase_retrieval_recipe()
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
    from library.phase_retrieval_physical import constrain_complex_spectrum
    with pytest.raises(ValueError, match="spectral_constraint"):
        constrain_complex_spectrum([1j], spectral_constraint="known_beta")


@pytest.mark.parametrize('absorption_part', ['real', 'imag'])
def test_kk_preserves_fitted_absorption(absorption_part):
    import numpy as np
    from library.phase_retrieval_physical import constrain_complex_spectrum
    energy = np.linspace(700., 710., 5)
    absorption = np.array([0., .1, .3, .1, 0.])
    initial = absorption + .2j if absorption_part == 'real' else .2 + 1j * absorption
    result = constrain_complex_spectrum(
        initial, spectral_constraint='kk', energy_values=energy,
        absorption_part=absorption_part,
    )
    np.testing.assert_allclose(result.real if absorption_part == 'real' else result.imag, absorption)
    assert np.isfinite(result).all()
    assert np.ptp(result.imag if absorption_part == 'real' else result.real) > 0
