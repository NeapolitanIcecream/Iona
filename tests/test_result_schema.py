from iona.pipeline.result_schema import PlateSolveResult


def test_plate_solve_to_wcs_ignores_non_wcs_commentary_cards() -> None:
    """Regression: nova WCS headers may include COMMENT text with embedded newlines."""
    plate = PlateSolveResult(
        success=True,
        wcs_header={
            "WCSAXES": 2,
            "CTYPE1": "RA---TAN",
            "CTYPE2": "DEC--TAN",
            "CRVAL1": 290.240574582,
            "CRVAL2": -23.9280015811,
            "CRPIX1": 673.354685465,
            "CRPIX2": 847.641438802,
            "CD1_1": -0.0442806276231,
            "CD1_2": 0.0156679714366,
            "CD2_1": -0.0155639247674,
            "CD2_2": -0.0446113541332,
            "COMMENT": "Created by Astrometry.net.\nFor more details, see example.",
        },
    )

    wcs = plate.to_wcs()

    assert wcs is not None
    assert abs(wcs.wcs.crval[0] - 290.240574582) < 1e-12
