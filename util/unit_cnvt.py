"""
Convert atomic energy values between inverse centimeters, Rydbergs, ergs,
joules, and electronvolts, and derive transition energies from wavelengths.
"""

from scipy.constants import h, c, e


def convert_units(value, from_unit, to_unit):
    """
    Convert values between inverse centimeters and supported energy units.

    Supported conversions are ``pcm`` to or from ``ryd``, ``ergs``, ``J``,
    and ``eV``.

    Parameters
    ----------
    value : float or array-like
        Value or values to convert.
    from_unit : str
        Input unit: ``pcm``, ``ryd``, ``ergs``, ``J``, or ``eV``.
    to_unit : str
        Output unit: ``pcm``, ``ryd``, ``ergs``, ``J``, or ``eV``.

    Returns
    -------
    float or array-like
        Converted value or values.
    """
    if from_unit == 'pcm':
        if to_unit == 'J':
            conversion_factor = (h * c) * 1e2
            return value * conversion_factor
        elif to_unit == 'eV':
            conversion_factor = (h * c) / e * 1e2
            return value * conversion_factor
        elif to_unit == 'ergs':
            conversion_factor = (h * c) * 1e9  # 1 erg = 1e-7 J
            return value * conversion_factor
        elif to_unit == 'ryd':
            conversion_factor = (h * c) / (e * 13.605693) * 1e2
            return value * conversion_factor
        else:
            raise ValueError("Unsupported unit conversion from cm^-1.")

    if to_unit == 'pcm':
        if from_unit == 'J':
            conversion_factor = 1e-2 / (h * c)
            return value * conversion_factor
        elif from_unit == 'eV':
            conversion_factor = e / (h * c) * 1e-2
            return value * conversion_factor
        elif from_unit == 'ergs':
            conversion_factor = 1e-9 / (h * c)
            return value * conversion_factor
        elif from_unit == 'ryd':
            conversion_factor = (13.605693 * e) / (h * c) * 1e-2
            return value * conversion_factor
        elif from_unit == 'pcm':
            return value
        else:
            raise ValueError("Unsupported unit conversion to cm^-1.")


def wav_to_E(wav, to_unit):
    """
    Convert a wavelength in angstroms to the corresponding energy difference.

    Parameters
    ----------
    wav : float or array-like
        Wavelength in angstroms.
    to_unit : str
        Output energy unit: ``ryd``, ``ergs``, ``J``, or ``eV``.

    Returns
    -------
    float or array-like
        Energy difference in the requested unit. Unsupported units return the
        input wavelength unchanged.
    """
    if to_unit == 'J':
        return h * c / (wav * 1e-10)
    elif to_unit == 'eV':
        return (h * c / (wav * 1e-10)) / e
    elif to_unit == 'ergs':
        return h * c / (wav * 1e-10) * 1e7
    elif to_unit == 'ryd':
        return (h * c / (wav * 1e-10)) / (13.605693 * e)
    else:
        return wav


def main():
    import pandas as pd

    df = pd.DataFrame({
        'eV': [10.0, 20.5, 30.2, 40.8, 50.1]
    })
    print(convert_units(df['eV'], 'eV', 'pcm'))


if __name__ == "__main__":
    main()
