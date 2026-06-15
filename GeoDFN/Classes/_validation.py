
VALID_LENGTH_PDFS = ('Log-Normal', 'Power-law', 'Constant', 'Exponential')
VALID_SPATIAL_PDFS = ('Log-Normal', 'Power-law', 'Uniform')
VALID_ORIENTATION_PDFS = ('Von-Mises', 'Uniform', 'Constant')
VALID_APERTURE_METHODS = ('constant', 'subLinear', 'Barton-Bandis', 'Lepillier')
VALID_BUFFER_METHODS = ('constant', 'linearRelationshipLength')

_LENGTH_REQUIRED_KEYS = {
    'Log-Normal':   ('mu', 'sigma', 'Lmin', 'Lmax'),
    'Power-law':    ('alpha', 'Lmin', 'Lmax'),
    'Constant':     ('L',),
    'Exponential':  ('lambda', 'Lmin', 'Lmax'),
}
_SPATIAL_REQUIRED_KEYS = {
    'Log-Normal':  ('mu', 'sigma', 'max distance'),
    'Power-law':   ('alpha', 'min distance', 'max distance'),
    'Uniform':     ('max distance',),
}
_ORIENTATION_REQUIRED_KEYS = {
    'Von-Mises': ('loc', 'kappa'),
    'Uniform':   ('thetaMin', 'thetaMax'),
    'Constant':  ('theta',),
}
_APERTURE_REQUIRED_KEYS = {
    'constant':      ('aperture',),
    'subLinear':     ('scalingCoefficient', 'scalingExponent'),
    'Barton-Bandis': ('JRC', 'JCS', 'sigma_Hmax', 'sigma_c', 'strike'),
    'Lepillier':     ('S_Hmax', 'S_hmin', 'E', 'nu', 'strike'),
}

_SET_REQUIRED_KEYS = (
    'I',
    'fractureLengthPDF', 'fractureLengthPDFParams',
    'spatialDistributionPDF', 'spatialDistributionPDFParams',
    'orientationDistributionPDF', 'orientationDistributionPDFParams',
    'bufferZone',
)


def validate_inputs(domain_x, domain_y, sets, aperture_params, num_realizations):
    _check_domain(domain_x, domain_y)
    _check_realizations(num_realizations)
    _check_aperture(aperture_params)
    if not sets:
        raise ValueError("'sets' must contain at least one fracture set.")
    for i, s in enumerate(sets):
        _check_set(s, index=i)


def validate_inputs_with_seed(domain_x, domain_y, sets, aperture_params, num_realizations):
    validate_inputs(domain_x, domain_y, sets, aperture_params, num_realizations)
    for i, s in enumerate(sets):
        if 'seed' not in s:
            raise ValueError(f"Set {i + 1} is missing required key 'seed'.")
        seed = s['seed']
        if 'X' not in seed or 'Y' not in seed:
            raise ValueError(f"Set {i + 1} 'seed' must have keys 'X' and 'Y'. Got: {list(seed.keys())}")
        if not (0 <= seed['X'] <= domain_x):
            raise ValueError(
                f"Set {i + 1} seed X={seed['X']} is outside the domain [0, {domain_x}]."
            )
        if not (0 <= seed['Y'] <= domain_y):
            raise ValueError(
                f"Set {i + 1} seed Y={seed['Y']} is outside the domain [0, {domain_y}]."
            )


def _check_domain(domain_x, domain_y):
    if not isinstance(domain_x, (int, float)) or domain_x <= 0:
        raise ValueError(f"domainLengthX must be a positive number. Got: {domain_x!r}")
    if not isinstance(domain_y, (int, float)) or domain_y <= 0:
        raise ValueError(f"domainLengthY must be a positive number. Got: {domain_y!r}")


def _check_realizations(n):
    if not isinstance(n, int) or n < 1:
        raise ValueError(f"numOfRealizations must be a positive integer. Got: {n!r}")


def _check_aperture(params):
    if 'method' not in params:
        raise ValueError(
            f"apertureCalculationParameters is missing required key 'method'. "
            f"Valid options: {VALID_APERTURE_METHODS}"
        )
    method = params['method']
    if method not in VALID_APERTURE_METHODS:
        raise ValueError(
            f"Unknown aperture method '{method}'. "
            f"Valid options: {VALID_APERTURE_METHODS}"
        )
    _check_required_keys(params, _APERTURE_REQUIRED_KEYS[method],
                         context=f"apertureCalculationParameters (method='{method}')")


def _check_set(s, index):
    label = f"Set {index + 1}"
    _check_required_keys(s, _SET_REQUIRED_KEYS, context=label)

    if not isinstance(s['I'], (int, float)) or s['I'] <= 0:
        raise ValueError(f"{label}: 'I' (fracture intensity) must be a positive number. Got: {s['I']!r}")

    length_pdf = s['fractureLengthPDF']
    if length_pdf not in VALID_LENGTH_PDFS:
        raise ValueError(
            f"{label}: Unknown 'fractureLengthPDF' value '{length_pdf}'. "
            f"Valid options: {VALID_LENGTH_PDFS}"
        )
    _check_required_keys(s['fractureLengthPDFParams'], _LENGTH_REQUIRED_KEYS[length_pdf],
                         context=f"{label} fractureLengthPDFParams (PDF='{length_pdf}')")

    spatial_pdf = s['spatialDistributionPDF']
    if spatial_pdf not in VALID_SPATIAL_PDFS:
        raise ValueError(
            f"{label}: Unknown 'spatialDistributionPDF' value '{spatial_pdf}'. "
            f"Valid options: {VALID_SPATIAL_PDFS}"
        )
    _check_required_keys(s['spatialDistributionPDFParams'], _SPATIAL_REQUIRED_KEYS[spatial_pdf],
                         context=f"{label} spatialDistributionPDFParams (PDF='{spatial_pdf}')")

    orient_pdf = s['orientationDistributionPDF']
    if orient_pdf not in VALID_ORIENTATION_PDFS:
        raise ValueError(
            f"{label}: Unknown 'orientationDistributionPDF' value '{orient_pdf}'. "
            f"Valid options: {VALID_ORIENTATION_PDFS}"
        )
    _check_required_keys(s['orientationDistributionPDFParams'], _ORIENTATION_REQUIRED_KEYS[orient_pdf],
                         context=f"{label} orientationDistributionPDFParams (PDF='{orient_pdf}')")

    _check_buffer_zone(s['bufferZone'], label)


def _check_buffer_zone(bz, label):
    if 'method' not in bz:
        raise ValueError(f"{label} bufferZone is missing required key 'method'. Valid options: {VALID_BUFFER_METHODS}")
    if bz['method'] not in VALID_BUFFER_METHODS:
        raise ValueError(
            f"{label} bufferZone method '{bz['method']}' is not valid. "
            f"Valid options: {VALID_BUFFER_METHODS}"
        )
    if 'constant' not in bz:
        raise ValueError(f"{label} bufferZone is missing required key 'constant'.")


def _check_required_keys(d, required_keys, context):
    missing = [k for k in required_keys if k not in d]
    if missing:
        raise ValueError(
            f"{context} is missing required key(s): {missing}. "
            f"Provided keys: {list(d.keys())}"
        )
