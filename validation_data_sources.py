"""
Bibliographic URLs and DOIs for launch-vehicle validation (paper Table tab:valrefs).

The IAU proceedings use a hand-formatted reference list without embedded URLs;
this module is the canonical traceability record for scripts and maintainers.
Used by validation_launch_vehicles.py and mission_dv.py.
"""

# Saturn V — Apollo statistics (GLOM, payload, translunar C3)
ORLOFF_SP_4029 = "https://history.nasa.gov/SP-4029/"

# Falcon Heavy — expendable GLOM and Mars payload (archived SpaceX page)
SPACEX_FALCON_HEAVY_ARCHIVE = (
    "https://web.archive.org/web/20180606120642/https://www.spacex.com/falcon-heavy"
)

# SLS Block 1 — NASA/MSFC reference guide (March 2022)
NASA_SLS_REFERENCE_GUIDE_2022 = (
    "https://www.nasa.gov/wp-content/uploads/2022/03/"
    "sls_reference_guide_2022_web.pdf"
)

# Atlas V 551 — gross lift-off mass (587,000 kg)
ASTRONAUTIX_ATLAS_V_551 = "http://astronautix.com/a/atlasv551.html"

# Atlas V 551 — escape payload at C3=0 (Table 1, NASA Launch Services Program tool)
SCHMIDT_2010_ACTA_ASTRONAUTICA_DOI = "https://doi.org/10.1016/j.actaastro.2009.07.037"

# N1 (1964) — draft project GLOM and LEO payload (never flown)
ASTRONAUTIX_N1_1964 = "http://astronautix.com/n/n11964.html"

# Electron — Payload User's Guide v7.0 (1 November 2022)
ROCKETLAB_ELECTRON_UG_V7 = (
    "https://rocketlabcorp.com/assets/Electron-Payload-User-Guide-7.0-v6.pdf"
)

# F-1 turbopump — Stangeland (1992) SAE paper
STANGELAND_1992_TURBOPUMPS_DOI = "https://doi.org/10.4271/921043"

# Human-readable index (keys match paper \\citealt{} labels where applicable)
DATA_SOURCES = {
    "Orloff2000": {
        "description": "Saturn V Apollo 8--17 GLOM, spacecraft mass, translunar C3",
        "url": ORLOFF_SP_4029,
    },
    "SpaceXFH": {
        "description": "Falcon Heavy expendable GLOM and payload-to-Mars",
        "url": SPACEX_FALCON_HEAVY_ARCHIVE,
    },
    "NASASLS2022": {
        "description": "SLS Block 1 GLOM and TLI payload",
        "url": NASA_SLS_REFERENCE_GUIDE_2022,
    },
    "AstronautixAtlas551": {
        "description": "Atlas V 551 gross lift-off mass (587,000 kg)",
        "url": ASTRONAUTIX_ATLAS_V_551,
    },
    "Schmidt2010": {
        "description": "Atlas V 551 escape payload at C3=0 (Acta Astronautica Table 1)",
        "url": SCHMIDT_2010_ACTA_ASTRONAUTICA_DOI,
    },
    "AstronautixN1": {
        "description": "N1 (1964) GLOM and LEO payload (draft, never flown)",
        "url": ASTRONAUTIX_N1_1964,
    },
    "RocketLabElectron": {
        "description": "Electron GLOM and LEO payload (User's Guide v7.0)",
        "url": ROCKETLAB_ELECTRON_UG_V7,
    },
    "stangeland1992turbopumps": {
        "description": "F-1 turbopump power (Sec. enginesize)",
        "url": STANGELAND_1992_TURBOPUMPS_DOI,
    },
}
