"""World Cup statistical-irregularity toolkit.

Auto-fetches men's FIFA World Cup data (1930-2022) and provides detectors that surface
statistical irregularities worth looking into. See the package modules:

- ``fetch``      : download + cache raw data, return tidy DataFrames
- ``elo``        : chronological Elo strength engine + strength-of-schedule
- ``models``     : goals model (Poisson / Dixon-Coles) for score-anomaly residuals
- ``referees``   : referee card/discipline outlier detection
- ``paths``      : per-team "easy path" / seeding-luck detection
- ``leadership`` : FIFA-president-era correlation (exploratory)
- ``anomalies``  : unified anomaly scoring + ranked "look into this" table
"""

__version__ = "0.1.0"
