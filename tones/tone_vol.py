def _tone_vol(v, safe_max=0.5):
    """
    Backwards compatible:
      - old style: 0.0..1.0   (1.0 == 100%)
      - new style: 0..100     (100 == 100%)
      - strings:   "50", "50%"
    Then maps 100% -> safe_max (your old 0.5).
    """
    if v is None:
        return safe_max

    if isinstance(v, str):
        s = v.strip()
        if s.endswith("%"):
            try:
                pct = float(s[:-1])
            except ValueError:
                return safe_max
            pct = max(0.0, min(100.0, pct))
            return (pct / 100.0) * safe_max
        try:
            v = float(s)
        except ValueError:
            return safe_max

    try:
        v = float(v)
    except (TypeError, ValueError):
        return safe_max

    # interpret <=1.0 as normalized; >1.0 as percent
    pct = (v * 100.0) if v <= 1.0 else v
    pct = max(0.0, min(100.0, pct))
    return (pct / 100.0) * safe_max
