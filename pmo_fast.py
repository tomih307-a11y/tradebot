"""Compare PMO settings on the same rules: default (35,20,10) vs fast (4,20,10). Rules as in pmo_only.py (15m zero-touch up, larger-frame PMO filters, exits E1/E2, stop 1.5xATR, 24h, costs 0.1%)."""
import os
import sys
import scanner as S

L1 = int(os.environ.get("L1", "35"))
_orig = S.pmo
S.pmo = lambda c, l1=L1, l2=20, sg=10: _orig(c, l1, l2, sg)
print(f"######## PMO ({L1}, 20, 10)")
import pmo_only                                                   # runs main() at import with the patched PMO
