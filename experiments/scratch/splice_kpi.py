import os
import shutil

here = os.path.dirname(os.path.abspath(__file__))
p = os.path.join(here, "..", "..", "dt_pruning", "reward.py")
shutil.copy(p, os.path.join(here, "reward_old_kpi.py.bak"))
L = open(p, encoding="utf-8").read().split("\n")
assert L[213].startswith("# ===") and "六个 KPI" in L[214], (L[213], L[214])
assert L[377].strip() == "}" and L[376].strip().startswith('"stability"'), (L[376], L[377])
new = open(os.path.join(here, "new_kpi_block.py"), encoding="utf-8").read().rstrip("\n").split("\n")
out = L[:213] + new + L[378:]
open(p, "w", encoding="utf-8", newline="\n").write("\n".join(out))
print("ok", len(L), "->", len(out))
