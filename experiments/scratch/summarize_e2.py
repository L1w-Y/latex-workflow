import json
import os

d = os.path.join(os.path.dirname(__file__), "..", "results", "dev", "e2_teacher", "val")
r = json.load(open(os.path.join(d, "teacher_student.json"), encoding="utf-8"))
s = json.load(open(os.path.join(d, "significance_dt.json"), encoding="utf-8"))
print(json.dumps(s, indent=0))
for k, v in r["profiles"].items():
    print(k, "R", round(v["total_reward"]["mean"]), "ratio", round(v["final_pruning_ratio"]["mean"], 3),
          "cuts", v["cuts_per_tree"]["mean"], "r/cut", round(v["reward_per_cut"]["mean"], 2),
          "rq", round(v["rq_per_cut"]["mean"], 2), "ri", round(v["ri_per_cut"]["mean"], 2),
          "rc", round(v["rc_per_cut"]["mean"], 3), "hit", round(v["bad_branch_hit_rate"], 3),
          "prot", round(v["protected_cut_rate"], 3), "len", round(v["cut_length"]["median"], 3),
          "pts", v["points_removed_per_cut"]["median"])
    print("   tags", {a: round(b, 2) for a, b in v["tag_rate"].items()})
for k, v in r["vs_expert"].items():
    print("VS", k, "jac", round(v["jaccard_by_branch"]["mean"], 3), "wjac",
          round(v["jaccard_by_points"]["mean"], 3), "div", v["first_divergence_step"])
    for sd in ["only_policy_side", "only_ref_side"]:
        x = v[sd]
        print("  ", sd, "n/tree", round(x["branches_per_tree"], 1), "pts/tree", round(x.get("points_per_tree", 0)),
              "len", {a: round(b, 3) for a, b in x["length"].items()}, "r", round(x["radius"]["median"], 4),
              "lvl", x["level"], "untag", round(x["untagged_rate"], 3))
        print("     tags", {a: round(b, 2) for a, b in x["tag_rate"].items()})
