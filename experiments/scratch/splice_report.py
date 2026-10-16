"""e3_comprehensive.py：报告段整体重排。"""
import io

P = r"d:\PROJECTS\latex-workflow\experiments\scripts\e3_comprehensive.py"
NEW = '''    def test_rows(items):
        rows = [
            "| 指标 | 对手 | 均值差(DT-对手) | 95% CI | Holm p | 效应量 | 胜率 |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
        for key, label, digits in items:
            for policy, result in tests[key].items():
                rows.append(
                    f"| {label} | {policy} | {result['diff']:+.{digits}f} | "
                    f"[{result['ci'][0]:+.{digits}f}, {result['ci'][1]:+.{digits}f}] | "
                    f"{fmt_p(result['p_holm'])} | {result['effect']:+.2f} | "
                    f"{result['win']:.0%} |"
                )
        return rows

    lines = [
        f"# E3 多维综合基线（{args.stage}/{args.out_exp}/{args.split}，{len(trees)} 棵）",
        "",
        "主表为**同剪枝量比较**与**曲线 AUC**：各策略剪枝量不同（见第 2 节），终点 KPI 混杂“剪多少”，"
        "故只作补充。过程曲线屏蔽学习策略 STOP 至 0.30，评价选枝能力。"
        f"DT 终点口径 `{args.dt_tag}`（推理时 STOP logit 偏置，验证树上选定，测试前固定）。",
        "",
        "## 1. 主表：同剪枝量 KPI 与曲线 AUC",
        "",
        "同剪枝量 KPI：每棵树上取各策略过程曲线在 DT 停止剪枝比处的 KPI 综合（线性插值），DT 取自身终点。",
        "",
        "| 策略 | 同剪枝量KPI | AUC | KPI@0.10 | KPI@0.15 | KPI@0.20 | KPI@0.25 | KPI@0.30 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for policy in POLICIES:
        m = means[policy]
        lines.append(
            f"| {policy} | {m['kpi_matched']:.1f} | {m['curve_auc']:.1f} | "
            + " | ".join(f"{m[f'kpi_at_{int(q * 100):02d}']:.1f}" for q in GRID)
            + " |"
        )
    lines += [
        "",
        *test_rows([("kpi_matched", "同剪枝量KPI", 1), ("curve_auc", "曲线AUC", 1)]),
        "",
        "## 2. 补充：终点结果（各策略剪枝量不同）",
        "",
        "| 策略 | KPI综合 | 剪枝比 | 刀数 |",
        "|---|---:|---:|---:|",
    ]
    for policy in POLICIES:
        m = means[policy]
        lines.append(f"| {policy} | {m['kpi_sum']:.1f} | {m['ratio']:.3f} | {m['cuts']:.1f} |")
    lines += [
        "",
        "六项 KPI 变化：",
        "",
        "| 策略 | " + " | ".join(CN[k] for k in KPI_KEYS) + " |",
        "|---|" + "---:|" * len(KPI_KEYS),
    ]
    for policy in POLICIES:
        m = means[policy]
        lines.append(f"| {policy} | " + " | ".join(f"{m[k]:+.4f}" for k in KPI_KEYS) + " |")
    lines += ["", *test_rows([("kpi_sum", "终点KPI", 1)])]

    lines += [
        "",
        "## 3. 回报、选枝行为与结构安全",
        "",
        "回报只在学习方法之间比较；单位剪除量规则质量按规则分计算，专家和 BC 按规则选枝，与之同源。",
        "",
        "| 策略 | 回报 | αRq | βRi | γRc | 单位剪除量规则质量 | 约束触发 | 骨干枝/树 | 点/刀 | 分枝/刀 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for policy in POLICIES:
        m = means[policy]
        lines.append(
            f"| {policy} | {m['return']:.0f} | {m['rq']:.0f} | {m['ri']:.0f} | "
            f"{m['rc']:.1f} | {m['quality_density']:.2f} | "
            f"{m['constraint_rate']:.0%} | {m['main_cuts']:.2f} | {m['points_per_cut']:.1f} | "
            f"{m['branches_per_cut']:.2f} |"
        )
    lines += [
        "",
        *test_rows([("return", "回报", 0), ("quality_density", "单位剪除量规则质量", 2),
                    ("main_cuts", "骨干枝/树", 2)]),
        "",
        "不良枝动作占比在学习方法中达到 99%-100%，缺乏区分度，保留在 JSON 中。",
        "",
        "## 4. 停止行为（仅描述，不做优劣检验）",
        "",
        "训练数据的停止档按树序号轮转、与树状态无关，评估时专家等策略也不自然停止，"
        "没有可对照的“应停位置”。自主 STOP：学习策略自己选择停止；触顶：剪枝比达到硬上限 0.30。",
        "",
        "| 策略 | 平均剪枝比 | 剪枝比<0.25占比 | 自主STOP | 触顶 |",
        "|---|---:|---:|---:|---:|",
    ]
    for policy in POLICIES:
        m = means[policy]
        auto = "—" if policy not in LEARNED else f"{m['autonomous_stop']:.0%}"
        lines.append(
            f"| {policy} | {m['ratio']:.3f} | {m['under_025']:.0%} | {auto} | {m['hard_stop']:.0%} |"
        )

'''

with io.open(P, encoding="utf-8") as f:
    lines = f.read().split("\n")
start = next(i for i, l in enumerate(lines) if l.strip() == "lines = [")
end = next(i for i, l in enumerate(lines) if l.strip() == "report = {")
lines[start:end] = NEW.rstrip("\n").split("\n") + [""]
src = "\n".join(lines)
src = src.replace('''            ("kpi_sum", "Final KPI"),
            ("curve_auc", "Curve AUC"),
            ("quality_density", "Rule-quality density"),
            ("main_cuts", "Main-branch cuts"),
            ("stop_regret_kpi", "Unrealized KPI to 0.30"),''', '''            ("kpi_matched", "Matched-amount KPI"),
            ("curve_auc", "Curve AUC"),
            ("kpi_sum", "Final KPI (supplement)"),
            ("quality_density", "Rule-quality density"),
            ("main_cuts", "Main-branch cuts"),''')
src = src.replace("- 学习策略在剪枝比 0.30 前屏蔽 STOP，仅用于选枝过程曲线和停止遗憾",
                  "- 学习策略在剪枝比 0.30 前屏蔽 STOP，用于过程曲线与同剪枝量比较")
with io.open(P, "w", encoding="utf-8", newline="") as f:
    f.write(src)
print("ok")
