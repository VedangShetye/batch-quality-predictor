"""
build_dashboard.py
Generates a single self-contained HTML dashboard from the scored batches --
double-click to open in any browser, no server needed. This is the visual,
presentable face of the pipeline for a non-technical viewer (or an interviewer).
"""

import pandas as pd
import json
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_DIR / "scored_batches.csv")
df = df.sort_values("fail_probability", ascending=False)

total = len(df)
flagged = int(df["flagged_for_review"].sum())
high = int((df["risk_level"] == "High").sum())
medium = int((df["risk_level"] == "Medium").sum())
low = int((df["risk_level"] == "Low").sum())

batches_json = json.dumps(df.to_dict(orient="records"))
risk_colors = {"High": "#C0504D", "Medium": "#E8A33D", "Low": "#4C9A6A"}

rows_html = ""
for _, row in df.iterrows():
    color = risk_colors[row["risk_level"]]
    flag_badge = (f'<span class="badge flag">Flagged</span>' if row["flagged_for_review"]
                  else '<span class="badge ok">-</span>')
    rows_html += f"""
    <tr>
      <td>{row['batch_id']}</td>
      <td><div class="bar-cell"><div class="bar" style="width:{row['fail_probability']}%;
          background:{color}"></div><span>{row['fail_probability']:.1f}%</span></div></td>
      <td><span class="badge" style="background:{color}22;color:{color}">{row['risk_level']}</span></td>
      <td>{flag_badge}</td>
      <td>{row['machine_id']}</td>
      <td>{row['operator_shift']}</td>
      <td>{row['mixing_time_min']:.1f}</td>
      <td>{row['temperature_C']:.1f}</td>
      <td>{row['humidity_pct']:.1f}</td>
    </tr>"""

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Batch Quality Risk Dashboard</title>
<style>
  :root {{
    --bg: #f7f8fa; --card: #ffffff; --text: #1f2430; --muted: #6b7280;
    --border: #e5e7eb; --accent: #4472C4;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
    background: var(--bg); color: var(--text); margin: 0; padding: 32px;
  }}
  .header {{ margin-bottom: 24px; }}
  .header h1 {{ margin: 0 0 4px 0; font-size: 24px; }}
  .header p {{ margin: 0; color: var(--muted); font-size: 13px; }}
  .cards {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 28px; }}
  .card {{
    background: var(--card); border: 1px solid var(--border); border-radius: 10px;
    padding: 18px 20px;
  }}
  .card .label {{ font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: .04em; }}
  .card .value {{ font-size: 28px; font-weight: 600; margin-top: 6px; }}
  .card.high .value {{ color: #C0504D; }}
  .card.medium .value {{ color: #E8A33D; }}
  .card.low .value {{ color: #4C9A6A; }}
  table {{
    width: 100%; border-collapse: collapse; background: var(--card);
    border: 1px solid var(--border); border-radius: 10px; overflow: hidden;
    font-size: 13px;
  }}
  thead th {{
    background: var(--accent); color: white; text-align: left; padding: 10px 12px;
    font-weight: 600; font-size: 12px; text-transform: uppercase; letter-spacing: .03em;
  }}
  tbody td {{ padding: 9px 12px; border-top: 1px solid var(--border); }}
  tbody tr:nth-child(even) {{ background: #fafbfc; }}
  .bar-cell {{ position: relative; background: #eef0f3; border-radius: 4px; height: 18px; width: 110px; }}
  .bar {{ position: absolute; left: 0; top: 0; height: 100%; border-radius: 4px; opacity: .85; }}
  .bar-cell span {{ position: relative; z-index: 1; font-size: 11px; padding-left: 6px; line-height: 18px; }}
  .badge {{ padding: 3px 9px; border-radius: 20px; font-size: 11px; font-weight: 600; }}
  .badge.flag {{ background: #C0504D22; color: #C0504D; }}
  .badge.ok {{ color: var(--muted); }}
  .footnote {{ margin-top: 20px; font-size: 11.5px; color: var(--muted); }}
</style>
</head>
<body>

  <div class="header">
    <h1>Batch Quality Risk Dashboard</h1>
    <p>Generated {datetime.now().strftime('%d %B %Y, %H:%M')} &middot; Prototype AI/ML pipeline &middot; synthetic data demonstration</p>
  </div>

  <div class="cards">
    <div class="card"><div class="label">Batches Scored</div><div class="value">{total}</div></div>
    <div class="card high"><div class="label">High Risk</div><div class="value">{high}</div></div>
    <div class="card medium"><div class="label">Medium Risk</div><div class="value">{medium}</div></div>
    <div class="card low"><div class="label">Low Risk</div><div class="value">{low}</div></div>
  </div>

  <table>
    <thead>
      <tr>
        <th>Batch ID</th><th>Fail Probability</th><th>Risk Level</th><th>Status</th>
        <th>Machine</th><th>Shift</th><th>Mixing (min)</th><th>Temp (&deg;C)</th><th>Humidity (%)</th>
      </tr>
    </thead>
    <tbody>
      {rows_html}
    </tbody>
  </table>

  <p class="footnote">
    Risk = model-estimated probability of QC failure based on process conditions
    (mixing time, temperature, humidity, moisture, machine, shift, raw material lot age).
    "Flagged" batches exceed the model's tuned decision threshold and are recommended
    for manual review before proceeding. Built on synthetic data as a pipeline demonstration.
  </p>

</body>
</html>
"""

out_path = REPORTS_DIR / "dashboard.html"
out_path.write_text(html, encoding="utf-8")
print(f"Dashboard saved -> reports/dashboard.html (open this file in any browser)")
