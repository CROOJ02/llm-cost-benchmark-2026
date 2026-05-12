# Day 13 Kickoff

Day 12 closed with the v1 writeup spine final in `analysis/out/headlines.md`. Day 13 layers charts on top of the Day 12 CSVs and addresses four carry-over TODOs from the analysis pass.

## Outstanding TODOs from Day 12

1. **Hardcoded "0.008 win margin" in Finding 4 of headlines.md.** Currently a string literal embedded in `analysis/04_headlines.py::render_markdown` (see the `# TODO` comment above the function body). Correct against the current data but will not auto-update if the customer_support cells shift. Replace with a computed value drawn from `analysis/out/category_breakdown.csv` (the `mean_canonical_score` field on the gpt-5.4 / customer_support / baseline vs batch rows).

2. **Model-naming convention is mixed across `headlines.md`.** Headline scalars use full IDs (`gpt-5.4-2026-03-05`, `gpt-5.4-mini-2026-03-17`) while Findings 1, 4, 5, and 6 use the friendly form (`gpt-5.4`, `gpt-5.4-mini`) via `friendly_model()`. Standardise to: **friendly names in narrative prose, full IDs in methodology references and CSV exports.** The headline scalars should read "gpt-5.4 + baseline" not "gpt-5.4-2026-03-05 + baseline". Update `render_markdown` to wrap both scalar model references with `friendly_model()`.

3. **Summarisation degrades similarly under output_cap (−0.067) and compression (−0.069) — close numbers, worth a footnote.** A reader could read the near-equality as a coincidence or as masking a structural effect. The mechanism is plausible: long-form summarisation outputs hit the output_cap *and* lose detail under compression in roughly the same way (both effectively reduce the model's output budget). Add a one-sentence footnote in Finding 3, framed as "summarisation's similar drops under output_cap and compression both reflect output-budget pressure, not a coincidence." Verify by inspecting `tier_1_status='truncated'` count on summarisation/output_cap before asserting.

4. **GPT-5.4 bit-for-bit batch determinism finding captured in project memory.** ✅ Done Day 12. Captured in operator-side project memory; not part of the public repo. Recorded so future cross-provider work can reference it without re-deriving. No further action.

## Day 13 deliverables — chart layer

Four chart scripts to add under `analysis/charts/` (read-only against the Day 12 CSVs, write PNG to `analysis/out/charts/`). Matplotlib-only, no seaborn dependency.

- **chart_a_cost_quality_scatter.py** — 16-point scatter, log-x cost, colour by provider, shape by lever, Pareto frontier connected by a line. Headline chart for the writeup.
- **chart_b_category_lever_heatmap.py** — 4×4 panel grid (one panel per task_category × lever, colour = Δcanon vs category baseline). Supports findings 2/3/4.
- **chart_c_reasoning_drill_4x4.py** — bar chart, model × lever, canon on primary axis, tier-1 pass rate on secondary. Supports finding 5.
- **chart_d_cost_sensitivity_strip.py** — strip plot, one strip per task_category, dots at each (model, lever) cell, near-best (≥90%) band shaded. Supports finding 6.

Each chart script: ≤ 100 lines, no DB access, regenerable in seconds.

## Acceptance criteria

Day 13 closes when:
- Four charts on disk under `analysis/out/charts/`
- Four TODOs above all resolved or explicitly deferred to Day 14+
- `headlines.md` regenerable with corrected win margin and unified naming
- Footnote on summarisation output-budget pressure landed
- Commit and push to main

No new data collection or DB writes on Day 13 — purely a presentation-layer pass.
