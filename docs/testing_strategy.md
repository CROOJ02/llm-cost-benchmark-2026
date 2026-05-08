Testing Strategy
Discipline
Every new component gets a smoke test before integration. Every integration point gets a pytest case. A full-pipeline dry-run runs at small scale before any production-scale matrix execution. Bugs surface cheap (pennies) rather than at scale (pounds).
This discipline applies to all Day 6+ work. The principle is asymmetric cost: catching a runner bug on Day 6 costs 30 minutes; catching the same bug mid-Day-7 costs hours plus £20-£40 of wasted API spend on corrupted result rows.
Four-Layer Testing Model
Testing is organised into four layers, each catching a different class of bug. All four are needed; each catches things the others can't.
Layer 1 — Unit tests (pytest)
Catches: logic errors in isolated functions.
Existing: 3 retry tests in tests/test_run_anthropic.py.
Day 6 additions:
Test: cost reservation under retry exhaustion. When a call hits 429 errors 6 times and exhausts retries, the runner writes an error row but accumulates no cost. The reservation made before the call must be released. Assertion: runs.cost_so_far_gbp is unchanged after retry-exhaustion compared to before. Catches the class of bug where reservations leak phantom cost into the cap accumulator.
Test: reservation release on any non-success exit path. Cost reservation must be released on any failure mode, not just retry exhaustion. Mock failure scenarios: DB error after API success, exception during result row insert, generic exception in run_one. Assertion: runs.cost_so_far_gbp is unchanged regardless of which failure path triggered. Catches the class of bug where reservations leak under unanticipated exception types.
Test: skip-if-exists across lever variants. A baseline result for prompt X must NOT cause a subsequent caching call for prompt X to be skipped. The skip composite key includes lever name. Assertion: after running prompt X baseline (1 row), running prompt X caching produces 3 new rows (baseline+write+read sequence). Catches the class of bug where the skip key is too coarse and silently suppresses lever calls.
Test: schema migration safety. Adding the batch_jobs table must not break existing queries against runs or results. Assertion: a representative query against runs and a representative query against results return expected shapes after migration. Catches the class of bug where schema changes break implicit assumptions held by working queries.
Layer 2 — Lever-level smoke tests
Catches: integration bugs between a single lever and the runner.
After each lever module is built (output_cap, batch, compression), run it against 1 prompt × 1 model in isolation before moving to the next lever. Each is a real API call — pennies, not pounds.
lever_output_cap smoke: cs-001 on Sonnet 4.6 with max_tokens=200. Verify result row has optimisation_lever='output_cap', optimisation_config includes the max_tokens value, response output_tokens ≤ 200, cost lower than baseline cs-001 already in DB.
lever_batch smoke (submit only): 2 prompts on Sonnet 4.6. Verify batch_jobs row created with status='submitted', batch_id populated, prompt_ids JSON correct. Lever returns within seconds (does not block waiting for batch completion). Retrieval is tested separately by the dry-run.
lever_compression smoke: sum-015 on Sonnet 4.6 with compression. Verify optimisation_config includes original_input_tokens and compressed_input_tokens both measured by Anthropic's count_tokens (not LLMLingua-2's BERT count). Compression ratio in the expected ~50% range. Cost lower than baseline sum-015.
Pause after each smoke test for review. If any fails, fix before moving to the next lever.
Layer 3 — Orchestrator integration tests (pytest with mocking)
Catches: phase-transition bugs and state-machine errors.
Built after the orchestrator lands. No real API calls — lever methods are mocked.
Test: phase transition state. Mock lever methods to return success. Run run_day_7() against a 2-prompt × 2-model fixture. Verify all four Day 7 phases execute in order, each phase logs to data/phase_log.jsonl with the expected structure, final state has expected results in DB and batch_jobs rows. Catches phase-transition bugs that would otherwise surface mid-Day-7.
Test: Day 8 idempotent re-entry. Mock batch retrieval to return completed results. Call run_day_8() twice. Verify the second call is a no-op (skip-if-exists prevents reprocessing), no duplicate rows in results, batch_jobs.status correctly tracks "already retrieved." Catches partial-failure recovery bugs.
Test: budget gate decision logic. Parameterised test with cost_so_far_gbp set to £100, £150, £200, £230, £270. For each, call decide_compression_tier() and verify the tier matches the PRD §9 Day 8 ladder:

£100 (£200 headroom) → full
£150 (£150 headroom) → 60-subset
£200 (£100 headroom) → 30-subset
£230 (£70 headroom) → operator-call
£270 (£30 headroom) → skip

Catches off-by-one errors at ladder boundaries.
Layer 4 — Full-pipeline dry-run
Catches: whatever the first three layers missed.
The integration test that proves end-to-end behaviour at small scale before production-scale execution.
Scope: 2 prompts × 4 models × 4 levers + caching engagement = ~32 calls.
Prompts: sum-015 and sum-020. Both clear caching thresholds; both work for output_cap, batch, and compression.
Models: Sonnet 4.6, Haiku 4.5, GPT-4o, GPT-4o-mini.
Levers: baseline, caching, output_cap, batch, compression.
Expected cost: ~£0.50.
Verifications (all five must pass):

Row count. ~50 rows in results (8 baseline + 18 caching where engaged + 8 output_cap + 8 from batch retrieval + 8 compression). 4 batch_jobs rows. Haiku records "caching unavailable" rather than caching result rows.
Cost accounting. cost_so_far_gbp matches sum of individual cost_usd values converted via GBP_USD_RATE, accurate to £0.001.
Engagement assertions. All caching write/read assertions pass per the existing engagement check logic. Compression rows show compressed_input_tokens < original_input_tokens.
No errors. No NULLs on critical fields (result_id, run_id, model_version, response_text, cost_usd). No retry exhaustion. No phase-log error events.
Phase log readable. data/phase_log.jsonl parses cleanly as line-delimited JSON, contains expected events for each phase (start, complete) with no orphaned events.

If any verification fails, debug before declaring Day 6 done.
Day 6 Done-Criterion
Day 6 is complete when all of the following hold:

Three lever modules exist: lever_output_cap.py, lever_batch.py, lever_compression.py
Each lever passed its Layer 2 smoke test
Orchestrator built per skeleton in runners/orchestrator.py, all NotImplementedError("Day 6 build") methods implemented
Seven new pytest cases pass: 4 unit-level (Layer 1) + 3 integration-level (Layer 3)
Full-pipeline dry-run completes successfully with all five verifications passing
Single Day 6 commit lands with all of the above

This is a real done-criterion: not "code exists" but "code works end-to-end at small scale."
Day 7 Risk Note
Even with all four testing layers passing, Day 7 introduces one variable not testable cheaply: scale. The dry-run is 32 calls; Day 7 baseline is 408 calls. Some failure modes only emerge at scale (rate-limit interactions, latency-tail patterns, transient errors).
Mitigation: Day 7's first phase is a warm-up — first 20 calls run at workers=2 with close monitoring. Cost tracking verified accurate at small N before ramping to full concurrency. If anything looks wrong in the first 20 calls, halt and debug. ~£2 of slow-start spend is cheap insurance for a 408-call run.
This is implemented as the orchestrator's run_day_7() first executing a \_warmup_phase() before transitioning to run_baseline() proper.
Retrospective Coverage Note
This testing strategy was formalised at the start of Day 6. Earlier work (Days 1-5) was tested with a less formal but adequate discipline: pause-and-review checkpoints at each build step, smoke tests against representative prompts, methodology review before commit.
The Day 6 dry-run functions as retrospective integration validation: it exercises the Day 4 Anthropic runner, the Day 5 OpenAI runner, and the Day 5 caching lever in combination with the new Day 6 levers and orchestrator. If the dry-run passes, the earlier components are validated end-to-end without separate retrospective test work.
When to Update This Document
This document is updated when:

A new component or lever is added to the benchmark (extend Layer 2)
A new orchestration phase is added (extend Layer 3)
A test catches a bug that should have been caught earlier (record the lesson and add to the relevant layer)

Days 7-14 will likely surface additions. Each addition gets recorded with date and rationale.
