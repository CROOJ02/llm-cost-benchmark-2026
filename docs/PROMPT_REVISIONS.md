# Prompt Revisions — All Categories

This document records prompt-design revisions applied during Days 1–5 of the benchmark, with rationale for each change. Engineering buyers reviewing methodology rigour can use this to verify that prompts were authored deliberately, with documented design decisions rather than as ad-hoc test cases.

The revisions span three prompt files — `customer_support.json`, `extraction.json`, and `rag_qa.json` — covering structural fixes (ambiguous JSON schemas, brittle rubrics) and realism upgrades (messy inputs, missing-field handling, contestable-answer notes). RAG QA needed no prompt content changes but received three scoring-methodology revisions.

---

## File 1: prompts/customer_support.json

### Revision 1.1 — Rewrite the system prompt (apply to all 20 prompts)

The current system prompt has a structurally ambiguous JSON schema description that may confuse weaker models. Replace the system prompt in all 20 prompts with this exact text:

```
You are a customer support classifier. Read the customer email and respond ONLY with a JSON object. The object must have exactly two fields. The "category" field must be exactly one of: "billing", "technical", "feature_request", "complaint", or "other". The "reply" field must be a 2-sentence acknowledgement that confirms receipt and indicates investigation without committing to specific outcomes. Output the JSON object with no other text.
```

This rewording is unambiguous about the schema shape (single object, two fields, allowed values for category, format of reply).

### Revision 1.2 — Replace 3 of the 7 easy prompts with messier real-customer-email patterns

The current easy prompts (cs-004, cs-007, cs-009) are too clean — perfect grammar, perfect punctuation, perfectly clear intent. Real customer emails don't look like this. Replace these three with prompts that reflect production reality.

**Replace cs-004 (clean credit card update) with a lowercase no-punctuation version.** Something like:

> "hey can someone help me update the card on file my old one expired last week account is customer@example.com — Customer B"

Keep complexity as easy and category as billing. Update the metadata note to describe the new pattern.

**Replace cs-007 (clean complaint about wait time) with a typo-and-fragments version.** Something like:

> "Honestly really fed up. Waited like 45 mins on chat yesteday and noone picked up?? Im paying customer too. customer@example.com - Customer A"

Keep complexity as easy and category as complaint. Update the metadata.

**Replace cs-009 (clean cancellation request) with an autocorrect-style mobile-typed version.** Something like:

> "Hi please cancle my subscription effective end of this billing period. Account is customer@example.com.\n\nSent from my phone — Customer A"

Keep complexity as easy and category as billing. Update the metadata.

These three replacements give the benchmark realistic exposure to production-style messy input without changing complexity distribution.

### Revision 1.3 — Diversify topic patterns

Current billing prompts (cs-001, cs-004, cs-009, cs-010, cs-016) all cluster around invoices and payment issues. Current technical prompts (cs-002, cs-005, cs-008, cs-011, cs-017) all cluster around bugs and login issues. Add some texture by replacing one billing prompt and one technical prompt with patterns currently uncovered.

**Replace cs-010 (proration question)** with a billing prompt about an unexpected trial expiry or plan downgrade. Something like:

> "Hi Acme Co support, my Acme Co trial seems to have ended early — I signed up on 1 April expecting 30 days free but I was charged on the 21st. Account customer@example.com. Can you check what happened? — Customer B"

Keep complexity as medium and category as billing.

**Replace cs-011 (date filter regression bug)** with a technical prompt about API rate limit confusion. Something like:

> "Hi Acme Co team, I'm getting 429 responses from the API even though I'm well below the documented 1000 requests/minute rate limit on the Pro plan. I'm seeing the limit kick in around 200-300 requests/minute. Is the documentation out of date or is there a separate per-endpoint limit I'm missing? Account customer@example.com. — Customer A"

Keep complexity as medium and category as technical.

These two replacements give the benchmark some coverage of patterns beyond the most obvious billing-and-bug clusters.

### Revision 1.4 — Add contestable-answer note to cs-019

cs-019 is a multi-issue email currently classified as "complaint" because of the "rough month" framing. The metadata should acknowledge this is contestable in production routing. Update its metadata notes field to:

> "Hard: a multi-issue email containing one billing, one technical, and one feature_request item. The customer's framing ('rough month', 'overall experience') makes the dominant theme dissatisfaction with the holistic experience rather than any individual item, so 'complaint' is the correct top-level classification. Note that production routing systems may legitimately route this to billing or technical first based on actionability — where models choose those classifications, treat as legitimately contestable rather than scoring as wrong."

This doesn't change the expected category but flags the case for the writeup's limitations section.

---

## File 2: prompts/extraction.json

### Revision 2.1 — Fix ext-013 null inference

The current expected output for ext-013 is `purchase_order: null`, but the input says "PO: (none provided)". A reasonable model could correctly output the literal string "(none provided)". The rubric is too strict.

Change the input from "PO: (none provided)" to "PO:" (with nothing after). This makes the null inference cleaner and removes ambiguity.

### Revision 2.2 — Add date anchor to ext-014

The prompt asks the model to interpret "the 30th of June this year" but the input doesn't anchor what year "this" is. Add a send date to the email. Change the start of the user message from:

> "Hi support,\n\nThis is Customer O..."

To:

> "Email received on 8 February 2026.\n\nHi support,\n\nThis is Customer O..."

Update the metadata note to: "tests parsing 'the 30th of June this year' to ISO 8601 using the explicit send date as the anchor for 'this year'."

### Revision 2.3 — Specify country format in ext-016

The current ext-016 schema says `country (string): the country they are based in`. The expected output is "United Kingdom" but a model could reasonably output "UK" or "GB" given the input says "UK based".

Update the schema definition for the country field to:

> "country (string): the country they are based in, expressed as the full English country name (e.g. 'United Kingdom', not 'UK' or 'GB'), or null if not present"

### Revision 2.4 — Restructure ext-018 applies_to as yes/no per category

The current `applies_to` field requires the model to extract verbatim natural-language phrases. Reasonable variation in extraction would score as wrong. Restructure as boolean fields per group.

Update the ext-018 schema to replace `applies_to (array of strings)` with three separate boolean fields:

```
- `applies_to_permanent_employees` (boolean): true if the policy applies to permanent employees, false if not
- `applies_to_fixed_term_contractors_six_months_plus` (boolean): true if the policy applies to fixed-term contractors on six-month-plus contracts, false if not
- `applies_to_short_term_agency_workers` (boolean): true if the policy applies to short-term agency workers, false if not
```

Update the expected output to:

```
{
  "applies_to_permanent_employees": true,
  "applies_to_fixed_term_contractors_six_months_plus": true,
  "applies_to_short_term_agency_workers": false,
  ... (other fields unchanged)
}
```

Update the metadata to note the change: "Restructured from free-text array to boolean-per-category to make rubric scoring deterministic."

### Revision 2.5 — Add 2 messier extraction prompts

Replace ext-002 and ext-006 with versions that have realistic noise.

**Replace ext-002 (clean contact block extraction)** with a version that has OCR-style character substitution and formatting noise:

> "Subject: Enquiry about enterpr1se tier\n\nHi there\n\nMy name is Customer B and Id like to know more about your enterprise tier\n\nBest\nCustomer B (Surname Two)\ncustomer.b@example.cc>m\n+44 77OO 9OO123\n\n--\nSent from my phone, please excuse typos"

Note the deliberate OCR-style errors: "1" instead of "i" in "enterprise", "cc>m" instead of "com" in email, capital "O" instead of "0" in phone number. Update the expected output to require the model to correct these to plausible values:

```
{
  "first_name": "Customer",
  "last_name": "Two",
  "email": "customer.b@example.com",
  "phone": "+44 7700 900123"
}
```

Update the metadata: "Tests OCR-style noise correction in email and phone extraction. Models should normalise '0' from 'O', correct 'cc>m' to 'com', and ignore the deliberate typo 'enterpr1se'."

**Replace ext-006 (clean meeting note extraction)** with a version that has formatting noise:

> "Internal note\n\n meeting: Q2 Planning Sync\n CHAIR:Customer F\nattendees present:6\n\n Notes to follow."

Note the inconsistent spacing, inconsistent casing of field labels, missing space after the colon. The expected output stays the same; the test is whether the model handles the noise.

Update the metadata: "Tests robustness to formatting noise — inconsistent whitespace, inconsistent label casing, missing spaces after colons. Expected output unchanged from clean version."

### Revision 2.6 — Add 2 prompts where required fields are genuinely absent

Add two new prompts (ext-021 and ext-022) to test missing-field handling. Expand to 22 total prompts rather than dropping existing ones. The extra cost is pennies and coverage gain is meaningful.

**ext-021** — invoice missing the due date entirely:

```json
{
  "prompt_id": "ext-021",
  "task_category": "extraction",
  "complexity": "medium",
  "input": {
    "system": "You extract structured data from unstructured text. Respond with ONLY a JSON object containing exactly these fields:\n- `invoice_number` (string): the invoice number, or null if not present\n- `customer_name` (string): the billed customer's name, or null if not present\n- `total` (number): the grand total in pounds as a number with no currency symbol, or null if not present\n- `invoice_date` (string): the invoice date in ISO 8601 format YYYY-MM-DD, or null if not present\n- `due_date` (string): the payment due date in ISO 8601 format YYYY-MM-DD, or null if not present\n\nDo not include any other text.",
    "user": "ACME CO LTD\n\nInvoice no. INV-2026-0512\n\nBill to: Customer V\n\nIssued: 15 March 2026\n\nProfessional services .................. £750.00\n\nTotal due ............................... £750.00\n\nPlease pay promptly."
  },
  "scoring": {
    "tier_1_deterministic": {
      "expected": {
        "invoice_number": "INV-2026-0512",
        "customer_name": "Customer V",
        "total": 750.0,
        "invoice_date": "2026-03-15",
        "due_date": null
      }
    }
  },
  "metadata": {
    "input_tokens_approx": 70,
    "notes": "Tests missing-field handling — the invoice has no explicit due date and 'pay promptly' is too vague to infer one. Models should output null rather than inventing a date."
  }
}
```

**ext-022** — lead form where company isn't stated:

```json
{
  "prompt_id": "ext-022",
  "task_category": "extraction",
  "complexity": "medium",
  "input": {
    "system": "You extract structured data from unstructured text. Respond with ONLY a JSON object containing exactly these fields:\n- `lead_name` (string): the prospective customer's name, or null if not present\n- `email` (string): the email address, or null if not present\n- `company` (string): the company name, or null if not stated\n- `team_size` (integer): the size of their team, or null if not stated\n\nDo not include any other text.",
    "user": "Hi,\n\nI'm Customer W and I'd love to learn more about your platform. We have around 15 people on the team and we're looking for something better than what we have now.\n\nReach me at customer.w@example.com.\n\nThanks!"
  },
  "scoring": {
    "tier_1_deterministic": {
      "expected": {
        "lead_name": "Customer W",
        "email": "customer.w@example.com",
        "company": null,
        "team_size": 15
      }
    }
  },
  "metadata": {
    "input_tokens_approx": 60,
    "notes": "Tests missing-field handling — the email domain is example.com which is not a real company. Models should output null for company rather than inferring from the email domain."
  }
}
```

---

## File 3: prompts/rag_qa.json (no prompt content changes — scoring methodology only)

The rag_qa.json file content is well-designed. The hard prompts (rag-018 postmortem, rag-020 security advisory, rag-016 sustainability) are excellent test cases. Three scoring methodology revisions are needed; none require changing the prompt JSON itself. Length question is dropped — contexts are within production-realistic range.

### Revision 3.1 — Loosen the `supporting_sentences` rubric (general rule)

The current rubric is too strict on which sentences count as "supporting." When the question's subject (the entity being asked about) appears in one sentence and the answer (the predicate) appears in another, both citation patterns should be acceptable.

Example from rag-001:

- Question: "In what year was Brightline Logistics founded?"
- Sentence [1] establishes the company exists
- Sentence [4] states the founding year (2014)
- Currently expects `supporting_sentences: [4]`
- A model citing `[1, 4]` is more rigorous, not wrong

**Apply this as a general scoring rule, not a per-prompt fix.** Update PRD Section 7 to specify:

> When evaluating `supporting_sentences`, the rubric accepts the minimal citation set that supports the answer, plus any superset that includes additional sentences explicitly establishing the question's subject or context. A model citing `[answer_sentence]` and a model citing `[subject_sentence, answer_sentence]` should both score 1.0 on supporting_sentences. A model citing unrelated sentences or missing the answer sentence scores 0.0.

This applies to all 20 prompts in rag_qa.json. No changes needed to the prompt files themselves; only to how scoring evaluates `supporting_sentences` matches.

### Revision 3.2 — Split scoring responsibility for the `answer` field

Currently both Tier 1 (deterministic) and Tier 2 (judge) appear to score the `answer` field. This conflates two different scoring approaches and produces brittle Tier 1 matches against verbatim text.

**The correct split:**

- **Tier 1 scores `supporting_sentences` only.** This is a deterministic integer array comparison (with the loosening from Revision 3.1 above). Easy to implement, easy to verify.
- **Tier 2 scores the `answer` field entirely.** Free-text answers go to the dual-judge blind evaluation per the PRD Section 7. Judges score against the `tier_2_judge.criteria` already in each prompt.

The `tier_1_deterministic.expected.answer` field becomes informational (used by the judges as the reference answer) rather than scored automatically.

**Update PRD Section 7** to clarify this split:

> For RAG Q&A, Tier 1 deterministic scoring covers only the `supporting_sentences` integer array (per Revision 3.1's general rule). The `answer` text field is scored entirely by the Tier 2 dual-judge layer against the prompt's `tier_2_judge.criteria`. The expected answer in `tier_1_deterministic.expected.answer` serves as the reference answer provided to judges, not as a verbatim string-match target.

This affects all 20 RAG prompts. No changes to the prompt content needed; only to scoring implementation logic.

### Revision 3.3 — Partial credit guidance for Tier 2 judges (forward-looking)

Some prompts test multiple facts in a single answer. For example:

- rag-019: "Which property recorded the highest customer satisfaction score, and what was the score?" — model could correctly name the property but miss the score, or vice versa
- rag-008: "What is the current production rate of Line C, and what was it before the 2024 retrofit?" — two facts required
- rag-015: "What is the recovery time objective for a manual failover, and what is the corresponding recovery point objective?" — two facts required

**Add explicit partial-credit guidance to the Tier 2 judge prompt template.** The judges should:

- Score 1.0 when all facts in the criteria are present and correct
- Score 0.5-0.7 when the answer covers part of the criteria correctly but is missing one component
- Score 0.0-0.3 when the answer is wrong or misleading on the central facts
- Use the full 0.0-1.0 range, not just the endpoints

**Update the judge prompt template** (which Claude Code will write in Day 10) to include this guidance explicitly. The instruction should look like:

> Score the response on a 0.0 to 1.0 scale. If the response satisfies all criteria fully and accurately, score 1.0. If the response covers part of the criteria correctly but is missing one or more components, score 0.5 to 0.7 depending on how much is covered. If the response is wrong on the central facts or misleading, score 0.0 to 0.3. Use intermediate values where appropriate; do not collapse to only 0 or 1.

This is a forward-looking change for the Day 10 judge implementation, not a prompt-file change.

---

## Revision scope summary

- `customer_support.json` — revisions 1.1 through 1.4 (system-prompt rewrite, 3 messier easy prompts, 2 diversified topic patterns, contestable-answer note on cs-019).
- `extraction.json` — revisions 2.1 through 2.6 (rubric fixes on ext-013/014/016/018, 2 messier prompts replacing ext-002/006, 2 new missing-field prompts at ext-021/022; file expanded from 20 to 22 prompts).
- `rag_qa.json` — revisions 3.1 through 3.3 (no prompt content changes; scoring-methodology revisions only — `supporting_sentences` rubric loosened, Tier-1 / Tier-2 split for the `answer` field, partial-credit guidance for Tier-2 judges).

---

## Design discipline for summarisation prompts

The summarisation category (added after the revisions above) follows the same design discipline as the prior categories:

- 20 prompts split 7 easy / 7 medium / 6 hard.
- All synthetic data (no real names, real companies, real public figures, real events).
- Production-realistic input documents (1500–3000 word range per PRD Section 3, Category 4).
- Easy: clearly structured documents with obvious main points.
- Medium: documents requiring synthesis across multiple sections.
- Hard: documents with subtle main points, ambiguous structure, or content that invites hallucination.
- Tier 2 only scoring (no Tier 1 deterministic — summarisation has no single right answer).
- Tier 2 criteria specify what the summary must cover, what it must not contain, and any specific facts/numbers that must be accurate.

**Two specific concerns shaped the summarisation prompt set:**

1. **Hallucination risk.** Summarisation is the category most likely to produce hallucinated facts. At least 3–5 of the prompts include "tempting" content that could be misremembered or fabricated — a number that's stated once and easy to misquote, a name introduced briefly, a date that's adjacent to other dates. The Tier 2 criteria explicitly call out the specific hallucinations to watch for.

2. **Length/coverage trade-off.** A 3-bullet summary of a 3000-word document is severely compressed. Different models choose different bullet structures. Some lead with the most important facts; some follow document order. The Tier 2 criteria specify the _facts_ the summary must convey, not the _structure_ — any 3-bullet structure that covers the required facts is acceptable.

---

## Why these revisions matter

Customer support had two structural issues — ambiguous JSON schema description and over-clean prompts — and one realism issue (over-narrow topic distribution). Fixing them tightens the rubric and makes the benchmark more representative of production patterns.

Extraction had four rubric issues (ext-013, ext-014, ext-016, ext-018) where reasonable model outputs would be scored as wrong, and two realism issues (no messy inputs, no missing fields). Fixing them prevents the benchmark from generating noisy scoring data and adds coverage for production patterns that currently aren't tested.

RAG Q&A had no prompt content issues — the 20 prompts as written are well-designed and within production-realistic length. Three scoring methodology revisions tighten how the data gets scored without changing what's tested. The split between Tier 1 (supporting_sentences only) and Tier 2 (answer text) is the most important fix — without it, Tier 1 scoring would be brittle against free-text answers.

All revisions together are 60-90 minutes of work and remove the methodological issues that would otherwise generate noisy data in week 2 of the benchmark run.
