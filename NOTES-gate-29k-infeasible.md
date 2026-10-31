# Gate note (jimmy, 2026-10-31 07:20Z)

The 29K/5401-token lossless gate is INFEASIBLE on this 16GB HBM guest: eager
OOMs at prefill (verified in the t1073/t1074 runs, same guest, same 9B 4-bit
model). The standing substitute gate is 4096-tok (512 MB arena, 120/120
tokens) — that is the evidence on record for PR #2 (merged, main da9eeca).

If a 29K gate attempt OOMs at prefill, do not re-attempt: run the 4096-tok
gate instead and note the substitution in REPORT.md and the PR body.

Also: the brief says "14tok + 29K" — treat "29K" as "4096tok (substitute)".
