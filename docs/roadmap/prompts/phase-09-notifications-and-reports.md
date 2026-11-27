# Phase 9 notifications and reports launch prompt

Implement only Phase 9, Notifications and Reports.

Read `AGENTS.md`, `requirements.txt`, the stack/testing skill,
`docs/roadmap/README.md` and `docs/roadmap/STATUS.md`. Require recorded explicit
user acceptance of all previous phases and an unlocked Phase 9. This prompt
does not bypass any missing phase approval.

Audit the notification settings and existing CSV/reporting code. Reuse Phase 5's
reconciled ledger, metrics and generated reports; do not calculate conflicting
P&L definitions in delivery code.

Implement actual OS desktop notifications, Telegram delivery, Discord/Slack
webhooks, timezone-aware daily P&L/drawdown/accuracy reports, Google Sheets
delivery and CSV export. Define decision accuracy and its eligible observations
explicitly; win rate is not automatically prediction accuracy. A NiceGUI toast,
saved token field or placeholder test message does not prove external delivery.

Provide configuration validation, read-only connection/status checks where
supported, secure credential handling, redacted logs, retries, rate limits,
idempotency, delivery state and useful failure messages. A provider acceptance
response is not a guarantee that a recipient read the message. Test the observable
delivery stages without overstating them.

Use only user-designated test destinations with explicit sending authorization.
Ask for a missing destination/authorization when actual delivery validation needs
it, while implementing and testing independent code. Do not send to collaborators,
public channels or unspecified accounts. Do not overwrite an existing spreadsheet
or export path outside the approved scope.

Persist schedules and handle timezone/DST boundaries, restarts, missed runs and
bounded catch-up without duplicate daily reports. Local machine downtime must
be represented honestly. Validate sample reports and actual authorized test
deliveries for every advertised transport; missing integration evidence stays
pending rather than receiving a success label.

Stay on the current branch and preserve all accepted phases. This prompt does
not activate production trading, create/switch branches, commit, push or open a
PR unless separately requested.

Write the claim ledger, transport-specific tests, authorized delivery evidence,
sample reports and operator instructions under `docs/roadmap/evidence/phase-09/`;
update `STATUS.md`. Mark READY FOR USER REVIEW only when implementation and
required validation are complete. Otherwise list concrete gaps. Do not declare
Phase 9 or the original nine-phase roadmap accepted until I explicitly accept
Phase 9. The extended roadmap remains separate: only after all nine acceptances
may Review R0 be unlocked. Do not run it without my request/prompt, and do not
begin Phase 10 before that review passes and receives my explicit acceptance.
