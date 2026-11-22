# Local testing logs — November 21, 2026

Snapshots captured for the team before committing the launcher, LAN dashboard
support, optional separator setup, and standalone Falcon comparison harness.
Active logs continue to be written under ignored `.tools/startup/`; these copies
are static and will not change when the app is started again.

- `initial/`: original capture/dashboard session before live separation was added.
- `separation-test/`: startup and session logs after pulling the teammate's live
  separation and phone-control updates through commit `248a423`.
- `smoke-separator.json`: real ReSepFormer CPU loading and generated-input smoke
  test, approximately 80.6 ms for one measured 1.5-second window. This is not a
  sustained streaming benchmark or evidence of speech isolation quality.

Uvicorn writes informational output to stderr, so `backend-error.log` contains
normal startup messages as well as warnings. The separation session includes
skipped-hop diagnostics; inspect the counters and timing values together rather
than interpreting the log's `>` wording as an independently checked comparison.

Verification completed before this commit: 22 backend tests, 50 offline tests,
3 PCM playback tests, TypeScript, ESLint, Next.js production build, Android debug
assembly/unit tests/lint, and launcher startup/shutdown. Logs here are runtime
snapshots, not a full transcript of those commands. Falcon remains outside the
live app; the unfinished live integration draft is not included.
