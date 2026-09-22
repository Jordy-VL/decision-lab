# Modal job status

From the repository root in WSL/Linux:

```sh
./jobs/modal-status.sh
```

This read-only command lists apps and running containers as JSON using the active Modal profile and environment. It does not start or stop jobs. Stopped apps with zero tasks and an empty container list indicate no active compute in that environment. Persistent Volumes remain stored separately.

## Verified findings — 2026-09-22

- Profiling app `ap-xqLBFwGmHPi6oq9PAnzbgQ`: stopped at 18:45:27 Europe/Paris, zero tasks.
- Training app `ap-OO22BxiK87M5jisChJeEOI`: stopped at 19:31:30 Europe/Paris, zero tasks.
- The running-container query returned `[]`.
- The `decision-lab-profile` Volume still contains `hf`, `runs`, and `profile.json`. The profiling report also exists locally at `runs/modal-profile/profile.json`; the profiling job intentionally saved no trained checkpoint.
- Profiling logs are retained: steps 31 and 32 appear at 18:45:21 Europe/Paris, followed by completion at 18:45:24. A dashboard time filter ending at 18:44:37 excludes those final logs. Disable Live and select a historical range covering the run.
- The full training attempt was cancelled before its final checkpoint. See [the retained outcome and evidence](../docs/results/ce-20260922-170803.md); its cancellation cause remains unresolved.

Retrieve the profiling logs without dashboard time filters:

```sh
uv run modal app logs ap-xqLBFwGmHPi6oq9PAnzbgQ --timestamps
```

These observations describe the check on this date; rerun the status script for current state.
