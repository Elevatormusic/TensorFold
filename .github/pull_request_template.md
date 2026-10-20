<!-- Read CONTRIBUTING.md first. Engine code targets deepseek-v41-tp2 or glm-dsa-tp4, not main. -->

## What this changes

<!-- One or two sentences, and the recipe issue it fixes if there is one. One change to a pull request. -->

## Type

<!-- Tick exactly one. The pr-template check adds the matching label. -->

- [ ] `type: fix`: a bug fix, or a fix after review findings
- [ ] `type: feature`: new code (a feature, tool or add-on)
- [ ] `type: docs`: docs only
- [ ] `type: results`: run results or a close-out
- [ ] `type: research`: a study or failure research, with no fix
- [ ] `type: spec`: a plan or spec before the work

## Receipt

<!-- A change to documentation or CI alone needs no receipt. -->

- Environment, with the TensorFold commit, PyTorch version, machine and GPU, checkpoint and revision:
- Exactness, from `tools/bench_concurrent.py --alone --serial` or `token_sha` against `"draft": false`:
- Decode tok/s before and after, from `tools/bench_openai.py`:
- Prefill tok/s and TTFT before and after, from `tools/prefill_cold.py`:
- Tests run, with pass, fail and skip counts:
- Not run, and why:

## Checklist

- [ ] Every token still goes through the lane rounds. No serial path, and nothing that needs drafts off.
- [ ] Drafted output equals `"draft": false`, a resumed prompt equals a fresh one, and concurrent equals solo.
- [ ] No precision traded for speed. If the bits change, the description says which sums change and why.
- [ ] Prompt processing is no slower than the last release.
- [ ] New tests fail before the change, pass after it, and skip cleanly without their dependency.
- [ ] No personal data, machine names, internal hosts or local paths. No AI attribution lines.

<details>
<summary>AI disclosure</summary>

AI: <model and size> via <harness>, e.g. "Qwen3-8B Q4 via Aider" or "Claude Opus 5.5 via Claude Code"
Checked: <what you did yourself, e.g. "read every line, ran tests">

</details>
