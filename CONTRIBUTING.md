# Contributing to this fork

This repo is a fork of [ashhart/TensorFold](https://github.com/ashhart/TensorFold). It holds the engine branches that the DGX Spark recipes pin. This page tells you where a change goes and what a pull request needs here.

Read the [Code of Conduct](CODE_OF_CONDUCT.md) first. To report a security problem, read [SECURITY.md](SECURITY.md).

## Where a change goes

| Change | Target |
|---|---|
| The DeepSeek-V4.1-Flash engine on two DGX Spark (TP=2) | branch `deepseek-v41-tp2` here |
| The GLM-5.3 engine on four DGX Spark (TP=4) | branch `glm-dsa-tp4` here |
| A change that helps every TensorFold user, not only these setups | upstream, [ashhart/TensorFold](https://github.com/ashhart/TensorFold) |
| This page, the templates and the CI of the fork | `main` here |

Do not open a pull request against `main` for engine code. `main` follows upstream and holds the repo files of the fork.

## Issues and questions

Issues are off in this fork. Report a problem with the recipe that uses the engine:

- DeepSeek-V4.1-Flash on two DGX Spark: [the recipe issues](https://github.com/Elevatormusic/DeepSeek-V4.1-Flash-1017-EXL3-TensorFold-2x-DGX-Spark/issues/new/choose). Use the bug report form and give the engine image id.
- GLM-5.3 on four DGX Spark (branch `glm-dsa-tp4`): this has no issue tracker yet. For now, open a post in [the DeepSeek recipe Discussions](https://github.com/Elevatormusic/DeepSeek-V4.1-Flash-1017-EXL3-TensorFold-2x-DGX-Spark/discussions), put `glm-dsa-tp4` in the title, and give the engine image id.
- A question: [the recipe Discussions](https://github.com/Elevatormusic/DeepSeek-V4.1-Flash-1017-EXL3-TensorFold-2x-DGX-Spark/discussions).

If the problem is in TensorFold itself and not in the fork, report it upstream.

## What a pull request needs

1. **Follow the engine rules.** The `CONTRIBUTING.md` on the target branch gives them: lanes, exact output, no precision traded for speed, prompt speed no slower, and the receipt. See the [deepseek-v41-tp2 copy](https://github.com/Elevatormusic/TensorFold/blob/deepseek-v41-tp2/CONTRIBUTING.md).
2. **One change in each pull request.** Do not mix a refactor with a change in behaviour.
3. **Give the receipt.** Use the standard serving terms: decode tok/s, prefill tok/s, TTFT, median, prefix cache. A speed claim needs before and after runs on the same machine. Say what you did not run.
4. **Fill in the template.** Tick exactly one type. The `pr-template` check adds the label. It fails if you tick no type, more than one type, or if the AI disclosure block is missing.
5. **Pass the checks.** CI runs the CPU tests, ruff, the workflow checks and a secret scan. Tests that need a GPU skip themselves.
6. **No personal data.** No machine names, home paths, internal hosts, public IP addresses, keys or tokens.

The maintainer reviews each pull request. An automatic review bot can also comment. Reply to each comment, or push a fix for it.

## Labels

| Label | Use it for |
|---|---|
| `type: fix` | a bug fix, or a fix after review findings |
| `type: feature` | new code: a feature, tool or add-on |
| `type: docs` | docs only |
| `type: results` | run results or a close-out |
| `type: research` | a study or failure research, with no fix |
| `type: spec` | a plan or spec before the work |

If your change needs two types, it is two pull requests.

## Releases

A release of this fork is an engine build that a recipe pins. Its name follows the recipe release that pins it: `dsv41-1.0.0` is the engine of the DeepSeek recipe `v1.0.0`. A new engine release comes with a recipe re-pin and a gate run on the hardware. The release page gives the commit, the image tag and the changes since upstream.

## Commits

- Keep your own authorship. Use your GitHub noreply address.
- No AI tool attribution lines, and no co-author trailers for tools.

## AI disclosure

If an AI helped, the block at the bottom of the pull request template says which one. If no AI helped, write `AI: none` in it.

```html
<details>
<summary>AI disclosure</summary>

AI: <model and size> via <harness>, e.g. "Qwen3-8B Q4 via Aider" or "Claude Opus 5.5 via Claude Code"
Checked: <what you did yourself, e.g. "read every line, ran tests">

</details>
```

This does not cause a rejection. It tells reviewers how carefully to check the work. Keep the empty lines after `<summary>` and before `</details>`.

## Documents

Write in [ASD-STE100 Simplified Technical English](https://www.asd-ste100.org/): short sentences, one instruction in each sentence, active voice.

## Licence

TensorFold is Apache-2.0 ([LICENSE](LICENSE), [NOTICE](NOTICE)). A contribution you submit is under that licence, as its section 5 says. Model weights keep their own licences.
