# Security policy

## Supported versions

Only the head of `deepseek-v41-tp2` and `glm-dsa-tp4` get security fixes. A fix reaches recipe users in the next engine release (for example `dsv41-2.0.0`). Older commits and images do not get fixes.

## Report a vulnerability

Do not open a public issue or pull request for a security problem.

1. Go to the [Security tab](https://github.com/Elevatormusic/TensorFold/security) of this repo.
2. Click **Report a vulnerability**. This opens a private report that only the maintainer can see.
3. Tell us the problem, the steps to see it, the files and the commit. Do not include real keys or tokens.

If the report form is not there, use the contact details on the GitHub profile of the maintainer, [@Elevatormusic](https://github.com/Elevatormusic).

The maintainer tries to reply in 7 days. You get a message when a fix is ready. You get credit if you want it.

## What is in scope

- The engine code on the fork branches, for example the server, the API key check, request parsing and the RDMA and NCCL links between the nodes.
- A secret, a machine name or a home path in a committed file.

A problem that is also in upstream TensorFold goes to [ashhart/TensorFold](https://github.com/ashhart/TensorFold) too. A problem in a recipe script goes to that recipe, for example [the DeepSeek recipe](https://github.com/Elevatormusic/DeepSeek-V4.1-Flash-1017-EXL3-TensorFold-2x-DGX-Spark/security). If you are not sure, report it here and we send it on.

## Safe use

- Serve on `127.0.0.1` unless you need network access. If you serve the LAN, use an API key.
- Do not commit keys or tokens.
