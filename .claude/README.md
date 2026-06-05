# Project Claude Assets

This directory vendors the project-specific Claude Code skills needed to
continue the paper-to-reasoning workflow on another machine:

- `skills/paper-to-reasoning`
- `skills/paper-to-reasoning-audit`

The full user-level `~/.claude` directory is intentionally not vendored here. It
contains global history, caches, plugin state, local settings, and possible
machine/account-specific data.

The workflow also expects the OpenAI Codex Claude Code plugin to be installed on
the target machine when running the independent Codex gate. The skill looks for
the companion script at:

```bash
~/.claude/plugins/cache/*/codex/*/scripts/codex-companion.mjs
```

If that path is missing on the target machine, install or set up the Codex plugin
there before resuming the Codex review gate.
