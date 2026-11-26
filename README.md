# nano Model Catalog

The list of AI models that [nano](https://github.com/nano-rs) Enterprise offers in **Settings > AI Configuration**, plus the default model for each AI agent.

## How nano uses it

nano Enterprise syncs this catalog from the `main` branch once a day, and an admin can sync it at any time with **Settings > AI Configuration > Sync Upstream**.

- **Upstream models** are added or updated on sync.
- **Custom models** you add yourself are never changed by sync.
- **Models removed from this catalog** are marked deprecated, not deleted. Admins get a notice when an agent still uses one.
- **Agent defaults** (`agent_defaults.yaml`) set the model for each agent. Agents you configured yourself are never changed by sync.

## Files

`models.yaml` lists the models:

```yaml
models:
  - model_id: anthropic/claude-sonnet-5-5   # provider/model format
    display_name: Claude Sonnet 5.5          # shown in the UI
    context_window: 1000000                  # optional, in tokens
    supports_temperature: false              # optional, false for reasoning models
    tier: standard                           # optional: economy | standard | full
```

Provider prefixes: `anthropic/`, `bedrock/`, `gemini/`, `openai/`, `workers-ai/`.

`agent_defaults.yaml` maps each agent to a tier (`capable`, `mid`, `fast`) and each provider's tiers to a model from `models.yaml`.

## License

Apache License 2.0. See [LICENSE](LICENSE).
