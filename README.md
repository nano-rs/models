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
    api: chat_completions                    # optional: chat_completions | responses
    supports_function_calling: true          # optional, false if it cannot call tools
```

`api` picks the OpenAI API nano calls. Leave it out (or use `chat_completions`) for every model except those that only call tools over OpenAI's Responses API, such as GPT-6.x, which use `api: responses`. nano v2.0.37 and later read it; older versions ignore it and use Chat Completions. If you leave `api` out of a model whose ID starts with `openai/gpt-6`, nano uses `responses` anyway.

`supports_function_calling` marks a model that cannot call tools, so nano's tool-using agents explain the problem instead of failing. Leave it out to keep the current setting.

Provider prefixes: `anthropic/`, `bedrock/`, `gemini/`, `openai/`, `workers-ai/`.

`agent_defaults.yaml` maps each agent to a tier (`capable`, `mid`, `fast`) and each provider's tiers to a model from `models.yaml`. A tier can also set `reasoning_effort` (how hard a reasoning model thinks: `none`, `minimal`, `low`, `medium`, `high`, `xhigh` or `max`, depending on the model). nano v2.0.37 and later apply it on sync; leave it out to keep each agent's current setting.

## License

Apache License 2.0. See [LICENSE](LICENSE).
