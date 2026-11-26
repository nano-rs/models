#!/usr/bin/env python3
"""Refresh models.yaml from each provider's official model list.

Standard library only. Run by .github/workflows/sync-models.yml.

Safety rules (fail closed):
- A provider whose API errors, returns nothing, or returns an unexpected shape
  is skipped for this run. Its models are left exactly as they are.
- A model is removed only after it is missing from the provider's full list on
  `confirm_runs` runs at least `confirm_min_hours` apart, and never when that
  would remove more than `max_fraction_per_run` of the provider's entries.
- Models used by agent_defaults.yaml are never removed automatically; they are
  reported instead, because removing one would leave agents on a deprecated
  model.
- Removing a model from models.yaml makes nano mark it deprecated on the next
  sync. Nothing is ever deleted from a nano install.
- New models get `supports_temperature: false` (omitting temperature is always
  accepted) and no `tier`. Missing numbers are left empty and flagged with a
  `# needs-review:` comment. Nothing is invented.
- Human-set fields are kept. A field is updated only when the provider's own
  value changed since the last run that saw it.
- bedrock/ entries are never touched.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Callable

ENTRY_RE = re.compile(r"^  - model_id: (\S+)\s*$")
FIELD_RE = re.compile(r"^    ([a-z_]+): (.*?)\s*$")
REVIEW_PREFIX = "  # needs-review:"
PREFIXES = {"anthropic": "anthropic", "openai": "openai", "gemini": "gemini", "workers-ai": "workers-ai"}
TIMEOUT = 30


class ProviderError(Exception):
    pass


@dataclass
class ProviderModel:
    id: str
    display_name: str | None = None
    context_window: int | None = None
    created: dt.datetime | None = None
    function_calling: bool | None = None
    generate_content: bool = True


@dataclass
class Entry:
    model_id: str
    start: int  # index of the "- model_id" line
    end: int  # index one past the last field line
    fields: dict[str, tuple[int, str]] = field(default_factory=dict)

    @property
    def prefix(self) -> str:
        return self.model_id.split("/", 1)[0]

    @property
    def name(self) -> str:
        return self.model_id.split("/", 1)[1]


# ---------------------------------------------------------------------------
# models.yaml (line based; tools/catalog-check enforces the layout)
# ---------------------------------------------------------------------------


def unquote(v: str) -> str:
    if len(v) >= 2 and v[0] == v[-1] == '"':
        return json.loads(v)
    if len(v) >= 2 and v[0] == v[-1] == "'":
        return v[1:-1].replace("''", "'")
    return v


def quote(v: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9@][A-Za-z0-9 ._()/@+-]*", v) and ": " not in v and not v.endswith(":"):
        return v
    return json.dumps(v)


def parse_entries(lines: list[str]) -> list[Entry]:
    entries: list[Entry] = []
    cur: Entry | None = None
    for i, line in enumerate(lines):
        m = ENTRY_RE.match(line)
        if m:
            if cur:
                entries.append(cur)
            cur = Entry(m.group(1), i, i + 1)
            continue
        if cur is None:
            continue
        f = FIELD_RE.match(line)
        if f and cur.end == i:
            cur.fields[f.group(1)] = (i, unquote(f.group(2)))
            cur.end = i + 1
            continue
        entries.append(cur)
        cur = None
    if cur:
        entries.append(cur)
    return entries


def agent_default_ids(text: str) -> set[str]:
    return set(re.findall(r"^\s+model_id:\s*(\S+)\s*$", text, re.M))


# ---------------------------------------------------------------------------
# Provider fetchers
# ---------------------------------------------------------------------------


def http_json(url: str, headers: dict[str, str]) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "nano-model-catalog-sync", **headers})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:  # noqa: S310 (fixed https hosts)
            body = resp.read()
    except urllib.error.HTTPError as e:
        raise ProviderError(f"HTTP {e.code} from {urllib.parse.urlsplit(url).netloc}") from None
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise ProviderError(f"network error: {type(e).__name__}") from None
    try:
        data = json.loads(body)
    except ValueError:
        raise ProviderError("response was not JSON") from None
    if not isinstance(data, dict):
        raise ProviderError("unexpected response shape")
    return data


def parse_time(v) -> dt.datetime | None:
    if isinstance(v, (int, float)):
        return dt.datetime.fromtimestamp(v, dt.timezone.utc)
    if isinstance(v, str):
        try:
            return dt.datetime.fromisoformat(v.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def fetch_anthropic(env: dict[str, str]) -> list[ProviderModel]:
    key = env.get("ANTHROPIC_API_KEY")
    if not key:
        raise KeyError("ANTHROPIC_API_KEY")
    out, after = [], None
    for _ in range(50):
        q = {"limit": "1000", **({"after_id": after} if after else {})}
        data = http_json(
            "https://api.anthropic.com/v1/models?" + urllib.parse.urlencode(q),
            {"x-api-key": key, "anthropic-version": "2023-06-01"},
        )
        items = data.get("data")
        if not isinstance(items, list):
            raise ProviderError("missing data[]")
        for m in items:
            if isinstance(m, dict) and isinstance(m.get("id"), str):
                ctx = m.get("max_input_tokens")
                out.append(ProviderModel(m["id"], m.get("display_name"), ctx if isinstance(ctx, int) else None,
                                         parse_time(m.get("created_at"))))
        if not data.get("has_more"):
            return out
        after = data.get("last_id")
        if not after:
            raise ProviderError("has_more without last_id")
    raise ProviderError("too many pages")


def fetch_openai(env: dict[str, str]) -> list[ProviderModel]:
    key = env.get("OPENAI_API_KEY")
    if not key:
        raise KeyError("OPENAI_API_KEY")
    data = http_json("https://api.openai.com/v1/models", {"Authorization": f"Bearer {key}"})
    items = data.get("data")
    if not isinstance(items, list):
        raise ProviderError("missing data[]")
    return [ProviderModel(m["id"], None, None, parse_time(m.get("created")))
            for m in items if isinstance(m, dict) and isinstance(m.get("id"), str)]


def fetch_gemini(env: dict[str, str]) -> list[ProviderModel]:
    key = env.get("GEMINI_API_KEY")
    if not key:
        raise KeyError("GEMINI_API_KEY")
    out, token = [], None
    for _ in range(50):
        q = {"pageSize": "1000", **({"pageToken": token} if token else {})}
        data = http_json("https://generativelanguage.googleapis.com/v1beta/models?" + urllib.parse.urlencode(q),
                         {"x-goog-api-key": key})
        items = data.get("models")
        if not isinstance(items, list):
            raise ProviderError("missing models[]")
        for m in items:
            if not isinstance(m, dict) or not isinstance(m.get("name"), str):
                continue
            ctx = m.get("inputTokenLimit")
            methods = m.get("supportedGenerationMethods") or []
            out.append(ProviderModel(m["name"].removeprefix("models/"), m.get("displayName"),
                                     ctx if isinstance(ctx, int) else None, None, None,
                                     "generateContent" in methods))
        token = data.get("nextPageToken")
        if not token:
            return out
    raise ProviderError("too many pages")


def fetch_workers_ai(env: dict[str, str]) -> list[ProviderModel]:
    token, account = env.get("CLOUDFLARE_API_TOKEN"), env.get("CLOUDFLARE_ACCOUNT_ID")
    if not token or not account:
        raise KeyError("CLOUDFLARE_API_TOKEN/CLOUDFLARE_ACCOUNT_ID")
    if not re.fullmatch(r"[0-9a-f]{32}", account):
        raise ProviderError("CLOUDFLARE_ACCOUNT_ID is not a 32-character hex id")
    out = []
    for page in range(1, 51):
        q = {"task": "Text Generation", "per_page": "100", "page": str(page)}
        data = http_json(f"https://api.cloudflare.com/client/v4/accounts/{account}/ai/models/search?"
                         + urllib.parse.urlencode(q), {"Authorization": f"Bearer {token}"})
        if data.get("success") is not True or not isinstance(data.get("result"), list):
            raise ProviderError("Cloudflare API returned success=false")
        items = data["result"]
        for m in items:
            if not isinstance(m, dict) or not isinstance(m.get("name"), str):
                continue
            props = {p.get("property_id"): p.get("value") for p in m.get("properties") or [] if isinstance(p, dict)}
            ctx = props.get("context_window")
            try:
                ctx = int(ctx) if ctx is not None else None
            except (TypeError, ValueError):
                ctx = None
            fc = props.get("function_calling")
            out.append(ProviderModel(m["name"], None, ctx, None,
                                     None if fc is None else str(fc).lower() == "true"))
        if len(items) < 100:
            return out
    raise ProviderError("too many pages")


FETCHERS: dict[str, Callable[[dict[str, str]], list[ProviderModel]]] = {
    "anthropic": fetch_anthropic,
    "openai": fetch_openai,
    "gemini": fetch_gemini,
    "workers-ai": fetch_workers_ai,
}

# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------


def present(name: str, ids: set[str]) -> bool:
    """A catalog alias like claude-haiku-4-5 counts as present when the
    provider lists a dated snapshot of it (claude-haiku-4-5-20251001)."""
    if name in ids:
        return True
    pat = re.compile(re.escape(name) + r"-(\d{8}|\d{4}-\d{2}-\d{2})$")
    return any(pat.match(i) for i in ids)


def family_key(name: str) -> str:
    return re.sub(r"\d+(\.\d+)*", "N", name)


def major(name: str) -> int | None:
    m = re.search(r"-(\d+)", name)
    return int(m.group(1)) if m else None


def display_from_id(prefix: str, name: str) -> str:
    if prefix == "openai":
        parts = name.split("-")
        return " ".join([parts[0].upper() + "-" + parts[1]] + [p.capitalize() for p in parts[2:]]) if len(parts) > 1 else name
    if prefix == "workers-ai":
        tail = name.rsplit("/", 1)[-1]
        return f"{tail} (Cloudflare)"
    return name


@dataclass
class Plan:
    add: list[tuple[str, list[str], list[str]]] = field(default_factory=list)  # (model_id, field lines, review)
    remove: list[str] = field(default_factory=list)
    update: list[tuple[str, str, str, str]] = field(default_factory=list)  # (model_id, key, old, new)
    notes: list[str] = field(default_factory=list)
    protected_missing: list[str] = field(default_factory=list)


def plan_provider(prefix: str, models: list[ProviderModel], entries: list[Entry], cfg: dict, rules: dict,
                  state: dict, protected: set[str], now: dt.datetime, plan: Plan) -> None:
    ids = {m.id for m in models}
    by_id = {m.id: m for m in models}
    mine = [e for e in entries if e.prefix == prefix]
    seen = state.setdefault("seen", {})
    missing_state = state.setdefault("missing", {})

    # 1. Removals (two-run confirmation, fraction guard, protected defaults).
    gone = [e for e in mine if not present(e.name, ids)]
    gone_ids = {e.model_id for e in gone}
    for key in list(missing_state):
        if key not in gone_ids:
            missing_state.pop(key)
    confirmed = []
    for e in gone:
        rec = missing_state.setdefault(e.model_id, {"first_missing": now.isoformat(), "runs": 0})
        rec["runs"] += 1
        first = dt.datetime.fromisoformat(rec["first_missing"])
        if rec["runs"] >= rules["confirm_runs"] and (now - first).total_seconds() >= rules["confirm_min_hours"] * 3600:
            confirmed.append(e)
        else:
            plan.notes.append(f"{e.model_id}: missing from {prefix} list (run {rec['runs']}), waiting to confirm")
    if mine and len(confirmed) > rules["max_fraction_per_run"] * len(mine):
        plan.notes.append(f"{prefix}: {len(confirmed)} of {len(mine)} models missing; over the safety limit, removing none")
        confirmed = []
    for e in confirmed:
        if e.model_id in protected:
            plan.protected_missing.append(e.model_id)
            continue
        plan.remove.append(e.model_id)
        missing_state.pop(e.model_id, None)

    # 2. Updates: only when the provider's own value changed since last seen.
    for e in mine:
        pm = by_id.get(e.name)
        if pm is None:
            continue
        prev = seen.get(e.model_id, {})
        for key, val in (("context_window", pm.context_window), ("display_name", pm.display_name)):
            if val is None:
                continue
            val = str(val)
            cur = e.fields.get(key, (None, None))[1]
            if key in prev and prev[key] != val and cur != val:
                plan.update.append((e.model_id, key, cur or "", val))
            prev[key] = val
        if prev:
            seen[e.model_id] = prev

    # 3. Additions.
    if not cfg.get("auto_add"):
        return
    inc, exc = re.compile(cfg["include"]), re.compile(cfg["exclude"])
    have = {e.name for e in mine}
    catalog_majors = [m for m in (major(e.name) for e in mine) if m is not None]
    for pm in sorted(models, key=lambda m: m.id):
        name = pm.id
        if name in have or not inc.search(name) or exc.search(name):
            continue
        if any(present(h, {name}) for h in have):
            continue
        if not pm.generate_content:
            continue
        if cfg.get("max_age_days") and (pm.created is None or (now - pm.created).days > cfg["max_age_days"]):
            continue
        if cfg.get("min_major_from_catalog") and catalog_majors:
            mj = major(name)
            if mj is None or mj < max(catalog_majors):
                continue
        if cfg.get("require_function_calling") and pm.function_calling is not True:
            continue
        if cfg.get("min_context_window") and (pm.context_window or 0) < cfg["min_context_window"]:
            continue
        review, lines = ["tier"], []
        model_id = f"{prefix}/{name}"
        lines.append(f"    display_name: {quote(pm.display_name or display_from_id(prefix, name))}")
        ctx = pm.context_window
        if ctx is None:
            fam = [e for e in mine if family_key(e.name) == family_key(name) and "context_window" in e.fields]
            if fam:
                ctx = int(fam[-1].fields["context_window"][1])
                review.append(f"context_window copied from {fam[-1].model_id}")
            else:
                review.append("context_window unknown")
        if ctx is not None:
            lines.append(f"    context_window: {ctx}")
        lines.append("    supports_temperature: false")
        plan.add.append((model_id, lines, review))
        seen[model_id] = {k: str(v) for k, v in (("context_window", pm.context_window),
                                                  ("display_name", pm.display_name)) if v is not None}


def apply_plan(lines: list[str], plan: Plan, today: str) -> list[str]:
    entries = {e.model_id: e for e in parse_entries(lines)}
    out = list(lines)
    # Updates in place (indices stay valid until removals/additions).
    for model_id, key, _old, new in plan.update:
        e = entries[model_id]
        val = new if key == "context_window" else quote(new)
        if key in e.fields:
            out[e.fields[key][0]] = f"    {key}: {val}"
    # Removals, bottom-up, including the needs-review comments above an entry.
    for model_id in sorted(plan.remove, key=lambda m: entries[m].start, reverse=True):
        e = entries[model_id]
        start = e.start
        while start > 0 and out[start - 1].startswith(REVIEW_PREFIX):
            start -= 1
        del out[start:e.end]
        # Collapse the double blank line the removed block leaves behind.
        if 0 < start < len(out) and out[start].strip() == "" and out[start - 1].strip() == "":
            del out[start]
    # Additions after the last entry with the same prefix (or at the end).
    for model_id, field_lines, review in plan.add:
        prefix = model_id.split("/", 1)[0]
        same = [e for e in parse_entries(out) if e.prefix == prefix]
        block = ["", f"{REVIEW_PREFIX} auto-added {today}; " + "; ".join(review),
                 f"  - model_id: {model_id}"] + field_lines
        at = same[-1].end if same else len(out)
        out[at:at] = block
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--state", required=True, help="path to the JSON state file (read and written)")
    ap.add_argument("--summary", required=True, help="markdown summary to write")
    ap.add_argument("--github-output", default=os.environ.get("GITHUB_OUTPUT"))
    args = ap.parse_args(argv)

    root = args.root
    with open(os.path.join(root, "tools/sync/providers.json")) as f:
        conf = json.load(f)
    with open(os.path.join(root, "models.yaml")) as f:
        text = f.read()
    with open(os.path.join(root, "agent_defaults.yaml")) as f:
        protected = agent_default_ids(f.read())
    try:
        with open(args.state) as f:
            state = json.load(f)
    except (FileNotFoundError, ValueError):
        state = {}

    now = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
    lines = text.splitlines()
    entries = parse_entries(lines)
    plan = Plan()
    status: dict[str, str] = {}
    pstate = state.setdefault("providers", {})

    for prefix, fetch in FETCHERS.items():
        cfg = conf["providers"][prefix]
        ps = pstate.setdefault(prefix, {"consecutive_failures": 0})
        try:
            models = fetch(dict(os.environ))
            if not models:
                raise ProviderError("empty model list")
        except KeyError as e:
            status[prefix] = f"skipped: secret {e.args[0]} not set"
            continue
        except ProviderError as e:
            ps["consecutive_failures"] = ps.get("consecutive_failures", 0) + 1
            ps["last_error"] = str(e)
            status[prefix] = f"FAILED ({e}); left unchanged"
            continue
        ps["consecutive_failures"] = 0
        ps.pop("last_error", None)
        ps["last_ok"] = now.isoformat()
        status[prefix] = f"ok ({len(models)} models listed)"
        plan_provider(prefix, models, entries, cfg, conf["removal"], state.setdefault(prefix + "_state", {}),
                      protected, now, plan)

    new_lines = apply_plan(lines, plan, now.date().isoformat())
    changed = new_lines != lines
    if changed:
        with open(os.path.join(root, "models.yaml"), "w") as f:
            f.write("\n".join(new_lines) + "\n")
    with open(args.state, "w") as f:
        json.dump(state, f, indent=2, sort_keys=True)
        f.write("\n")

    failing = [p for p, s in pstate.items() if s.get("consecutive_failures", 0) >= conf["failure_issue_after_runs"]]
    review = sorted(set(re.findall(r"^\s+- model_id: (\S+)", "\n".join(
        nl for i, nl in enumerate(new_lines) if i > 0 and new_lines[i - 1].startswith(REVIEW_PREFIX)), re.M)))

    md = ["## Model catalog sync", "", "| Provider | Status |", "|---|---|"]
    md += [f"| {p} | {s} |" for p, s in status.items()]
    if plan.add:
        md += ["", "### Added", ""] + [f"- `{m}` (needs review: {', '.join(r)})" for m, _, r in plan.add]
    if plan.remove:
        md += ["", "### Removed (nano marks these deprecated on its next sync)", ""] + [f"- `{m}`" for m in plan.remove]
    if plan.update:
        md += ["", "### Updated (provider changed the value)", ""] + [
            f"- `{m}` {k}: `{o}` -> `{n}`" for m, k, o, n in plan.update]
    if plan.protected_missing:
        md += ["", "### Default models missing from the provider (not removed; a person must pick a new default)", ""] + [
            f"- `{m}`" for m in plan.protected_missing]
    if plan.notes:
        md += ["", "### Notes", ""] + [f"- {n}" for n in plan.notes]
    with open(args.summary, "w") as f:
        f.write("\n".join(md) + "\n")

    if args.github_output:
        with open(args.github_output, "a") as f:
            f.write(f"changed={'true' if changed else 'false'}\n")
            f.write(f"failing={','.join(failing)}\n")
            f.write(f"protected_missing={','.join(plan.protected_missing)}\n")
            f.write(f"needs_review={','.join(review)}\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
