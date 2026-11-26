"""Offline tests for sync_models.py. Run: python3 -m unittest discover -s tools/sync"""

import datetime as dt
import json
import os
import shutil
import tempfile
import unittest

import sync_models as sm

NOW = dt.datetime.now(dt.timezone.utc)
RECENT = NOW - dt.timedelta(days=3)
OLD = NOW - dt.timedelta(days=400)

MODELS = """# header
models:
  # OpenAI
  - model_id: openai/gpt-6.2-sol
    display_name: GPT-6.2 Sol
    context_window: 1050000
    supports_temperature: false
    tier: standard

  - model_id: openai/gpt-6-luna
    display_name: GPT-6 Luna
    context_window: 1050000
    supports_temperature: false
    tier: economy

  - model_id: openai/gpt-5.6-sol
    display_name: GPT-5.6 Sol
    context_window: 1050000
    supports_temperature: false
    tier: full

  # Anthropic
  - model_id: anthropic/claude-haiku-4-5
    display_name: Claude Haiku 4.5
    context_window: 200000
    tier: economy

  - model_id: anthropic/claude-opus-5-5
    display_name: Claude Opus 5.5
    context_window: 1000000
    supports_temperature: false
    tier: full

  - model_id: bedrock/global.anthropic.claude-opus-5-5
    display_name: Claude Opus 5.5 (Bedrock)
    context_window: 1000000
    tier: full
"""

DEFAULTS = """providers:
  openai:
    _default:
      model_id: openai/gpt-6.2-sol
      max_tokens: 16384
      temperature: 0.7
agents:
  query: fast
"""

OPENAI_ALL = [sm.ProviderModel("gpt-6.2-sol", created=RECENT), sm.ProviderModel("gpt-6-luna", created=RECENT),
              sm.ProviderModel("gpt-5.6-sol", created=OLD)]
ANTHROPIC_ALL = [sm.ProviderModel("claude-haiku-4-5-20251001", "Claude Haiku 4.5", 200000, OLD),
                 sm.ProviderModel("claude-opus-5-5", "Claude Opus 5.5", 1000000, RECENT)]


class SyncTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.dir, "tools/sync"))
        shutil.copy(os.path.join(os.path.dirname(__file__), "providers.json"),
                    os.path.join(self.dir, "tools/sync/providers.json"))
        self.write("models.yaml", MODELS)
        self.write("agent_defaults.yaml", DEFAULTS)
        self.state = os.path.join(self.dir, "state.json")
        self.summary = os.path.join(self.dir, "summary.md")
        self.out = os.path.join(self.dir, "gh_output")
        self.lists = {"openai": list(OPENAI_ALL), "anthropic": list(ANTHROPIC_ALL)}
        self.saved = dict(sm.FETCHERS)
        for p in sm.FETCHERS:
            sm.FETCHERS[p] = self.fake(p)

    def tearDown(self):
        sm.FETCHERS.clear()
        sm.FETCHERS.update(self.saved)
        shutil.rmtree(self.dir)

    def fake(self, prefix):
        def f(_env):
            v = self.lists.get(prefix)
            if v is None:
                raise KeyError(prefix.upper() + "_API_KEY")
            if isinstance(v, Exception):
                raise v
            return v
        return f

    def write(self, name, text):
        with open(os.path.join(self.dir, name), "w") as f:
            f.write(text)

    def read(self, name):
        with open(os.path.join(self.dir, name)) as f:
            return f.read()

    def run_sync(self):
        if os.path.exists(self.out):
            os.remove(self.out)
        sm.main(["--root", self.dir, "--state", self.state, "--summary", self.summary, "--github-output", self.out])
        with open(self.out) as f:
            return dict(line.split("=", 1) for line in f.read().splitlines())

    def load_state(self):
        with open(self.state) as f:
            return json.load(f)

    def age_missing(self, prefix, hours):
        st = self.load_state()
        for rec in st[prefix + "_state"]["missing"].values():
            rec["first_missing"] = (dt.datetime.fromisoformat(rec["first_missing"]) - dt.timedelta(hours=hours)).isoformat()
        with open(self.state, "w") as f:
            json.dump(st, f)

    def ids(self):
        return [e.model_id for e in sm.parse_entries(self.read("models.yaml").splitlines())]

    def test_no_change_is_noop(self):
        out = self.run_sync()
        self.assertEqual(out["changed"], "false")
        self.assertEqual(self.read("models.yaml"), MODELS)

    def test_outage_changes_nothing_and_counts_failures(self):
        self.lists["openai"] = sm.ProviderError("HTTP 500")
        for _ in range(2):
            out = self.run_sync()
            self.assertEqual(out["changed"], "false")
            self.assertEqual(out["failing"], "")
        out = self.run_sync()
        self.assertEqual(out["failing"], "openai")
        self.assertEqual(self.read("models.yaml"), MODELS)
        self.lists["openai"] = list(OPENAI_ALL)
        out = self.run_sync()
        self.assertEqual(out["failing"], "")

    def test_empty_list_is_a_failure_not_a_wipe(self):
        self.lists["openai"] = []
        self.run_sync()
        self.assertEqual(self.read("models.yaml"), MODELS)

    def test_removal_needs_two_runs_a_day_apart(self):
        self.lists["openai"] = [m for m in OPENAI_ALL if m.id != "gpt-5.6-sol"]
        self.assertEqual(self.run_sync()["changed"], "false")
        self.assertEqual(self.run_sync()["changed"], "false")  # second run, but too soon
        self.age_missing("openai", 21)
        out = self.run_sync()
        self.assertEqual(out["changed"], "true")
        self.assertNotIn("openai/gpt-5.6-sol", self.ids())
        self.assertIn("openai/gpt-6-luna", self.ids())
        self.assertNotIn("\n\n\n", self.read("models.yaml"))

    def test_model_coming_back_resets_the_count(self):
        self.lists["openai"] = [m for m in OPENAI_ALL if m.id != "gpt-5.6-sol"]
        self.run_sync()
        self.lists["openai"] = list(OPENAI_ALL)
        self.run_sync()
        self.lists["openai"] = [m for m in OPENAI_ALL if m.id != "gpt-5.6-sol"]
        self.age_missing("openai", 30) if self.load_state()["openai_state"]["missing"] else None
        self.assertEqual(self.run_sync()["changed"], "false")

    def test_fraction_guard_blocks_mass_removal(self):
        self.lists["openai"] = [sm.ProviderModel("gpt-unrelated", created=OLD)]
        self.run_sync()
        self.age_missing("openai", 30)
        self.run_sync()
        self.assertEqual(self.read("models.yaml"), MODELS)
        self.assertIn("safety limit", self.read("summary.md"))

    def test_default_model_is_never_removed(self):
        self.lists["openai"] = [m for m in OPENAI_ALL if m.id != "gpt-6.2-sol"]
        self.run_sync()
        self.age_missing("openai", 30)
        out = self.run_sync()
        self.assertIn("openai/gpt-6.2-sol", self.ids())
        self.assertEqual(out["protected_missing"], "openai/gpt-6.2-sol")

    def test_dated_snapshot_counts_as_alias(self):
        self.run_sync()
        self.age_missing("anthropic", 30) if self.load_state()["anthropic_state"].get("missing") else None
        self.run_sync()
        self.assertIn("anthropic/claude-haiku-4-5", self.ids())

    def test_gpt6_is_held_out(self):
        self.lists["openai"] = OPENAI_ALL + [sm.ProviderModel("gpt-6.3-sol", created=RECENT)]
        self.assertEqual(self.run_sync()["changed"], "false")

    def test_new_recent_model_is_added_with_review_flag(self):
        self.lists["openai"] = OPENAI_ALL + [sm.ProviderModel("gpt-7.1-sol", created=RECENT),
                                             sm.ProviderModel("gpt-7.1-cyber", created=RECENT),
                                             sm.ProviderModel("gpt-4.1", created=OLD),
                                             sm.ProviderModel("gpt-7.1-sol-2026-11-01", created=RECENT)]
        out = self.run_sync()
        text = self.read("models.yaml")
        self.assertEqual(out["changed"], "true")
        self.assertEqual(self.ids().count("openai/gpt-7.1-sol"), 1)
        for bad in ("gpt-7.1-cyber", "gpt-4.1", "gpt-7.1-sol-2026-11-01"):
            self.assertNotIn(bad, text)
        block = text.split("openai/gpt-7.1-sol")[1].split("\n\n")[0]
        self.assertIn("display_name: GPT-7.1 Sol", block)
        self.assertIn("context_window: 1050000", block)  # copied from the gpt-N-sol family
        self.assertIn("supports_temperature: false", block)
        self.assertNotIn("tier:", block)
        self.assertIn("# needs-review: auto-added", text)
        self.assertEqual(out["needs_review"], "openai/gpt-7.1-sol")
        # New entry sits with the other openai entries, before anthropic.
        self.assertLess(self.ids().index("openai/gpt-7.1-sol"), self.ids().index("anthropic/claude-haiku-4-5"))

    def test_unknown_context_is_left_empty(self):
        self.lists["anthropic"] = ANTHROPIC_ALL + [sm.ProviderModel("claude-zephyr-6", "Claude Zephyr 6", None, RECENT)]
        self.run_sync()
        text = self.read("models.yaml")
        block = text.split("anthropic/claude-zephyr-6")[1].split("\n\n")[0]
        self.assertNotIn("context_window", block)
        self.assertIn("context_window unknown", text)

    def test_human_fields_kept_unless_provider_changes_them(self):
        self.lists["anthropic"] = [ANTHROPIC_ALL[0], sm.ProviderModel("claude-opus-5-5", "Opus 5.5 by Anthropic", 1000000, RECENT)]
        self.run_sync()  # first sighting: record only
        self.assertIn("display_name: Claude Opus 5.5\n", self.read("models.yaml"))
        self.run_sync()  # unchanged provider value: still kept
        self.assertIn("display_name: Claude Opus 5.5\n", self.read("models.yaml"))
        self.lists["anthropic"] = [ANTHROPIC_ALL[0], sm.ProviderModel("claude-opus-5-5", "Opus 5.5 by Anthropic", 2000000, RECENT)]
        self.run_sync()
        text = self.read("models.yaml")
        self.assertIn("context_window: 2000000", text)
        self.assertIn("display_name: Claude Opus 5.5\n", text)

    def test_bedrock_untouched_and_missing_secret_skips(self):
        self.lists.pop("anthropic")
        out = self.run_sync()
        self.assertEqual(out["changed"], "false")
        self.assertIn("bedrock/global.anthropic.claude-opus-5-5", self.ids())
        self.assertIn("skipped", self.read("summary.md"))


if __name__ == "__main__":
    unittest.main()
