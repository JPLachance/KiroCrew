"""A write that did not happen must not be reported as success.

Two endpoints used to answer `{"ok": True}` for work they had not done:

  * `patch_config` accepted `customPresets` and `kiro.accessory` and dropped both,
    so a saved colour preset or a chosen dress-up prop vanished on the next load —
    while the UI showed it applied, because the renderer had already set its own
    state.
  * `_save_locked` logged an OSError and returned, so a full or read-only data
    home produced an HTTP 200: the panel cleared its input and the reminder was
    gone after a restart.

The second is the more dangerous shape, and its test asserts the part that is easy
to get wrong — that memory is rolled back to what is actually on disk, so the
process cannot keep serving a reminder the file does not contain.
"""

from __future__ import annotations

import json
from datetime import datetime

import pytest

from kiro_crew.apps.builtins.crew_companion.reminders import parse_iso, to_iso
from kiro_crew.apps.builtins.crew_companion.store import CompanionStore


class Clock:
    def __init__(self, start: str = "2026-07-31T14:00:00") -> None:
        self.now = parse_iso(start)

    def __call__(self) -> datetime:
        return self.now


def _store(tmp_path) -> CompanionStore:
    s = CompanionStore(tmp_path, now=Clock())
    s.load()
    return s


class TestConfigFieldsSurviveAReload:
    def test_a_chosen_accessory_is_still_worn(self, tmp_path):
        s = _store(tmp_path)
        s.patch_config({"kiro": {"accessory": "sunglasses"}})

        # a fresh store reads the file the way the gateway does after a restart
        again = _store(tmp_path)
        assert again.snapshot()["kiro"]["accessory"] == "sunglasses"

    def test_saved_colour_presets_are_still_there(self, tmp_path):
        presets = [{"id": "p1", "name": "Dusk", "colors": {"body": "#334455"}}]
        s = _store(tmp_path)
        s.patch_config({"customPresets": presets})

        assert _store(tmp_path).snapshot()["customPresets"] == presets

    def test_the_defaults_are_the_ones_the_desktop_app_used(self, tmp_path):
        snap = _store(tmp_path).snapshot()
        assert snap["kiro"]["accessory"] == "none"
        assert snap["customPresets"] == []

    @pytest.mark.parametrize(
        "patch",
        [
            {"kiro": {"accessory": ""}},        # blank is not a choice
            {"kiro": {"accessory": 7}},
            {"kiro": "sunglasses"},             # not the nested shape
            {"customPresets": "not-a-list"},
        ],
    )
    def test_unusable_values_leave_the_setting_alone(self, tmp_path, patch):
        s = _store(tmp_path)
        s.patch_config({"kiro": {"accessory": "partyhat"}})
        s.patch_config({"customPresets": [{"id": "keep"}]})

        s.patch_config(patch)   # must not clobber what the user did choose

        snap = s.snapshot()
        assert snap["kiro"]["accessory"] == "partyhat"
        assert snap["customPresets"] == [{"id": "keep"}]


class TestAFailedWriteIsNotSuccess:
    def test_add_raises_instead_of_reporting_ok(self, tmp_path, monkeypatch):
        s = _store(tmp_path)

        def refuse(*a, **kw):
            raise OSError(28, "No space left on device")

        monkeypatch.setattr("pathlib.Path.write_text", refuse)
        with pytest.raises(OSError):
            s.add("water the plants", to_iso(Clock().now))

    def test_memory_matches_the_file_after_a_failed_write(self, tmp_path, monkeypatch):
        """The rollback. Without it the process serves a reminder that isn't saved."""
        s = _store(tmp_path)
        s.add("first", to_iso(Clock().now))          # succeeds, and is on disk
        before = s.snapshot()["reminders"]
        assert len(before) == 1

        def refuse(*a, **kw):
            raise OSError(30, "Read-only file system")

        monkeypatch.setattr("pathlib.Path.write_text", refuse)
        with pytest.raises(OSError):
            s.add("second", to_iso(Clock().now))

        # in-memory state is back to the last state that reached disk ...
        assert s.snapshot()["reminders"] == before
        # ... and it agrees with what a restart would read
        monkeypatch.undo()
        assert len(_store(tmp_path).snapshot()["reminders"]) == 1

    def test_no_temp_file_is_left_behind(self, tmp_path, monkeypatch):
        s = _store(tmp_path)
        real = json.dumps

        def refuse_after_serialising(obj, **kw):
            return real(obj, **kw)

        monkeypatch.setattr(json, "dumps", refuse_after_serialising)

        def refuse(*a, **kw):
            raise OSError(28, "No space left on device")

        monkeypatch.setattr("pathlib.Path.write_text", refuse)
        with pytest.raises(OSError):
            s.add("x", to_iso(Clock().now))

        monkeypatch.undo()
        assert not list(tmp_path.glob("*.tmp.*")), "a failed write left a temp file"
