"""Re-editing a pack must not delete the categories the editor never saw.

A manifest has three art sections: required `states`, optional `moods`, and
open-ended `random` clips. `pack_detail` returned only `states`, and the editor
saves what it was handed — so loading a PetDex pack with random clips and pressing
save rebuilt the pack WITHOUT them, deleting the art. Nothing errored; the clips
simply stopped playing.
"""

from __future__ import annotations

from kiro_crew.apps.builtins.crew_companion import appearances as ap


def _manifest() -> dict:
    return {
        "meta": {"id": "petdexish", "format": "svg", "type": "custom"},
        "states": {"idle": "idle.svg"},
        "moods": {"happy": "happy.svg"},
        "random": {"wave": "random-wave.svg"},
    }


FILES = {
    "idle.svg": "<svg id='idle'/>",
    "happy.svg": "<svg id='happy'/>",
    "random-wave.svg": "<svg id='wave'/>",
}


class TestDetailCarriesEveryCategory:
    def test_states_moods_and_random_all_come_back(self, tmp_path):
        store = ap.AppearanceStore(tmp_path)
        assert store.save_pack("petdexish", _manifest(), FILES)

        detail = store.pack_detail("petdexish")
        assert detail is not None
        anim = detail["animations"]
        assert set(anim) == {"idle", "happy", "wave"}, f"detail dropped art: {sorted(anim)}"
        assert anim["wave"]["content"] == "<svg id='wave'/>"

    def test_a_round_trip_through_the_editor_keeps_the_random_clip(self, tmp_path):
        """The data-loss path: load a pack, save it back, lose nothing."""
        store = ap.AppearanceStore(tmp_path)
        assert store.save_pack("petdexish", _manifest(), FILES)

        # What the editor does: read the detail, then save from what it read.
        detail = store.pack_detail("petdexish")
        assert detail is not None
        anim = detail["animations"]
        rebuilt_files = {f"{slot}.svg": entry["content"] for slot, entry in anim.items()}
        rebuilt = {
            "meta": detail["meta"] | {"format": "svg"},
            "states": {"idle": "idle.svg"},
            "moods": {"happy": "happy.svg"},
            "random": {"wave": "wave.svg"},
        }
        assert store.save_pack("petdexish", rebuilt, rebuilt_files)

        reloaded = store.pack_detail("petdexish")
        assert reloaded is not None
        after = reloaded["animations"]
        assert "wave" in after, "the random clip was lost on re-save"
        assert "happy" in after, "the mood clip was lost on re-save"

    def test_a_pack_with_only_states_is_unaffected(self, tmp_path):
        store = ap.AppearanceStore(tmp_path)
        m = {"meta": {"id": "plain", "format": "svg", "type": "custom"},
             "states": {"idle": "idle.svg"}}
        assert store.save_pack("plain", m, {"idle.svg": "<svg/>"})
        plain = store.pack_detail("plain")
        assert plain is not None
        assert set(plain["animations"]) == {"idle"}
