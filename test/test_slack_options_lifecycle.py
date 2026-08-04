"""OPTIONS lifecycle on the Slack surface.

Two behaviours are locked in here:

* every outbound path renders a trailing ``[OPTIONS: …]`` tag as a control,
  never as literal text — including the link-time backfill, which used to post
  message bodies verbatim;
* a control stops being answerable once the conversation moves past the question
  it asked, whichever surface the next turn arrives on.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from chat_test_helpers import _make_state

from kiro_crew.slack.format import (
    OPTIONS_CHECKBOXES_ACTION,
    OPTIONS_SUBMIT_ACTION,
    replace_options_blocks,
)
from kiro_crew.slack.outbound import (
    PostedOptions,
    expire_options,
    post_assistant_text,
    render_for_slack,
)

AWS_KEY = "AKIAIOSFODNN7EXAMPLE"


def _slack() -> MagicMock:
    slack = MagicMock()
    slack.post_message = AsyncMock(return_value="body_ts")
    slack.post_blocks = AsyncMock(return_value="opt_ts")
    slack.update_message = AsyncMock()
    return slack


def _posted_texts(slack: MagicMock) -> list[str]:
    return [c.args[1] for c in slack.post_message.await_args_list]


def _is_live_control(blocks: list[dict]) -> bool:
    """True when *blocks* contain a clickable OPTIONS control."""
    return any(
        el.get("action_id") in (OPTIONS_CHECKBOXES_ACTION, OPTIONS_SUBMIT_ACTION)
        for b in blocks
        if b.get("type") == "actions"
        for el in b.get("elements", [])
    )


def _context_text(blocks: list[dict]) -> str:
    return " ".join(
        el.get("text", "")
        for b in blocks
        if b.get("type") == "context"
        for el in b.get("elements", [])
    )


class TestPostAssistantText:
    """The shared outbound seam."""

    @pytest.mark.asyncio
    async def test_options_post_as_control_not_literal_text(self):
        """The tag never reaches the channel as text a reader has to parse."""
        slack = _slack()

        posted = await post_assistant_text(
            slack, "C1", "Pick one.\n\n[OPTIONS: Ship it | Hold off]", "T1"
        )

        assert all("[OPTIONS:" not in t for t in _posted_texts(slack))
        assert _posted_texts(slack) == ["Pick one."]
        blocks = slack.post_blocks.await_args.args[1]
        assert _is_live_control(blocks)
        assert posted is not None
        assert posted.choices == ("Ship it", "Hold off")
        assert posted.ts == "opt_ts"
        assert posted.channel == "C1"

    @pytest.mark.asyncio
    async def test_no_options_posts_nothing_extra(self):
        slack = _slack()

        posted = await post_assistant_text(slack, "C1", "Just an answer.", "T1")

        assert posted is None
        slack.post_blocks.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_non_interactive_renders_spent_and_is_not_recorded(self):
        """Replayed history must not invite an answer to a stale question."""
        slack = _slack()

        posted = await post_assistant_text(
            slack, "C1", "Old question.\n\n[OPTIONS: A | B]", "T1", interactive=False
        )

        blocks = slack.post_blocks.await_args.args[1]
        assert not _is_live_control(blocks)
        assert _context_text(blocks) == "~A~  |  ~B~"
        # Nothing to expire later — it was never answerable.
        assert posted is None

    @pytest.mark.asyncio
    async def test_markdown_is_converted_to_mrkdwn(self):
        slack = _slack()

        await post_assistant_text(slack, "C1", "**bold** and [text](http://x)", "T1")

        body = _posted_texts(slack)[0]
        assert "*bold*" in body
        assert "<http://x|text>" in body

    @pytest.mark.asyncio
    async def test_credential_in_a_choice_is_redacted_before_it_becomes_a_value(self):
        """Redaction runs over the whole text BEFORE the tag is split off.

        Extracting first would lift the raw choice out of redaction's reach and
        straight into a Block Kit button value.
        """
        slack = _slack()

        posted = await post_assistant_text(
            slack, "C1", f"Choose.\n\n[OPTIONS: use {AWS_KEY} | cancel]", "T1"
        )

        assert posted is not None
        assert AWS_KEY not in " ".join(posted.choices)
        assert "REDACTED" in posted.choices[0]
        blocks = slack.post_blocks.await_args.args[1]
        assert AWS_KEY not in str(blocks)

    @pytest.mark.asyncio
    async def test_truncation_caps_the_body_but_keeps_the_control(self):
        slack = _slack()

        posted = await post_assistant_text(
            slack,
            "C1",
            "x" * 5000 + "\n\n[OPTIONS: A | B]",
            "T1",
            truncate_to=100,
        )

        assert len(_posted_texts(slack)[0]) <= 100
        assert posted is not None
        assert posted.choices == ("A", "B")

    @pytest.mark.asyncio
    async def test_truncation_redacts_before_slicing(self):
        """A credential straddling the cut must not survive as a fragment."""
        slack = _slack()

        await post_assistant_text(
            slack, "C1", "y" * 40 + AWS_KEY, "T1", truncate_to=50
        )

        assert AWS_KEY[:10] not in _posted_texts(slack)[0]

    @pytest.mark.asyncio
    async def test_a_credential_straddling_the_formatter_cut_is_redacted(self):
        """``to_slack_mrkdwn`` truncates at ``SLACK_MAX_TEXT`` on its own.

        Redacting after it let a credential sitting across that cut survive as a
        prefix — short enough that the credential regex no longer matched it, so
        the fragment posted to the channel. The fragment lands near the 39,000th
        character, so this has to inspect every chunk, not just the first.
        """
        from kiro_crew.slack.format import SLACK_MAX_TEXT

        slack = _slack()

        await post_assistant_text(
            slack, "C1", "y" * (SLACK_MAX_TEXT - 5) + AWS_KEY, "T1"
        )

        posted = "".join(_posted_texts(slack))
        assert AWS_KEY not in posted
        assert AWS_KEY[:5] not in posted

    @pytest.mark.asyncio
    async def test_a_body_past_the_formatter_cut_still_yields_its_control(self):
        """The tag trails the body, so truncation reached it first and ate it.

        A long reply then arrived with its question unanswerable. Splitting the
        tag off before conversion keeps it out of the formatter's reach.
        """
        from kiro_crew.slack.format import SLACK_MAX_TEXT

        slack = _slack()

        posted = await post_assistant_text(
            slack, "C1", "y" * (SLACK_MAX_TEXT + 500) + "\n\n[OPTIONS: A | B]", "T1"
        )

        assert posted is not None
        assert posted.choices == ("A", "B")
        assert _is_live_control(slack.post_blocks.await_args.args[1])

    @pytest.mark.asyncio
    async def test_pipe_separated_choices_survive_intact(self):
        """A choice list is `|`-separated, the same shape as a table row.

        Characterization, not a regression guard: the formatter does not flatten
        the tag today. Extracting before conversion keeps it that way even if the
        table handling later widens.
        """
        slack = _slack()

        posted = await post_assistant_text(
            slack, "C1", "Pick.\n\n[OPTIONS: Ship it | Hold off | Ask again]", "T1"
        )

        assert posted is not None
        assert posted.choices == ("Ship it", "Hold off", "Ask again")

    def test_render_for_slack_returns_body_and_choices(self):
        body, choices = render_for_slack("Answer.\n\n[OPTIONS: A | B]")
        assert body == "Answer."
        assert choices == ["A", "B"]


class TestExpireOptions:
    """Spending a control the conversation has moved past."""

    @pytest.mark.asyncio
    async def test_every_choice_is_struck_through(self):
        slack = _slack()
        posted = await post_assistant_text(
            slack, "C1", "Pick.\n\n[OPTIONS: A | B]", "T1"
        )
        assert posted is not None

        await expire_options(slack, posted)

        blocks = slack.update_message.await_args.kwargs["blocks"]
        assert not _is_live_control(blocks)
        assert _context_text(blocks) == "~A~  |  ~B~"
        assert slack.update_message.await_args.args == ("C1", "opt_ts")

    @pytest.mark.asyncio
    async def test_surrounding_blocks_survive(self):
        """A turn's timing footer shares the message with its control."""
        slack = _slack()
        footer = {"type": "section", "text": {"type": "mrkdwn", "text": "12.3s"}}
        control = {
            "type": "actions",
            "elements": [
                {"type": "checkboxes", "action_id": OPTIONS_CHECKBOXES_ACTION},
                {"type": "button", "action_id": OPTIONS_SUBMIT_ACTION},
            ],
        }
        posted = PostedOptions(
            channel="C1",
            ts="opt_ts",
            choices=("A", "B"),
            blocks=(footer, control),
            text="12.3s",
        )

        await expire_options(slack, posted)

        blocks = slack.update_message.await_args.kwargs["blocks"]
        assert footer in blocks
        assert control not in blocks
        assert not _is_live_control(blocks)

    @pytest.mark.asyncio
    async def test_slack_failure_is_swallowed(self):
        """A thread keeping a live control is the status quo, not a new failure."""
        slack = _slack()
        slack.update_message = AsyncMock(side_effect=Exception("channel_not_found"))
        posted = PostedOptions(
            channel="C1", ts="opt_ts", choices=("A",), blocks=({"type": "actions"},)
        )

        await expire_options(slack, posted)  # must not raise

        slack.update_message.assert_awaited_once()


class TestReplaceOptionsBlocks:
    """The block surgery shared by the click path and the expiry path."""

    def test_replaces_control_in_place_and_preserves_the_rest(self):
        top = {"type": "section", "text": {"type": "mrkdwn", "text": "top"}}
        control = {
            "type": "actions",
            "elements": [{"type": "checkboxes", "action_id": OPTIONS_CHECKBOXES_ACTION}],
        }
        tail = {"type": "context", "elements": [{"type": "mrkdwn", "text": "tail"}]}
        spent = [{"type": "context", "elements": [{"type": "mrkdwn", "text": "~A~"}]}]

        result = replace_options_blocks([top, control, tail], spent)

        assert result == [top, spent[0], tail]

    def test_unrelated_actions_block_is_left_alone(self):
        other = {
            "type": "actions",
            "elements": [{"type": "button", "action_id": "mc_link_dashboard"}],
        }
        spent = [{"type": "context", "elements": [{"type": "mrkdwn", "text": "~A~"}]}]

        result = replace_options_blocks([other], spent)

        assert other in result

    def test_appends_when_no_control_is_present(self):
        spent = [{"type": "context", "elements": [{"type": "mrkdwn", "text": "~A~"}]}]

        result = replace_options_blocks([], spent)

        assert result == spent

    def test_input_blocks_are_not_mutated(self):
        control = {
            "type": "actions",
            "elements": [{"type": "checkboxes", "action_id": OPTIONS_CHECKBOXES_ACTION}],
        }
        blocks = [control]

        replace_options_blocks(blocks, [{"type": "context", "elements": []}])

        assert blocks == [control]


class TestLifecycleOnTheSlot:
    """remember / expire / forget against a real slot registry."""

    def _state(self, tmp_path, monkeypatch):
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        state.sessions.get_slack_link = MagicMock(return_value=(None, None))
        state.push_slots_update = MagicMock()
        state.slack_client = _slack()
        return state

    def _posted(self) -> PostedOptions:
        return PostedOptions(
            channel="C1",
            ts="opt_ts",
            choices=("A", "B"),
            blocks=(
                {
                    "type": "actions",
                    "elements": [
                        {"type": "checkboxes", "action_id": OPTIONS_CHECKBOXES_ACTION}
                    ],
                },
            ),
        )

    @pytest.mark.asyncio
    async def test_a_recorded_control_is_expired_on_the_next_turn(
        self, tmp_path, monkeypatch
    ):
        from kiro_crew.dashboard.chat_utils import (
            effective_session_key,
            expire_slack_options,
            remember_slack_options,
        )

        state = self._state(tmp_path, monkeypatch)
        slot = state.get_or_create_slot("s1")
        key = effective_session_key(slot)

        remember_slack_options(state, key, self._posted())
        assert slot._slack_options_posted

        await expire_slack_options(state, key)

        state.slack_client.update_message.assert_awaited_once()
        assert slot._slack_options_posted == ()

    @pytest.mark.asyncio
    async def test_expiry_runs_once_even_if_more_turns_follow(
        self, tmp_path, monkeypatch
    ):
        """The record is cleared before the edit, so a failure is not retried."""
        from kiro_crew.dashboard.chat_utils import (
            effective_session_key,
            expire_slack_options,
            remember_slack_options,
        )

        state = self._state(tmp_path, monkeypatch)
        slot = state.get_or_create_slot("s1")
        key = effective_session_key(slot)
        remember_slack_options(state, key, self._posted())

        await expire_slack_options(state, key)
        await expire_slack_options(state, key)

        assert state.slack_client.update_message.await_count == 1

    @pytest.mark.asyncio
    async def test_forget_stops_expiry_erasing_the_users_selection(
        self, tmp_path, monkeypatch
    ):
        """A Send click already re-rendered the message with the choice made.

        Striking every choice through afterwards would erase it, so the click
        drops the record instead.
        """
        from kiro_crew.dashboard.chat_utils import (
            effective_session_key,
            expire_slack_options,
            forget_slack_options,
            remember_slack_options,
        )

        state = self._state(tmp_path, monkeypatch)
        slot = state.get_or_create_slot("s1")
        key = effective_session_key(slot)
        remember_slack_options(state, key, self._posted())

        forget_slack_options(state, key)
        await expire_slack_options(state, key)

        state.slack_client.update_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_state_without_slots_cannot_break_the_turn(
        self, tmp_path, monkeypatch
    ):
        """Bookkeeping is cleanup, so it must never abort the turn it runs in.

        ``_run_chat`` takes whatever state object its caller passes; several
        callers pass a stand-in that has no slot registry at all.
        """
        from kiro_crew.dashboard.chat_utils import (
            expire_slack_options,
            forget_slack_options,
            remember_slack_options,
        )

        class _NoSlots:
            slack_client = None

        bare = _NoSlots()

        remember_slack_options(bare, "dashboard:s1", self._posted())
        await expire_slack_options(bare, "dashboard:s1")
        forget_slack_options(bare, "dashboard:s1")

    @pytest.mark.asyncio
    async def test_a_raising_slot_registry_cannot_break_the_turn(
        self, tmp_path, monkeypatch
    ):
        from kiro_crew.dashboard.chat_utils import expire_slack_options

        state = self._state(tmp_path, monkeypatch)
        state.get_slot = MagicMock(side_effect=RuntimeError("registry down"))

        await expire_slack_options(state, "dashboard:s1")

        state.slack_client.update_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_click_clears_the_record_a_mirroring_session_holds(
        self, tmp_path, monkeypatch
    ):
        """A thread can be owned by a dashboard session mirroring into it.

        The control is then recorded under the dashboard key, so clearing only
        the ``slack:<ts>`` key leaves it live — and the next dashboard turn
        strikes it through, erasing the selection the user just made.

        The link is established through the real endpoint, not by seeding the
        thread index by hand: an earlier version of this test set
        ``state._slack_to_slot`` itself and so passed while production never
        registered that mapping at all.
        """
        from kiro_crew.dashboard.chat_utils import (
            effective_session_key,
            expire_slack_options,
            remember_slack_options,
        )
        from kiro_crew.slack import interactions

        state = self._state(tmp_path, monkeypatch)
        state.slack_client.open_dm = AsyncMock(return_value="C1")
        state.slack_client.post_message = AsyncMock(return_value="1785370133.085469")
        state.owner_id = "U1"
        state.sessions.set_slack_link = MagicMock()
        slot = state.get_or_create_slot("s1")

        async with TestClient(TestServer(_slack_app(state))) as client:
            resp = await client.post("/api/chat/slots/s1/slack-link", json={})
            assert resp.status == 200
            thread_ts = (await resp.json())["thread_ts"]

        key = effective_session_key(slot)
        remember_slack_options(state, key, self._posted())

        orch = MagicMock()
        orch.dashboard_state = state
        monkeypatch.setattr(interactions, "_orch", orch)
        interactions._forget_options_control(thread_ts)

        await expire_slack_options(state, key)

        state.slack_client.update_message.assert_not_awaited()
        assert slot._slack_options_posted == ()

    def test_linking_registers_the_thread_so_a_click_can_route_back(
        self, tmp_path, monkeypatch
    ):
        """The reverse index is what resolves a click back to this conversation.

        The link handler used to assign the slot's fields directly and skip the
        state helper that writes it, so a click on the replayed control could
        not find the slot and answered into a separate Slack session.
        """
        from kiro_crew.dashboard.chat_utils import slack_options_owner_key

        state = self._state(tmp_path, monkeypatch)
        slot = state.get_or_create_slot("s1")
        thread_ts = "1785370133.085469"

        state.link_slack("s1", thread_ts, "C1")

        assert state.get_linked_slot(thread_ts) is slot
        assert slack_options_owner_key(state, thread_ts) == "dashboard:s1"

    def test_the_owner_key_survives_a_missing_thread_index(
        self, tmp_path, monkeypatch
    ):
        """The index is written by a helper a caller can forget.

        Resolving through it alone is what made this silently return the wrong
        session, so the resolver also matches on the slot's own link fields.
        """
        from kiro_crew.dashboard.chat_utils import slack_options_owner_key

        state = self._state(tmp_path, monkeypatch)
        slot = state.get_or_create_slot("s1")
        thread_ts = "1785370133.085469"
        slot._slack_linked = True
        slot._slack_channel = "C1"
        slot._slack_thread_ts = thread_ts
        state._slack_to_slot.clear()

        assert slack_options_owner_key(state, thread_ts) == "dashboard:s1"

    def test_an_unowned_thread_resolves_to_its_own_slack_key(
        self, tmp_path, monkeypatch
    ):
        from kiro_crew.dashboard.chat_utils import slack_options_owner_key

        state = self._state(tmp_path, monkeypatch)

        assert (
            slack_options_owner_key(state, "1785370133.085469")
            == "slack:1785370133.085469"
        )

    @pytest.mark.asyncio
    async def test_missing_slot_and_missing_state_are_no_ops(
        self, tmp_path, monkeypatch
    ):
        from kiro_crew.dashboard.chat_utils import (
            expire_slack_options,
            forget_slack_options,
            remember_slack_options,
        )

        state = self._state(tmp_path, monkeypatch)

        # A Slack thread can be mid-turn before its slot exists.
        remember_slack_options(state, "slack:1.0", self._posted())
        await expire_slack_options(state, "slack:1.0")
        forget_slack_options(state, "slack:1.0")
        remember_slack_options(None, "slack:1.0", self._posted())
        await expire_slack_options(None, "slack:1.0")

        state.slack_client.update_message.assert_not_awaited()


class TestTurnEntryWiring:
    """The expiry has to actually fire on a new turn, from either surface.

    The lifecycle helpers above are exercised directly, which would stay green
    if the calls into them were deleted. These tests cover the call sites.
    """

    @pytest.mark.asyncio
    async def test_dashboard_turn_expires_before_doing_anything_else(
        self, tmp_path, monkeypatch
    ):
        """Covers dashboard sends, queue drains, regenerate, rewind, cron
        injection and the Slack-linked-thread route — every turn that runs
        through the dashboard engine."""
        from kiro_crew.dashboard import chat_runner

        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        state.sessions.get_slack_link = MagicMock(return_value=(None, None))
        state.push_slots_update = MagicMock()
        slot = state.get_or_create_slot("s1")

        calls: list[str] = []

        async def _record(_state, session_key):
            calls.append(session_key)

        monkeypatch.setattr(chat_runner, "expire_slack_options", _record)
        # The turn itself is irrelevant — the expiry runs before any provider
        # work, so let the turn fail however it likes.
        try:
            await chat_runner._run_chat(state, slot, "hello")
        except Exception:
            pass

        assert calls == ["dashboard:s1"]

    @pytest.mark.asyncio
    async def test_dashboard_prompt_expansion_is_not_a_new_turn(
        self, tmp_path, monkeypatch
    ):
        """A /prompts reference re-enters the same turn; re-expiring there would
        spend a control the user has not been shown an answer to yet."""
        from kiro_crew.dashboard import chat_runner

        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        state.sessions.get_slack_link = MagicMock(return_value=(None, None))
        state.push_slots_update = MagicMock()
        slot = state.get_or_create_slot("s1")

        calls: list[str] = []

        async def _record(_state, session_key):
            calls.append(session_key)

        monkeypatch.setattr(chat_runner, "expire_slack_options", _record)
        try:
            await chat_runner._run_chat(state, slot, "hello", _prompt_depth=1)
        except Exception:
            pass

        assert calls == []

    @pytest.mark.asyncio
    async def test_slack_inbound_turn_expires(self, monkeypatch):
        """Covers the live Slack path, which never reaches the dashboard engine."""
        from kiro_crew.slack import transport_dispatch

        calls: list[str] = []

        async def _record(_state, session_key):
            calls.append(session_key)

        async def _no_linked_thread(*_a, **_k):
            return False

        # Patch the binding the module actually calls, not the definition site:
        # the import is module-scope, so `transport_dispatch` holds its own
        # reference and patching `chat_utils` would not intercept it.
        monkeypatch.setattr(transport_dispatch, "expire_slack_options", _record)
        monkeypatch.setattr(
            transport_dispatch, "maybe_route_linked_thread", _no_linked_thread
        )
        monkeypatch.setattr(transport_dispatch, "_hydrate_thread_overrides", MagicMock())
        monkeypatch.setattr(transport_dispatch, "_hydrate_conv_flags", MagicMock())

        slack = _slack()
        sessions = MagicMock()
        # Slack-born: the thread index has no owner for it, so the turn runs
        # under the syntactic slack:<ts> key. Must be set explicitly — a bare
        # MagicMock returns a truthy stub and the dispatcher would reroute to it.
        sessions.get_session_for_thread.return_value = None
        # "ping" short-circuits immediately AFTER the expiry, so the turn never
        # needs a provider.
        await transport_dispatch.handle_message_transport(
            slack,
            sessions,
            "C1",
            "ping",
            "1785370133.085469",
            "1785370133.085469",
            "U1",
        )

        assert calls == ["slack:1785370133.085469"]
        slack.post_message.assert_awaited_once()

    @pytest.mark.parametrize(
        "module_name,func_name",
        [
            ("kiro_crew.slack.transport_dispatch", "handle_message_transport"),
            ("kiro_crew.slack.handler", "handle_message"),
        ],
    )
    def test_slack_inbound_expires_again_after_the_turn_serializes(
        self, module_name, func_name
    ):
        """Expiry must run BOTH before and after the turn serializes.

        `get_or_create` is where a turn waits for its session, so an expiry that
        only runs before it is decided on pre-wait state. Two messages arriving
        together both clear the OLD control; the first turn then ends by posting a
        NEW one, which the second turn never expires because its only pass already
        happened — leaving the user live buttons for a question the conversation
        has moved past.

        This is a STRUCTURAL check on call order rather than a behavioural one:
        the inbound path bails long before session acquisition under any harness
        cheap enough to build here (governance, hooks and provider setup all sit
        in between), so driving it end-to-end would test the stubs. Asserting the
        order in the source is what actually fails when someone deletes the second
        pass, which is the regression worth catching. Same omission-detector shape
        the rest of this class uses.
        """
        import importlib
        import inspect

        module = importlib.import_module(module_name)
        source = inspect.getsource(getattr(module, func_name))

        acquire = source.find("get_or_create(")
        assert acquire != -1, f"{func_name} no longer calls get_or_create"

        before = source.find("expire_slack_options(")
        after = source.find("expire_slack_options(", acquire)

        assert before != -1 and before < acquire, (
            f"{func_name} must expire the control BEFORE acquiring the session"
        )
        assert after != -1, (
            f"{func_name} must expire AGAIN after get_or_create returns, or a "
            "control posted while this turn was queued stays clickable"
        )

    @pytest.mark.asyncio
    async def test_slack_inbound_expires_the_threads_owning_session(self, monkeypatch):
        """A Slack reply in a dashboard-owned thread must spend THAT session's control.

        The dispatcher resolves the thread to its owning session before the turn
        begins, so a thread the dashboard created answers under its own key. The
        expiry has to run after that resolution — expiring ``slack:<ts>`` would
        leave the dashboard session's control live and strike through nothing.
        """
        from kiro_crew.slack import transport_dispatch

        calls: list[str] = []

        async def _record(_state, session_key):
            calls.append(session_key)

        async def _no_linked_thread(*_a, **_k):
            return False

        # Patch the binding the module actually calls, not the definition site:
        # the import is module-scope, so `transport_dispatch` holds its own
        # reference and patching `chat_utils` would not intercept it.
        monkeypatch.setattr(transport_dispatch, "expire_slack_options", _record)
        monkeypatch.setattr(
            transport_dispatch, "maybe_route_linked_thread", _no_linked_thread
        )
        monkeypatch.setattr(transport_dispatch, "_hydrate_thread_overrides", MagicMock())
        monkeypatch.setattr(transport_dispatch, "_hydrate_conv_flags", MagicMock())

        slack = _slack()
        sessions = MagicMock()
        sessions.get_session_for_thread.return_value = "dashboard:chat-7-1785370000"
        await transport_dispatch.handle_message_transport(
            slack,
            sessions,
            "C1",
            "ping",
            "1785370133.085469",
            "1785370133.085469",
            "U1",
        )

        assert calls == ["dashboard:chat-7-1785370000"]


def _slack_app(state):
    from kiro_crew.dashboard.chat_slack import api_chat_slot_slack_link

    app = web.Application()
    app["state"] = state
    app.router.add_post("/api/chat/slots/{slot}/slack-link", api_chat_slot_slack_link)
    return app


class TestLinkTimeBackfill:
    """Replaying context into a freshly-linked thread."""

    def _state(self, tmp_path, monkeypatch):
        monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
        state = _make_state(tmp_path)
        state.slack_client = MagicMock()
        state.slack_client.open_dm = AsyncMock(return_value="C1")
        state.slack_client.post_message = AsyncMock(return_value="ts1")
        state.slack_client.post_blocks = AsyncMock(return_value="opt_ts")
        state.owner_id = "U1"
        state.sessions.get_slack_link = MagicMock(return_value=(None, None))
        state.sessions.set_slack_link = MagicMock()
        state.push_slots_update = MagicMock()
        return state

    @pytest.mark.asyncio
    async def test_replayed_options_are_a_control_not_literal_text(
        self, tmp_path, monkeypatch
    ):
        """The backfill posted bodies verbatim, so the tag arrived as text.

        This is the path that carries the reply when the Slack link is created
        after the turn already finished — the mirror has nothing to send by
        then, so the backfill is the only thing that puts the answer in Slack.
        """
        state = self._state(tmp_path, monkeypatch)
        slot = state.get_or_create_slot("s1")
        slot.append("user", "which one?")
        slot.append("assistant", "Your call.\n\n[OPTIONS: Ship it | Hold off]")
        slot.drain()

        async with TestClient(TestServer(_slack_app(state))) as client:
            resp = await client.post("/api/chat/slots/s1/slack-link", json={})
            assert resp.status == 200

        texts = [c.args[1] for c in state.slack_client.post_message.await_args_list]
        assert all("[OPTIONS:" not in t for t in texts)
        blocks = state.slack_client.post_blocks.await_args.args[1]
        assert _is_live_control(blocks)

    @pytest.mark.asyncio
    async def test_the_newest_reply_stays_answerable_and_is_recorded(
        self, tmp_path, monkeypatch
    ):
        state = self._state(tmp_path, monkeypatch)
        slot = state.get_or_create_slot("s1")
        slot.append("assistant", "Latest.\n\n[OPTIONS: A | B]")
        slot.drain()

        async with TestClient(TestServer(_slack_app(state))) as client:
            await client.post("/api/chat/slots/s1/slack-link", json={})

        assert _is_live_control(state.slack_client.post_blocks.await_args.args[1])
        assert slot._slack_options_posted
        assert slot._slack_options_posted[0].choices == ("A", "B")

    @pytest.mark.asyncio
    async def test_a_superseded_question_is_replayed_spent(
        self, tmp_path, monkeypatch
    ):
        """The user already answered it — replaying it live would re-ask it."""
        state = self._state(tmp_path, monkeypatch)
        slot = state.get_or_create_slot("s1")
        slot.append("assistant", "Older.\n\n[OPTIONS: A | B]")
        slot.append("user", "A")
        slot.drain()

        async with TestClient(TestServer(_slack_app(state))) as client:
            await client.post("/api/chat/slots/s1/slack-link", json={})

        blocks = state.slack_client.post_blocks.await_args.args[1]
        assert not _is_live_control(blocks)
        assert _context_text(blocks) == "~A~  |  ~B~"
        assert slot._slack_options_posted == ()

    @pytest.mark.asyncio
    async def test_a_trailing_system_row_does_not_spend_the_newest_reply(
        self, tmp_path, monkeypatch
    ):
        """The transcript holds rows that are never replayed.

        A completed turn appends one, so the last reply is not the last row.
        Judging "newest" by raw position spent the very control the replay
        exists to deliver — the user saw a struck-through question again.
        """
        state = self._state(tmp_path, monkeypatch)
        slot = state.get_or_create_slot("s1")
        slot.append("assistant", "Your call.\n\n[OPTIONS: Ship it | Hold off]")
        slot.append("done", "turn complete")
        slot.drain()

        async with TestClient(TestServer(_slack_app(state))) as client:
            await client.post("/api/chat/slots/s1/slack-link", json={})

        blocks = state.slack_client.post_blocks.await_args.args[1]
        assert _is_live_control(blocks)
        assert slot._slack_options_posted

    @pytest.mark.asyncio
    async def test_a_users_own_options_syntax_survives_the_replay(
        self, tmp_path, monkeypatch
    ):
        """A person can type the OPTIONS syntax — quoting it, or discussing it.

        Routing user rows through the agent-authored path lifted the tag out of
        their words and rendered it as struck-through choices they never offered.
        Their text has to come back verbatim.
        """
        state = self._state(tmp_path, monkeypatch)
        slot = state.get_or_create_slot("s1")
        # The tag must END the message: the parser is line-anchored, so trailing
        # words after it would make this pass whatever the code did.
        slot.append("user", "should I paste this?\n\n[OPTIONS: A | B]")
        slot.drain()

        async with TestClient(TestServer(_slack_app(state))) as client:
            await client.post("/api/chat/slots/s1/slack-link", json={})

        texts = [c.args[1] for c in state.slack_client.post_message.await_args_list]
        assert any("[OPTIONS: A | B]" in t for t in texts)
        state.slack_client.post_blocks.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_only_the_newest_reply_is_answerable(self, tmp_path, monkeypatch):
        state = self._state(tmp_path, monkeypatch)
        slot = state.get_or_create_slot("s1")
        slot.append("assistant", "First.\n\n[OPTIONS: A | B]")
        slot.append("user", "A")
        slot.append("assistant", "Second.\n\n[OPTIONS: C | D]")
        slot.drain()

        async with TestClient(TestServer(_slack_app(state))) as client:
            await client.post("/api/chat/slots/s1/slack-link", json={})

        posts = state.slack_client.post_blocks.await_args_list
        assert len(posts) == 2
        assert not _is_live_control(posts[0].args[1])
        assert _is_live_control(posts[1].args[1])


class TestNonStringKeyIsRefused:
    """A key that is not a real string must never reach slot-key normalization.

    ``_normalize_slot_key`` strips a repeated ``dashboard_`` prefix with an
    unbounded ``while`` loop, which terminates only for a genuine ``str``.
    Anything whose ``startswith`` is always truthy -- a MagicMock standing in
    for a Slack payload field, for instance -- spins forever AND manufactures a
    fresh child object every iteration. Silent, so no test fails and no timeout
    trips; it simply eats memory until the process dies. On CI that presented
    as a runner shutdown at a random point in the shard with zero FAILED lines.

    The click path reaches this lookup with a thread id taken straight off the
    interaction payload, so the type is not guaranteed at the call site -- the
    guard belongs here.
    """

    @pytest.mark.timeout(5)
    def test_a_non_string_key_returns_none_instead_of_spinning(self):
        from kiro_crew.dashboard import chat_utils

        state = MagicMock()

        assert chat_utils.slack_options_slot(state, MagicMock()) is None
        # Refused before any lookup: proves we never entered normalization.
        state.get_slot.assert_not_called()

    @pytest.mark.timeout(5)
    def test_a_real_string_key_still_reaches_the_lookup(self):
        from kiro_crew.dashboard import chat_utils

        state = MagicMock()
        sentinel = object()
        state.get_slot.return_value = sentinel

        assert chat_utils.slack_options_slot(state, "chat-1-123") is sentinel
        state.get_slot.assert_called_once()


class TestUnlinkSpendsTheControl:
    """Unlinking must EXPIRE the OPTIONS control, not merely orphan the record.

    Two failure modes, both closed by expiring rather than forgetting:

    1. Forget-only leaves the buttons live in Slack. After the link is gone a
       click answers a question from a conversation this thread is no longer
       attached to, landing that stale answer in a brand-new session.
    2. Doing nothing leaves the record unreachable once the thread -> slot
       reverse index is popped, so the next dashboard turn's expiry strikes every
       choice through — erasing an answer the user had already given.

    Expiry does both halves: it strikes the choices through in Slack AND clears
    the record. It has to run before the link is torn down, because popping the
    index is what makes the record unreachable.
    """

    @pytest.mark.asyncio
    async def test_unlink_expires_a_live_options_control(self, tmp_path, monkeypatch):
        from kiro_crew.dashboard.chat_slack import api_chat_slot_slack_unlink

        monkeypatch.setattr("kiro_crew.dashboard.chat_slack.sel", lambda: MagicMock())

        state = _make_state(tmp_path)
        # Returns into the JSON body, so it has to be a real bool not a mock.
        state.sessions.clear_slack_link = MagicMock(return_value=True)
        slack = MagicMock()
        slack.update_message = AsyncMock()
        slack.post_message = AsyncMock()
        state.slack_client = slack

        slot = state.get_or_create_slot("s1")
        slot._slack_linked = True
        slot._slack_channel = "C-1"
        slot._slack_thread_ts = "thread-1"
        state._slack_to_slot["thread-1"] = slot.key

        slot._slack_options_posted = (
            PostedOptions(
                channel="C-1",
                ts="opt-1",
                choices=("A", "B"),
                blocks=(),
            ),
        )

        app = web.Application()
        app["state"] = state
        app.router.add_post("/api/chat/slots/{slot}/slack-unlink", api_chat_slot_slack_unlink)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post(f"/api/chat/slots/{slot.key}/slack-unlink")
            assert resp.status == 200

        # The buttons are struck through in Slack, so a click after the unlink
        # cannot inject a stale answer into a new session.
        slack.update_message.assert_awaited_once()
        assert slack.update_message.await_args.args[0] == "C-1"
        assert slack.update_message.await_args.args[1] == "opt-1"
        # And the record is gone, so a later dashboard turn has nothing to strike
        # through and cannot erase the user's selection.
        assert slot._slack_options_posted == ()
        # The reverse index is still dropped — spending the control must not come
        # at the cost of the thread continuing to resolve here.
        assert "thread-1" not in state._slack_to_slot


class TestEveryOutstandingControlIsExpired:
    """A newer control must not displace the record of an older live one.

    One turn can post more than one OPTIONS message, and a single slot is
    reachable from several posting paths. When the record was a single slot the
    newer post overwrote the older one, so the older control stayed on screen
    with nothing tracking it — clicking it answered a question the conversation
    had already moved past, and the answer landed as if it were current.
    """

    @pytest.mark.asyncio
    async def test_expiry_drains_all_recorded_controls(self, tmp_path):
        from kiro_crew.dashboard.chat_utils import (
            expire_slack_options,
            remember_slack_options,
        )

        state = _make_state(tmp_path)
        slack = MagicMock()
        slack.update_message = AsyncMock()
        state.slack_client = slack

        slot = state.get_or_create_slot("s1")

        first = PostedOptions(channel="C-1", ts="opt-1", choices=("A",), blocks=())
        second = PostedOptions(channel="C-1", ts="opt-2", choices=("B",), blocks=())
        remember_slack_options(state, slot.key, first)
        remember_slack_options(state, slot.key, second)

        # Both are tracked — the second did not displace the first.
        assert slot._slack_options_posted == (first, second)

        await expire_slack_options(state, slot.key)

        edited = {call.args[1] for call in slack.update_message.await_args_list}
        assert edited == {"opt-1", "opt-2"}, edited
        assert slot._slack_options_posted == ()

    @pytest.mark.asyncio
    async def test_recording_the_same_control_twice_queues_one_edit(self, tmp_path):
        """A retry, or two paths recording one post, is not two live controls."""
        from kiro_crew.dashboard.chat_utils import (
            expire_slack_options,
            remember_slack_options,
        )

        state = _make_state(tmp_path)
        slack = MagicMock()
        slack.update_message = AsyncMock()
        state.slack_client = slack

        slot = state.get_or_create_slot("s1")
        posted = PostedOptions(channel="C-1", ts="opt-1", choices=("A",), blocks=())
        remember_slack_options(state, slot.key, posted)
        remember_slack_options(state, slot.key, posted)

        assert slot._slack_options_posted == (posted,)

        await expire_slack_options(state, slot.key)
        slack.update_message.assert_awaited_once()


class TestForgettingIsScopedToTheClickedControl:
    """Answering one control must not un-track the others.

    Once several controls can be outstanding at once, clearing the whole
    collection on a click leaves the other ones on screen with nothing tracking
    them — so a later click on one answers a superseded question and no expiry
    can ever reach it. A click spends exactly the control it was made on.

    The unscoped form is still correct for an unlink, where the entire
    conversation is detaching and every control should stop being tracked.
    """

    def test_a_click_forgets_only_the_control_it_answered(self, tmp_path):
        from kiro_crew.dashboard.chat_utils import (
            forget_slack_options,
            remember_slack_options,
        )

        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("s1")

        first = PostedOptions(channel="C-1", ts="opt-1", choices=("A",), blocks=())
        second = PostedOptions(channel="C-1", ts="opt-2", choices=("B",), blocks=())
        remember_slack_options(state, slot.key, first)
        remember_slack_options(state, slot.key, second)

        forget_slack_options(state, slot.key, "opt-1")

        # The answered one is spent; the other is STILL tracked, so a later turn
        # can expire it instead of leaving it clickable forever.
        assert slot._slack_options_posted == (second,)

    def test_omitting_the_ts_still_clears_everything(self, tmp_path):
        """The unlink path detaches the whole conversation, so all of them go."""
        from kiro_crew.dashboard.chat_utils import (
            forget_slack_options,
            remember_slack_options,
        )

        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("s1")
        remember_slack_options(
            state, slot.key, PostedOptions(channel="C-1", ts="opt-1", choices=("A",), blocks=())
        )
        remember_slack_options(
            state, slot.key, PostedOptions(channel="C-1", ts="opt-2", choices=("B",), blocks=())
        )

        forget_slack_options(state, slot.key)

        assert slot._slack_options_posted == ()


class TestControlPostedAfterTheWindowIsSpent:
    """A control recorded after its expiry window closed must be spent on the spot.

    Both sites post the control AFTER the point where a concurrent turn's expiry
    pass would have seen it — the native handler releases the session permit long
    before the timing footer goes up, and the link backfill awaits a Slack post per
    replayed message. In both cases the superseding turn's expiry runs over a record
    that does not exist yet, so nothing else will ever spend it. Each site therefore
    re-checks after recording and expires immediately if the conversation moved on.
    """

    def test_native_footer_path_rechecks_business_after_recording(self):
        """Structural: the recheck must sit AFTER remember_slack_options.

        Driving `handle_message` to the footer requires a provider, a stream and a
        full turn; the assertion that actually protects this is that the recheck
        exists and comes after the record, so deleting it fails here.
        """
        import inspect

        from kiro_crew.slack import handler

        source = inspect.getsource(handler.handle_message)
        record = source.find("remember_slack_options(")
        assert record != -1, "handle_message no longer records the control"
        recheck = source.find("is_busy(", record)
        assert recheck != -1, (
            "handle_message must re-check is_busy AFTER recording, or a control "
            "posted once the permit is released stays live for a superseded question"
        )
        expire_after = source.find("expire_slack_options(", recheck)
        assert expire_after != -1, "the is_busy re-check must lead to an expiry"

    @pytest.mark.asyncio
    async def test_backfill_expires_when_the_transcript_advanced_mid_drain(
        self, tmp_path, monkeypatch
    ):
        """The seeding drain is backgrounded, so a reply can land while it runs.

        Every post awaits Slack (roughly one message per second), so the newest
        reply's control can be recorded AFTER a new turn already ran its expiry
        pass — leaving live buttons for a superseded question. The drain re-checks
        and spends it itself.
        """
        from kiro_crew.dashboard import chat_slack
        from kiro_crew.dashboard.chat_backfill import BackfillSelection

        expired: list[str] = []

        async def _record_expire(_state, session_key):
            expired.append(session_key)

        monkeypatch.setattr(chat_slack, "expire_slack_options", _record_expire)

        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("s1")
        reply = {"role": "assistant", "content": "pick one [OPTIONS: A | B]"}
        slot.messages.extend([{"role": "user", "content": "hi"}, reply])

        monkeypatch.setattr(
            chat_slack,
            "select_backfill_messages",
            lambda _s, _sl: BackfillSelection(first_turn=[], recent=[[reply]], skipped_turns=0),
        )

        slack = MagicMock()
        slack.post_blocks = AsyncMock(return_value="opt-1")
        state.slack_client = slack

        # A reply lands mid-drain: the transcript grows while we are posting.
        # Bump total_messages too — that is what state.add_message does, and it is
        # the signal the drain compares.
        async def _grow(*_a, **_k):
            slot.messages.append({"role": "user", "content": "actually, neither"})
            slot.total_messages += 1
            return "p1"

        slack.post_message = _grow

        await chat_slack.drain_slack_backfill(state, slot, "C-1", "thread-1")

        assert expired, "a transcript that advanced during the drain must spend the control"

    @pytest.mark.asyncio
    async def test_backfill_renders_the_newest_reply_as_a_live_control(
        self, tmp_path, monkeypatch
    ):
        """An OPTIONS tag in replayed history must be a control, not literal text.

        Guards the headline defect of this PR against the seeding rewrite: the
        drain posts bodies as plain text, so without lifting the tag out the
        choices would arrive as the characters `[OPTIONS: A | B]`. Only the newest
        reply is answerable; an earlier one renders struck through.
        """
        from kiro_crew.dashboard import chat_slack
        from kiro_crew.dashboard.chat_backfill import BackfillSelection

        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("s1")
        older = {"role": "assistant", "content": "an old question [OPTIONS: X | Y]"}
        newest = {"role": "assistant", "content": "pick one [OPTIONS: A | B]"}
        typed = {"role": "user", "content": "I typed [OPTIONS: not | mine] myself"}

        monkeypatch.setattr(
            chat_slack,
            "select_backfill_messages",
            lambda _s, _sl: BackfillSelection(
                first_turn=[], recent=[[older], [typed], [newest]], skipped_turns=0
            ),
        )

        slack = MagicMock()
        slack.post_message = AsyncMock(return_value="p1")
        slack.post_blocks = AsyncMock(return_value="opt-1")
        state.slack_client = slack

        await chat_slack.drain_slack_backfill(state, slot, "C-1", "thread-1")

        # Two agent tags become two control messages; the user's typed tag does not.
        assert slack.post_blocks.await_count == 2, slack.post_blocks.await_args_list
        # Only the newest is recorded, so only it can be expired later — the older
        # one was posted already spent and there is nothing to strike through.
        assert [p.ts for p in slot._slack_options_posted] == ["opt-1"]
        # The user's own words survive verbatim, tag included.
        posted_text = " ".join(str(c.args[1]) for c in slack.post_message.await_args_list)
        assert "[OPTIONS: not | mine]" in posted_text

    @pytest.mark.asyncio
    async def test_backfill_redacts_credentials_inside_options_choices(
        self, tmp_path, monkeypatch
    ):
        """Choices bypass the body pipeline, so they need their own redaction.

        The body goes through _format_backfill_parts, which redacts. The choices
        go out as Block Kit label/value instead, so extracting them from raw
        history would put a credential into Slack with no redaction anywhere on
        that path.
        """
        from kiro_crew.dashboard import chat_slack
        from kiro_crew.dashboard.chat_backfill import BackfillSelection

        secret = "AKIAIOSFODNN7EXAMPLE"
        reply = {"role": "assistant", "content": f"which key? [OPTIONS: {secret} | cancel]"}

        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("s1")
        monkeypatch.setattr(
            chat_slack,
            "select_backfill_messages",
            lambda _s, _sl: BackfillSelection(first_turn=[], recent=[[reply]], skipped_turns=0),
        )

        slack = MagicMock()
        slack.post_message = AsyncMock(return_value="p1")
        slack.post_blocks = AsyncMock(return_value="opt-1")
        state.slack_client = slack

        await chat_slack.drain_slack_backfill(state, slot, "C-1", "thread-1")

        blocks_text = json.dumps(slack.post_blocks.await_args_list[0].args[1])
        assert secret not in blocks_text, "a credential must never reach an OPTIONS control"

    @pytest.mark.asyncio
    async def test_backfill_expiry_survives_a_slot_at_the_trim_cap(self, tmp_path, monkeypatch):
        """At the message cap, a new turn does not change len(slot.messages).

        add_message trims from the front once the list exceeds
        _MAX_SLOT_MESSAGES, so a slot sitting at the cap appends and trims in the
        same step and its length is identical before and after. Comparing lengths
        would miss the advancement on exactly the busiest slots — where the race
        is most likely. total_messages is a lifetime counter, so it still moves.
        """
        from kiro_crew.dashboard import chat_slack
        from kiro_crew.dashboard.chat_backfill import BackfillSelection

        expired: list[str] = []

        async def _record_expire(_state, session_key):
            expired.append(session_key)

        monkeypatch.setattr(chat_slack, "expire_slack_options", _record_expire)

        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("s1")
        reply = {"role": "assistant", "content": "pick one [OPTIONS: A | B]"}
        monkeypatch.setattr(
            chat_slack,
            "select_backfill_messages",
            lambda _s, _sl: BackfillSelection(first_turn=[], recent=[[reply]], skipped_turns=0),
        )

        slack = MagicMock()
        slack.post_blocks = AsyncMock(return_value="opt-1")
        state.slack_client = slack

        # A turn lands mid-drain on a slot AT the cap: one row in, one row out, so
        # the list length is unchanged while total_messages advances.
        async def _grow_at_cap(*_a, **_k):
            slot.messages.append({"role": "user", "content": "actually, neither"})
            del slot.messages[:1]
            slot.total_messages += 1
            return "p1"

        slack.post_message = _grow_at_cap
        before = len(slot.messages)

        await chat_slack.drain_slack_backfill(state, slot, "C-1", "thread-1")

        assert len(slot.messages) == before, "the length must be unchanged for this test to bite"
        assert expired, "advancement must be detected via total_messages, not list length"

    @pytest.mark.asyncio
    async def test_backfill_spends_the_control_when_a_turn_runs_throughout(
        self, tmp_path, monkeypatch
    ):
        """A turn in flight for the WHOLE drain moves neither baseline.

        A long cron or injected turn that is already running when the drain starts
        and still running when it ends leaves slot.running identical at both ends,
        and it may not have appended a row yet — so a before/after comparison sees
        nothing. The agent is mid-reply the entire time, which is precisely when
        the replayed question is stale, so the control must still be spent.
        """
        from kiro_crew.dashboard import chat_slack
        from kiro_crew.dashboard.chat_backfill import BackfillSelection

        expired: list[str] = []

        async def _record_expire(_state, session_key):
            expired.append(session_key)

        monkeypatch.setattr(chat_slack, "expire_slack_options", _record_expire)

        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("s1")
        reply = {"role": "assistant", "content": "pick one [OPTIONS: A | B]"}
        # Running before, during and after — and no new row lands. `running` is a
        # derived property (task is not None and not task.done()), so an in-flight
        # turn is simulated with a pending future rather than by assignment.
        slot.task = asyncio.get_running_loop().create_future()
        assert slot.running, "the test must actually present an in-flight turn"
        before_total = slot.total_messages

        monkeypatch.setattr(
            chat_slack,
            "select_backfill_messages",
            lambda _s, _sl: BackfillSelection(first_turn=[], recent=[[reply]], skipped_turns=0),
        )

        slack = MagicMock()
        slack.post_message = AsyncMock(return_value="p1")
        slack.post_blocks = AsyncMock(return_value="opt-1")
        state.slack_client = slack

        await chat_slack.drain_slack_backfill(state, slot, "C-1", "thread-1")

        assert slot.total_messages == before_total, "no row may land for this test to bite"
        assert expired, "a turn running throughout the drain must spend the control"
        slot.task.cancel()

    def test_render_strips_ansi_before_redacting_so_choices_are_covered(self):
        """An ANSI-split credential inside a choice must not survive.

        The body is redacted twice — the second pass runs after to_slack_mrkdwn
        has removed the escapes, so a credential they had broken up is caught
        there. The choices get no second pass: they are split off first and go
        straight into Block Kit labels and values. So the escapes have to be gone
        before the FIRST redaction, or a split credential inside a choice matches
        nothing and reaches Slack intact.
        """
        from kiro_crew.slack.format import strip_ansi
        from kiro_crew.slack.outbound import render_for_slack

        # An AWS key broken up by a colour escape mid-token.
        secret = "AKIA\x1b[0mIOSFODNN7EXAMPLE"
        body, choices = render_for_slack(f"which key? [OPTIONS: {secret} | cancel]")

        assert choices, "the tag must still be parsed"
        # Assert on the ANSI-STRIPPED choice: leaving the escape in place is not a
        # defence, because whatever renders the choice strips it and the reader
        # sees the credential whole. Asserting on the raw text would pass
        # vacuously — the escape splits the token so the contiguous secret is
        # absent either way.
        assert "AKIAIOSFODNN7EXAMPLE" not in strip_ansi("".join(choices))
        assert "AKIAIOSFODNN7EXAMPLE" not in strip_ansi(body)
