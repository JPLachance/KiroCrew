"""Slack integration — link sessions, handoff, channel listing."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from aiohttp import web

from kiro_crew.config.loader import KiroCrewConfig
from kiro_crew.dashboard import state as dashboard_state
from kiro_crew.dashboard.chat_backfill import (
    backfill_content,
    gap_summary,
    select_backfill_messages,
    session_deep_link,
)
from kiro_crew.dashboard.chat_persistence import save_slot_off_loop
from kiro_crew.dashboard.chat_utils import (
    effective_session_key,
    expire_slack_options,
    remember_slack_options,
)
from kiro_crew.dashboard.state import DashboardState, _log_task_exception
from kiro_crew.platform.context import redact_via_context
from kiro_crew.security import redact_and_truncate, redact_credentials, redact_exfiltration_urls
from kiro_crew.sel import sel
from kiro_crew.slack.channel_resolver import _CACHE_FILENAME, ChannelNameResolver
from kiro_crew.slack.format import (
    SLACK_MAX_TEXT,
    SLACK_MSG_LIMIT,
    build_options_blocks,
    build_options_selected_blocks,
    extract_options,
    split_message,
    strip_ansi,
    to_slack_mrkdwn,
)
from kiro_crew.slack.outbound import OPTIONS_FALLBACK_TEXT, PostedOptions
from kiro_crew.sync_bridge import handoff_to_slack

logger = logging.getLogger(__name__)

# Fresh-anchor title fallback: when the slot has no LLM title yet
# (titles land seconds after session creation), fall back to a one-line snippet
# of the first user prompt, then to a neutral default. The raw slot key must
# never be user-visible.
_ANCHOR_TITLE_SNIPPET_CHARS = 60
_ANCHOR_TITLE_DEFAULT = "New session"


def _first_user_prompt(slot) -> str:  # noqa: ANN001 — _ChatSlot (avoids import cycle)
    """Return the slot's first user prompt collapsed to a single line, or ""."""
    for m in slot.messages:
        if m.get("role") == "user":
            text = " ".join(str(m.get("content") or "").split())
            if text:
                return text
    return ""


def _get_channel_resolver(state: DashboardState) -> ChannelNameResolver:
    """Lazily construct the shared ChannelNameResolver on first use.

    The cache path is derived from ``dashboard_state.config_dir`` (accessed as a
    module attribute, not a ``from`` import) so it flows through the same seam
    tests patch — isolating the on-disk cache to ``tmp_path`` under test while
    resolving to the real ``~/.kiro/crew`` dir in production.
    """
    if state._channel_resolver is None:
        cache_path = dashboard_state.config_dir() / _CACHE_FILENAME
        state._channel_resolver = ChannelNameResolver(cache_path=cache_path)
    return state._channel_resolver


_USER_ICON = "\U0001f9d1"
_AGENT_ICON = "\U0001f916"


def _format_backfill_parts(content: str) -> list[str]:
    """Normalise, redact, convert to Slack mrkdwn, redact again, then split.

    ANSI escapes are stripped FIRST, before any redaction. ``to_slack_mrkdwn``
    strips them itself, and that strip can *reassemble* a credential the escapes
    had broken up -- so a secret written as ``AKIA<esc>IOSF...`` is invisible to
    the regex until the strip happens. Normalising up front means the first
    redaction pass sees the credential whole, while the text is still one piece.
    Redacting per block after the strip is NOT equivalent: if the split boundary
    falls inside the credential, each block holds only an unmatchable fragment
    and two adjacent posts reassemble it for the reader.

    Redaction then runs over the whole normalised text, because
    ``to_slack_mrkdwn`` self-truncates at ``SLACK_MAX_TEXT`` before converting:
    converting first would cut a credential at that boundary and leave a prefix
    the regex no longer matches.

    It runs a second time on each converted block, because conversion can still
    reorder or drop characters (inline markup, link rewriting) in ways that
    reveal a secret only afterwards. Redacting on both sides of the transform is
    what makes the guarantee independent of what conversion does to the bytes.

    The pre-split into blocks below the limit exists for that same truncation
    reason: converting the whole message and splitting after would silently drop
    everything past 39,000 characters -- the tail loss the 2,000-char cap used to
    cause, just further out. Blocks are halved against the limit so a conversion
    that *grows* text (table and mermaid rewriting) still cannot reach it.

    When -- and only when -- that split actually produces more than one block,
    tables are left as raw markdown. ``_convert_tables`` keys a table's labels off
    the first ``|`` row it sees, so a block beginning part-way through a table
    adopts a DATA row as its header: that row's values are then only ever emitted
    as labels (and vanish entirely if no data rows follow it, because
    ``_flush_table`` returns early on an empty body), while every later row is
    labelled with the wrong names. Raw pipes read worse on mobile than the
    vertical-list conversion, which is why this is not the default -- but a
    message that never splits keeps the nicer rendering, and one that does keeps
    all of its rows.
    """
    cleaned = redact_via_context(strip_ansi(content or ""))
    blocks = split_message(cleaned, limit=SLACK_MAX_TEXT // 2)
    # A single block IS the whole message, so no table can be straddled.
    keep_tables = len(blocks) > 1
    parts: list[str] = []
    for block in blocks:
        converted = redact_via_context(to_slack_mrkdwn(block, keep_tables=keep_tables))
        parts.extend(split_message(converted, limit=SLACK_MSG_LIMIT))
    return parts


async def drain_slack_backfill(
    state: DashboardState,
    slot: Any,
    channel: str,
    thread_ts: str,
) -> None:
    """Seed a freshly linked Slack thread with readable conversation history.

    Posts the opening turn, a gap marker naming how many turns were skipped, then
    the last few turns in full. Runs as a background task rather than inline in
    the link request: Slack accepts roughly one message per second per channel,
    so a long history split across many parts would hold the HTTP request open
    long enough for the browser fetch to time out while posts kept landing --
    the user would see a failure on a link that actually worked.

    Backgrounding is safe here specifically because the Slack link path has no
    per-message governance gate to fail closed on (unlike the configured-channel
    mirror in ``chat_mirror.py``, which stays inline for that reason).
    """
    client = state.slack_client
    if client is None:
        return
    # Baseline for detecting that the conversation moved on while we work. Taken
    # BEFORE the selection await, not after: selection reads the on-disk
    # transcript and can take a while, so a turn that completes during it would
    # be invisible to a baseline captured afterwards -- leaving a superseded
    # control clickable. Compared against after the posting loops.
    #
    # ``total_messages``, not ``len(slot.messages)``: the message list is capped
    # at _MAX_SLOT_MESSAGES and trimmed from the front on append, so a slot
    # sitting at the cap grows and trims in the same step and its LENGTH never
    # changes. A turn completing mid-drain would then be undetectable on the one
    # slot busy enough to make the race likely. total_messages is a lifetime
    # counter and survives trimming.
    started_running = slot.running
    started_total = slot.total_messages
    session_key = effective_session_key(slot)

    # Offloaded: selection reads the on-disk transcript when the opening turn is
    # off-window, and read_messages_chained parses every tab_id sibling file (and
    # globs the sessions dir to rebuild a stale index). On the loop thread that
    # would stall every other chat turn and the liveness heartbeat.
    selection = await asyncio.to_thread(select_backfill_messages, state, slot)
    if not selection.messages:
        return

    async def _post(text: str) -> bool:
        try:
            await client.post_message(channel, text, thread_ts)
            return True
        except Exception:
            # Best-effort: a partially seeded thread is still usable, and the
            # link itself is already persisted. Never bare-pass -- a silent
            # swallow here is what made the original failure invisible.
            logger.debug("slack backfill: post failed", exc_info=True)
            return False

    async def _post_options(choices: list[str], *, interactive: bool) -> None:
        """Post a replayed OPTIONS tag as a control instead of literal text.

        The body and the control are separate Slack messages, so this composes
        with the body pipeline above rather than replacing it -- the body keeps
        its table-safe conversion and full-length redaction, and the choices ride
        in a Block Kit message of their own.

        *interactive* only for the newest reply. Every earlier one asked a
        question this replay has already moved past, so it renders struck through
        and cannot be answered.
        """
        blocks = (
            build_options_blocks(choices)
            if interactive
            else build_options_selected_blocks(choices, [])
        )
        try:
            ts = await client.post_blocks(channel, blocks, OPTIONS_FALLBACK_TEXT, thread_ts)
        except Exception:
            logger.debug("slack backfill: options control post failed", exc_info=True)
            return
        if interactive and ts:
            remember_slack_options(
                state,
                session_key,
                PostedOptions(
                    channel=channel,
                    ts=ts,
                    choices=tuple(choices),
                    blocks=tuple(blocks),
                ),
            )

    for row in selection.first_turn:
        icon = _USER_ICON if row.get("role") == "user" else _AGENT_ICON
        content, choices = _split_backfill_options(row)
        for part in _format_backfill_parts(content):
            if not await _post(f"{icon} {part}"):
                return
        if choices:
            # The opening turn is superseded by definition — spent, never live.
            await _post_options(choices, interactive=False)

    if selection.skipped_turns and selection.recent:
        summary = gap_summary(selection.skipped_turns)
        link = ""
        try:
            # Offloaded: KiroCrewConfig.load() reads and validates the config
            # file, which is blocking I/O like the transcript read above.
            cfg = await asyncio.to_thread(KiroCrewConfig.load)
            link = session_deep_link(cfg.dashboard.url, slot.key)
        except Exception:
            logger.debug("slack backfill: could not build session link", exc_info=True)
        marker = f"_… {summary} — <{link}|open in the dashboard>_" if link else f"_… {summary}_"
        await _post(marker)

    newest = len(selection.recent_rows) - 1
    for idx, row in enumerate(selection.recent_rows):
        icon = _USER_ICON if row.get("role") == "user" else _AGENT_ICON
        content, choices = _split_backfill_options(row)
        for part in _format_backfill_parts(content):
            if not await _post(f"{icon} {part}"):
                return
        if choices:
            await _post_options(choices, interactive=idx == newest)

    # Did the conversation move past the replayed question while we were
    # draining? A turn that was running at any point, or a transcript that grew,
    # means the newest reply we just rendered as a LIVE control is already
    # superseded — and that turn's own expiry ran before our record existed, so
    # nothing else will spend it. Expire it here rather than leaving live buttons
    # for an answer the conversation no longer wants.
    #
    # ``started_running or slot.running``, not a before/after comparison: a turn
    # that is already in flight when the drain begins and is STILL in flight when
    # it ends (a long cron or injected turn) leaves the flag identical at both
    # ends and may not have appended a row yet, so both a `!=` on running and the
    # total_messages check see nothing. The agent is mid-reply the whole time,
    # which is exactly when the replayed question is most certainly stale.
    if started_running or slot.running or slot.total_messages != started_total:
        try:
            await expire_slack_options(state, session_key)
        except Exception:
            logger.debug(
                "slack backfill: could not expire a control superseded mid-drain",
                exc_info=True,
            )


def _split_backfill_options(row: dict[str, Any]) -> tuple[str, list[str]]:
    """Split a replayed row into body text and OPTIONS choices.

    Only AGENT-authored rows are parsed. A person's own message can legitimately
    contain the OPTIONS syntax — quoting it, or discussing it — and lifting the
    tag out of their words would render choices they never offered, so a user row
    is returned verbatim with no choices.
    """
    content = backfill_content(row)
    if row.get("role") == "user":
        return content, []
    # Redact BEFORE extracting. The choices bypass _format_backfill_parts (they
    # go out as Block Kit label/value, not body text), so extracting first would
    # put a credential from assistant history straight into a Slack block with no
    # redaction anywhere on that path. ANSI is stripped first for the same reason
    # the body pipeline does it: the strip can reassemble a credential the escape
    # codes had broken up. Redaction then runs at FULL length, before the tag is
    # split off, so a secret straddling the tag boundary is still matchable.
    redacted, _ = redact_exfiltration_urls(strip_ansi(content))
    redacted, _ = redact_credentials(redacted)
    return extract_options(redacted)


def _spawn_slack_backfill(
    state: DashboardState,
    slot: Any,
    channel: str,
    thread_ts: str,
) -> None:
    """Fire the backfill drain as a tracked background task.

    Uses the established three-callback shape: keep a strong reference so the
    task is not garbage-collected mid-flight, discard it on completion, and log
    any exception through ``_log_task_exception`` (which redacts first). Omitting
    the third callback is a documented defect -- the failure would surface only
    as an unretrieved-exception warning at interpreter shutdown.

    ``state._background_tasks`` is never cancelled at shutdown, so a gateway stop
    mid-drain abandons the task and leaves a partially seeded thread. That is
    accepted: the link is already persisted and the thread is live.
    """
    task = asyncio.create_task(drain_slack_backfill(state, slot, channel, thread_ts))
    state._background_tasks.add(task)
    task.add_done_callback(state._background_tasks.discard)
    task.add_done_callback(_log_task_exception)


async def api_chat_slot_slack_link(request: web.Request) -> web.Response:
    """POST /api/chat/slots/{name}/slack-link — link a dashboard session to Slack."""

    state: DashboardState = request.app["state"]
    name = request.match_info.get("name") or request.match_info.get("slot", "")
    slot = state.get_slot(name) or state._slots.get(name)
    if not slot:
        return web.json_response({"error": "not found"}, status=404)
    if not state.slack_client:
        return web.json_response({"error": "Slack not connected"}, status=503)
    owner_id = getattr(state, "owner_id", None)
    if not owner_id:
        return web.json_response({"error": "owner not configured"}, status=500)

    # The slot's OWN session key: a channel-born slot's turns run on the
    # channel session, so the link has to live there for the turn path and the
    # link projection (state._slot_links) to find it.
    session_key = effective_session_key(slot)

    # Check if already linked
    existing_ts, existing_chan = state.sessions.get_slack_link(session_key)
    if existing_ts and existing_chan:
        try:
            await state.slack_client.post_message(
                existing_chan, "🔗 Session linked from dashboard — continuing here.", existing_ts
            )
        except Exception:
            pass
        return web.json_response(
            {"ok": True, "already_linked": True, "thread_ts": existing_ts, "channel": existing_chan}
        )

    body = await request.json() if request.content_length else {}
    raw_channel = body.get("channel", "")
    # When the caller supplies an existing thread_ts (challenge-and-redirect
    # auto-link from a Slack thread the user replied in), link to THAT thread
    # rather than posting a new one — this is what makes a thread reply route
    # back to its dashboard session bidirectionally.
    existing_thread = str(body.get("thread_ts", "") or "")
    if not raw_channel or raw_channel == "dm":
        target_channel = await state.slack_client.open_dm(owner_id)
    else:
        target_channel = raw_channel

    if existing_thread:
        thread_ts = existing_thread
    else:
        # redact_and_truncate applies both redact_exfiltration_urls +
        # redact_credentials. Fallback chain: LLM title → first-prompt snippet
        # → neutral default. Redaction runs on the full snippet text before
        # truncation so a truncation boundary can never split (and hide) a
        # credential. Slots initialize title to their raw key
        # (state.py), so gate on display_title — a slot still showing
        # NEW_SESSION_TITLE has no real title, while cron/plan/handoff slots
        # (real titles, _titled unset) pass their title through.
        base = slot.title if slot.display_title != dashboard_state.NEW_SESSION_TITLE else ""
        title = redact_and_truncate(base, max_chars=200)
        if not title:
            title = redact_and_truncate(
                _first_user_prompt(slot), max_chars=_ANCHOR_TITLE_SNIPPET_CHARS
            )
        if not title:
            title = _ANCHOR_TITLE_DEFAULT
        thread_ts = await state.slack_client.post_message(
            target_channel, f"\U0001f9f5 *{title}*\nSession linked from dashboard."
        )
        if not thread_ts:
            return web.json_response({"error": "failed to create thread"}, status=500)

    # Route through the ONE canonical link writer. ``link_slack`` sets the same
    # three slot fields and persists via ``set_slack_link``, but it ALSO
    # registers the thread -> slot reverse index that inbound Slack replies
    # resolve through, and releases the thread from any slot that held it
    # before. Hand-assigning the fields here duplicated everything except that
    # index, so a reply in the mirrored thread routed and persisted correctly
    # while nothing ever told the open tab it had arrived. That same index is
    # what resolves an OPTIONS click on the control replayed below back to this
    # conversation -- without it the click would answer into a separate session.
    state.link_slack(slot.key, thread_ts, target_channel)

    # Seed the new thread with readable history — only when we created a NEW
    # thread. Linking to an existing thread (challenge-and-redirect) would
    # duplicate messages the thread already contains.
    if not existing_thread:
        _spawn_slack_backfill(state, slot, target_channel, thread_ts)

    sel().log_api_access(
        caller="dashboard",
        operation="chat.slack_link",
        outcome="success",
        source="dashboard",
        resources=slot.key,
    )
    state.push_slots_update()
    return web.json_response({"ok": True, "thread_ts": thread_ts, "channel": target_channel})


async def api_chat_slot_slack_unlink(request: web.Request) -> web.Response:
    """POST /api/chat/slots/{slot}/slack-unlink — stop mirroring to Slack.

    Symmetric counterpart to ``api_chat_slot_slack_link``. Clears the Slack
    link so subsequent dashboard turns are no longer mirrored, while keeping
    the session, its history, and the existing Slack thread intact. Idempotent:
    unlinking a session with no link returns ``{ok, was_linked: false}``.

    Auth posture is identical to slack-link, with no new auth surface: both are
    reachable as mixed-internal via the ``/api/chat`` prefix in
    ``mixed_internal_paths`` (server.py; token_auth.py prefix-matches sub-routes),
    so on loopback they accept the internal secret and otherwise fall back to
    normal dashboard-token + CSRF auth. No separate allowlist entry is needed —
    and it must NOT be added to the strict ``internal_paths`` set, which would
    wrongly restrict this browser action to loopback-only callers.
    """
    state: DashboardState = request.app["state"]
    name = request.match_info.get("name") or request.match_info.get("slot", "")
    slot = state.get_slot(name) or state._slots.get(name)
    if not slot:
        return web.json_response({"error": "not found"}, status=404)

    # Authoritative key = the slot's own session key. Deriving it from the slot
    # NAME instead would build "dashboard:slack:<ts>" for a channel-born slot,
    # leaving the real link untouched so mirroring silently resumes next turn.
    session_key = effective_session_key(slot)
    cleared = state.sessions.clear_slack_link(session_key)
    # chat_runner copies a dashboard session's link from the bare key onto the
    # "dashboard:"-prefixed one when a turn runs, so both spellings must go or
    # the next turn re-inherits the link. A channel key has no such twin.
    if session_key.startswith("dashboard:"):
        cleared = state.sessions.clear_slack_link(session_key[len("dashboard:") :]) or cleared

    prev_channel = slot._slack_channel
    prev_thread_ts = slot._slack_thread_ts
    # Tear the link down SYNCHRONOUSLY first, in one uninterrupted block, and only
    # then expire. Expiry awaits a Slack edit, which is slow enough for another tab
    # to relink this slot mid-await; resuming afterwards would clear the
    # REPLACEMENT link's in-memory fields while its persisted link survived,
    # leaving the two disagreeing. Nothing is lost by deferring the expiry: a
    # PostedOptions record carries its own channel and ts, and the slot still
    # resolves by session key, so neither the cleared link fields nor the popped
    # reverse index is needed to strike the choices through.
    #
    # The ordering is also safe in the other direction. Between teardown and
    # expiry the buttons are briefly still live, but the reverse index is already
    # gone, so a click in that window cannot resolve a thread to this slot and
    # cannot inject an answer -- it fails closed.
    slot._slack_linked = False
    slot._slack_channel = ""
    slot._slack_thread_ts = ""
    # Drop the thread -> slot reverse index too, or the thread keeps resolving to
    # this conversation after the link is gone.
    if prev_thread_ts:
        state._slack_to_slot.pop(prev_thread_ts, None)

    # EXPIRE rather than just dropping the record. Expiry does both halves: it
    # strikes the choices through in Slack so the buttons cannot be clicked after
    # the link is gone (a click then would answer a question from a conversation
    # this thread is no longer attached to, in a brand-new session), and it clears
    # the record so no later turn can strike through a selection the user already
    # made. Forgetting alone would leave live buttons.
    await expire_slack_options(state, session_key)

    # Best-effort courtesy note so a Slack watcher knows why the thread went
    # quiet. Same redaction path as the link endpoint; failure is non-fatal.
    if cleared and state.slack_client and prev_channel and prev_thread_ts:
        try:
            await state.slack_client.post_message(
                prev_channel,
                "\U0001f50c _Unlinked from dashboard — replies here no longer sync._",
                prev_thread_ts,
            )
        except Exception:
            logger.debug("Failed to post unlink courtesy note to Slack", exc_info=True)

    sel().log_api_access(
        caller="dashboard",
        operation="chat.slack_unlink",
        outcome="success" if cleared else "noop",
        source="dashboard",
        resources=slot.key,
    )
    state.push_slots_update()
    return web.json_response({"ok": True, "was_linked": cleared})


async def list_slack_channels(state: DashboardState) -> list[dict]:
    """List configured Slack destinations, resolving display names."""
    cfg = KiroCrewConfig.load()
    channels: list[dict] = [{"id": "dm", "name": "Direct Message"}]
    seen: set[str] = set()
    unresolved: list[str] = []  # channel IDs that need name lookup

    for tc in cfg.slack.tracking_channels:
        cid = tc.get("channel_id", "")
        if cid and cid not in seen:
            name = tc.get("name") or ""
            channels.append({"id": cid, "name": name or cid})
            seen.add(cid)
            if not name:
                unresolved.append(cid)
    for cid, cc in cfg.slack_channels.items():
        if cid not in seen and cc.activation in ("always", "mention", "observe"):
            channels.append({"id": cid, "name": cid})  # placeholder — resolved below
            seen.add(cid)
            unresolved.append(cid)

    # Resolve placeholder names via cached Slack API call
    if unresolved and state.slack_client is not None:
        try:
            resolver = _get_channel_resolver(state)
            resolved = await resolver.resolve_many(state.slack_client, unresolved)
            for ch in channels:
                if ch["id"] in unresolved:
                    ch["name"] = resolved.get(ch["id"], ch["id"])
        except Exception:
            # Resolution failure leaves placeholder names in place — non-fatal
            logger.debug("Channel name resolution failed", exc_info=True)

    return channels


async def api_slack_channels(request: web.Request) -> web.Response:
    """GET /api/slack/channels — list channels the bot can reply in."""
    state: DashboardState = request.app["state"]
    return web.json_response(await list_slack_channels(state))


async def api_chat_slot_handoff(request: web.Request) -> web.Response:
    """POST /api/chat/slots/{slot}/handoff — hand off session to Slack DM thread."""

    state: DashboardState = request.app["state"]
    name = request.match_info.get("slot") or request.match_info.get("name", "")
    slot = state.get_slot(name) or state._slots.get(name)
    if not slot:
        return web.json_response({"error": "not found"}, status=404)
    if not state.slack_client:
        return web.json_response({"error": "Slack not connected"}, status=503)
    if not state.conversation_log:
        return web.json_response({"error": "no conversation log"}, status=500)

    try:
        await save_slot_off_loop(state, slot)
    except Exception:
        pass

    channel = None
    try:
        body = await request.json()
        channel = body.get("channel")
    except Exception:
        pass

    history_key = effective_session_key(slot)
    thread_ts = await handoff_to_slack(
        state.slack_client,
        state.owner_id,
        state.conversation_log,
        history_key,
        title=slot.title if slot._titled else "",
        channel=channel,
        sessions=state.sessions,
    )
    if not thread_ts:
        return web.json_response({"error": "handoff failed"}, status=500)

    sel().log_api_access(
        caller="dashboard",
        operation="chat.slot_handoff",
        outcome="allowed",
        source="dashboard",
        resources=slot.key,
    )
    return web.json_response({"ok": True, "thread_ts": thread_ts})


async def api_handoff_channels(request: web.Request) -> web.Response:
    """GET /api/handoff-channels — deprecated, use /api/slack/channels instead."""
    return web.json_response({})
