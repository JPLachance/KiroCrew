/**
 * The pending-fire cursor must survive a reload, or history replays.
 *
 * Reading `/pending` is non-destructive by design, so the backend keeps every fire
 * and answers "everything after `since`". That is what lets a lost response or a
 * second display's overlay still see a reminder. It also means the queue outlives
 * the page — and starting from 0 on every load re-delivered the whole history, so
 * each restart of the desktop shell put an already-seen, already-fired reminder back
 * on screen. From the user's side it read as a bubble that could not be closed.
 *
 * The desktop app this was ported from never hit this: its queue lived in the
 * Electron main process and died with the app.
 */
import { describe, it, expect, beforeEach } from 'vitest'

const CURSOR_KEY = 'cc:pendingCursor'

/**
 * The two helpers as pet.tsx defines them. Duplicated rather than exported: the
 * module pulls in the whole overlay (bridge, rAF loops, CSS) on import, and the
 * behaviour under test is this pair of rules, not the wiring.
 */
function readStoredCursor(): number {
  try {
    const n = Number(window.localStorage.getItem(CURSOR_KEY))
    return Number.isFinite(n) && n > 0 ? n : 0
  } catch {
    return 0
  }
}

function writeStoredCursor(n: number): void {
  try {
    window.localStorage.setItem(CURSOR_KEY, String(n))
  } catch {
    /* ignored */
  }
}

/**
 * The drain rule from pet.tsx's poll, in isolation.
 *
 * `serverSeq` is the backend's in-memory counter — it starts again at 0 whenever
 * the gateway restarts, which is the whole reason this rule exists.
 */
function drain(
  serverSeq: number,
  pending: { seq: number }[],
): { asked: number[]; shown: number[]; stored: number } {
  const asked: number[] = []
  const call = (since: number) => {
    asked.push(since)
    return { cursor: serverSeq, fires: pending.filter((f) => f.seq > since) }
  }

  const since = readStoredCursor()
  let data = call(since)
  // A cursor below the one we asked from means the sequence restarted.
  if (data.cursor < since) data = call(0)
  writeStoredCursor(data.cursor)
  return { asked, shown: data.fires.map((f) => f.seq), stored: readStoredCursor() }
}

beforeEach(() => {
  window.localStorage.clear()
})

describe('a restarted gateway must not swallow a fire', () => {
  it('re-reads from zero when the server cursor went backwards', () => {
    // Yesterday's session left the cursor at 42; the gateway has since restarted
    // and a newly due reminder is sitting there as seq 1.
    writeStoredCursor(42)
    const r = drain(1, [{ seq: 1 }])

    expect(r.asked).toEqual([42, 0])
    expect(r.shown).toEqual([1])   // without the re-read this is [] — and gone for good
    expect(r.stored).toBe(1)
  })

  it('does not re-read when the cursor is merely unchanged', () => {
    // The ordinary quiet poll: nothing new, and no reason to replay anything.
    writeStoredCursor(7)
    const r = drain(7, [{ seq: 7 }])

    expect(r.asked).toEqual([7])
    expect(r.shown).toEqual([])
    expect(r.stored).toBe(7)
  })

  it('still shows only what is new when the server moved forward', () => {
    writeStoredCursor(7)
    const r = drain(9, [{ seq: 7 }, { seq: 8 }, { seq: 9 }])

    expect(r.asked).toEqual([7])
    expect(r.shown).toEqual([8, 9])
    expect(r.stored).toBe(9)
  })
})

describe('a batch of fires loses none of them', () => {
  /** The drain rule: show the oldest speakable, and only move the cursor to it. */
  function drainOnce(
    serverSeq: number,
    pending: { seq: number; kind: string }[],
  ): { shown: number | null; stored: number } {
    const since = readStoredCursor()
    const fires = pending.filter((f) => f.seq > since)
    const speakable = fires.filter((f) => f.kind !== 'command')
    if (speakable.length === 0) {
      writeStoredCursor(serverSeq)
      return { shown: null, stored: readStoredCursor() }
    }
    const shown = speakable[0]
    writeStoredCursor(shown.seq)
    return { shown: shown.seq, stored: readStoredCursor() }
  }

  it('shows two same-tick reminders across two polls instead of dropping one', () => {
    // Both came due in the same second, so /pending returns them together.
    const pending = [
      { seq: 1, kind: 'reminder' },
      { seq: 2, kind: 'reminder' },
    ]
    const first = drainOnce(2, pending)
    expect(first.shown).toBe(1)      // the OLDEST, not the newest
    expect(first.stored).toBe(1)     // cursor did NOT jump past seq 2

    const second = drainOnce(2, pending)
    expect(second.shown).toBe(2)     // the other one still arrives
    expect(second.stored).toBe(2)
  })

  it('consumes a command-only batch whole, since commands are acted on not shown', () => {
    const pending = [{ seq: 1, kind: 'command' }, { seq: 2, kind: 'command' }]
    const r = drainOnce(2, pending)
    expect(r.shown).toBeNull()
    expect(r.stored).toBe(2)
  })

  it('does not let a trailing command strand an earlier reminder', () => {
    const pending = [
      { seq: 1, kind: 'reminder' },
      { seq: 2, kind: 'command' },
    ]
    const r = drainOnce(2, pending)
    expect(r.shown).toBe(1)
    expect(r.stored).toBe(1)
  })
})

describe('pending cursor persistence', () => {
  it('starts at 0 the very first time, so nothing is missed', () => {
    expect(readStoredCursor()).toBe(0)
  })

  it('resumes where the last run left off instead of replaying', () => {
    writeStoredCursor(7)
    expect(readStoredCursor()).toBe(7)
  })

  it('a fire newer than the cursor is still delivered late', () => {
    // The property that forbids the tempting "just start at the current cursor"
    // shortcut: a reminder that fired while the companion was off must arrive on
    // the user's return, which only works if the cursor is BEHIND that fire.
    writeStoredCursor(7)
    const fires = [{ seq: 7 }, { seq: 8 }].filter((f) => f.seq > readStoredCursor())
    expect(fires.map((f) => f.seq)).toEqual([8])
  })

  it('falls back to 0 on a corrupt value rather than muting everything', () => {
    // Failing towards "replay once" is recoverable; failing towards a huge cursor
    // would silently swallow every future reminder.
    window.localStorage.setItem(CURSOR_KEY, 'not-a-number')
    expect(readStoredCursor()).toBe(0)
    window.localStorage.setItem(CURSOR_KEY, '-5')
    expect(readStoredCursor()).toBe(0)
  })

  it('survives a simulated restart: the same fire is not shown twice', () => {
    const history = [{ seq: 1 }, { seq: 2 }]
    // First run drains everything and records the cursor.
    const firstRun = history.filter((f) => f.seq > readStoredCursor())
    expect(firstRun).toHaveLength(2)
    writeStoredCursor(2)
    // Restart: same backend history, nothing new to show.
    const secondRun = history.filter((f) => f.seq > readStoredCursor())
    expect(secondRun).toHaveLength(0)
  })
})
