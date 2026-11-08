import { useEffect, useMemo, useState } from "react";
import { pretty, shortModel, toneOf, type Row, type Step } from "../app";

/** The action the controller has picked but whose result has not arrived yet (live and replay only). */
export type Pending = { t: number; action: string; args: Step["args"] & Record<string, unknown> } | null;

const RETRIEVAL = ["search", "follow", "open_page", "find_pages", "web_pages"];
const isStop = (action: string) => action === "commit" || action === "abstain";

/**
 * One moment of a run. A tool step is three frames (the controller picks it, the tool runs,
 * the result lands in the episode state); commit/abstain is two (pick, then stop -> answer).
 */
export type Frame =
    | { kind: "input" }
    | { kind: "choose"; i: number }                       // controller thinking, action not known yet
    | { kind: "decide"; i: number; step: Pick<Step, "action" | "args"> }
    | { kind: "run"; i: number; step: Pick<Step, "action" | "args">; done: boolean }
    | { kind: "result"; i: number; step: Step }
    | { kind: "stop"; i: number; ended: string };

/** Frames for the steps so far. Prefix-stable: new events only append or upgrade the last frame. */
export function buildFrames(steps: Step[], pending: Pending, busy: boolean, row: Row | null): Frame[] {
    const frames: Frame[] = [{ kind: "input" }];
    steps.forEach((step, i) => {
        frames.push({ kind: "decide", i, step });
        if (isStop(step.action)) frames.push({ kind: "stop", i, ended: step.action });
        else frames.push({ kind: "run", i, step, done: true }, { kind: "result", i, step });
    });
    const n = steps.length;
    const last = steps[n - 1];
    if (busy) {
        if (pending && pending.t >= n) {
            frames.push({ kind: "decide", i: n, step: pending });
            if (!isStop(pending.action)) frames.push({ kind: "run", i: n, step: pending, done: false });
        } else if (!last || !isStop(last.action)) {
            frames.push({ kind: "choose", i: n });
        }
    } else if (row && n && !isStop(last!.action)) {
        frames.push({ kind: "stop", i: n - 1, ended: (row.final as { ended_by?: string }).ended_by ?? "budget" });
    }
    return frames;
}

/** Steps whose results the frame has already shown. */
export const stepsShown = (f: Frame) =>
    f.kind === "input" ? 0 : f.kind === "result" || f.kind === "stop" ? f.i + 1 : f.i;

const HOLD: Record<Frame["kind"], number> = { input: 600, choose: 700, decide: 650, run: 700, result: 600, stop: 900 };

/**
 * Walks through the frames at a readable pace, however fast events arrive. `runKey` changes on
 * every new run: animated runs (live, replay) start from the first frame and play, saved ones
 * open on the last frame, paused.
 */
export function usePlayer(frames: Frame[], runKey: number, animate: boolean, busy: boolean) {
    const [cursor, setCursor] = useState(0);
    const [playing, setPlaying] = useState(false);
    const [speed, setSpeedState] = useState(() => Number(localStorage.getItem("arch-speed")) || 1);
    const setSpeed = (s: number) => { setSpeedState(s); localStorage.setItem("arch-speed", String(s)); };
    const last = frames.length - 1;

    useEffect(() => {
        setCursor(animate ? 0 : Number.MAX_SAFE_INTEGER);
        setPlaying(animate);
    }, [runKey]);

    const at = Math.min(cursor, last);
    useEffect(() => {
        if (!playing) return;
        if (at >= last) {
            if (!busy) setPlaying(false); // done; a live run keeps waiting on the tail frame
            return;
        }
        const id = setTimeout(() => setCursor(at + 1), HOLD[frames[at]!.kind] / speed);
        return () => clearTimeout(id);
    }, [playing, at, last, busy, speed]);

    return {
        frame: frames[at]!, at, last, playing, speed, setSpeed,
        seek: (i: number) => { setPlaying(false); setCursor(Math.max(0, Math.min(i, last))); },
        toggle: () => {
            if (playing) return setPlaying(false);
            if (at >= last && !busy) setCursor(0);
            setPlaying(true);
        },
        /** Jump to the moment step i's result arrived (or its stop). */
        seekStep: (i: number) => {
            setPlaying(false);
            const j = frames.findIndex((f) => (f.kind === "result" || f.kind === "stop") && f.i === i);
            if (j >= 0) setCursor(j);
        },
    };
}

const VERB: Record<string, string> = {
    search: "searching the index", follow: "opening a linked page", open_page: "opening a page",
    find_pages: "listing page titles", web_pages: "searching the web", decompose: "splitting the claim",
    verify: "judging the evidence",
};

/** What the frame's status line says while a run is playing back. */
export function frameStatus(f: Frame): string {
    switch (f.kind) {
        case "input": return "Claim received";
        case "choose": return "Controller choosing the next action…";
        case "decide": return `Controller chose ${f.step.action}`;
        case "run": return `${f.step.action}: ${VERB[f.step.action] ?? "running"}…`;
        case "result": return "Result added to the episode state";
        case "stop": return "Final answer";
    }
}

/** The actions the run's controller could take, in display order (commit/abstain go to Final answer). */
function actionsOf(run: Row["run"] | null | undefined, steps: Step[]) {
    const list = [...(run?.tools === "nosearch" ? [] : ["search"]), ...(run?.tool_names ?? []), "decompose", "verify"];
    // older traces without tool_names: add whatever the steps show
    for (const s of steps) if (!isStop(s.action) && !list.includes(s.action)) list.splice(list.length - 2, 0, s.action);
    return list;
}

const verifierName = (v?: string) =>
    !v ? "verifier" : /grpo/.test(v) ? "2B GRPO" : /sft/.test(v) ? "2B SFT" : shortModel(v.split("+")[0]) ?? "verifier";

/** What runs behind each action. */
function behind(action: string, run: Row["run"] | null | undefined) {
    const what: Record<string, string> = {
        search: "BM25 · 5 pages", follow: "opens a linked page", open_page: "page by title",
        find_pages: "lists page titles", web_pages: "Exa web search", decompose: "Qwen flash → sub-claims",
        verify: `${verifierName(run?.verifier)} → verdict`,
    };
    return what[action] ?? action;
}

const colorOf = (action: string) =>
    action === "decompose" ? "var(--warning)" : action === "verify" ? "var(--success)" : "var(--primary)";
const CONTROLLER = "var(--arch-controller)", STATE_COLOR = "var(--muted-foreground)";

const clip = (s: string, n: number) => (s.length > n ? `${s.slice(0, Math.max(1, n - 1))}…` : s);
const argText = (step: Pick<Step, "action" | "args">) => {
    const a = step.args;
    const target = a.query ?? a.title ?? a.page ?? a.target ?? a.link;
    return target ? `${step.action} “${target}”` : step.action;
};

/** Episode-state numbers after the first `n` steps. */
function episodeState(steps: Step[], n: number) {
    const pages = new Set<string>(), sentences = new Set<string>();
    let subclaims = 0, verdict: string | null | undefined, confidence: number | null | undefined;
    for (const s of steps.slice(0, n)) {
        for (const h of s.result.hits ?? []) { pages.add(h.page); sentences.add(`${h.page}:${h.line}`); }
        for (const p of s.result.pages ?? []) pages.add(p);
        if (s.result.subclaims) subclaims = s.result.subclaims.length;
        if (s.action === "verify") { verdict = s.result.verdict; confidence = (s.result as { confidence?: number | null }).confidence; }
    }
    return { pages: pages.size, sentences: sentences.size, subclaims, verdict, confidence };
}

function resultText(step: Step): string {
    const r = step.result as Step["result"] & { confidence?: number | null };
    if (step.action === "verify") return `${pretty(r.verdict)}${r.confidence != null ? ` · ${(r.confidence * 100).toFixed(0)}%` : ""}`;
    if (r.subclaims) return `${r.subclaims.length} sub-claims`;
    if (r.hits?.length) return `${r.hits.length} sentences`;
    if (r.pages?.length) return `${r.pages.length} pages`;
    return r.status ? r.status.replaceAll("_", " ") : "done";
}

// ---- drawing -------------------------------------------------------------------------------

type Rect = { x: number; y: number; w: number; h: number };
const W = 770, H = 352;
const CLAIM: Rect = { x: 24, y: 14, w: 160, h: 80 };
const CTL: Rect = { x: 226, y: 14, w: 244, h: 80 };
const FINAL: Rect = { x: 596, y: 14, w: 160, h: 80 };
const TOOL_Y = 162, TOOL_H = 70, TOOLS_X0 = 44, TOOLS_X1 = 726, GAP = 14;
const STATE: Rect = { x: 115, y: 272, w: 540, h: 70 };
const cx = (b: Rect) => b.x + b.w / 2;
const FAN_Y = 136, MERGE_Y = 258, BACK_Y = 116;

/** Greedy word wrap to `n` characters a line, at most `max` lines (the last one clipped). */
function wrap(text: string, n: number, max: number) {
    const lines: string[] = [];
    let cur = "";
    for (const word of text.split(/\s+/).filter(Boolean)) {
        if (cur && (cur + " " + word).length > n) { lines.push(cur); cur = word; } else cur = cur ? `${cur} ${word}` : word;
    }
    if (cur) lines.push(cur);
    if (lines.length > max) lines.splice(max - 1, lines.length, `${lines.slice(max - 1).join(" ")}`);
    return lines.map((l) => clip(l, n));
}

function Node({ b, lit, color, title, sub, extra, lines, dim, mono = false }: {
    b: Rect; lit: boolean; color: string; title: string; sub?: string; extra?: string; lines?: string[]; dim?: boolean; mono?: boolean;
}) {
    const chars = Math.floor((b.w - 20) / 6.3); // what fits on one line at 11px
    return (
        <g style={{ transition: "opacity 300ms" }} opacity={dim ? 0.45 : 1}>
            <rect x={b.x} y={b.y} width={b.w} height={b.h} rx={10}
                fill={lit ? `color-mix(in oklch, ${color} 13%, var(--card))` : "var(--card)"}
                stroke={lit ? color : "var(--border)"} strokeWidth={lit ? 2.2 : 1.2}
                style={{ transition: "all 250ms", filter: lit ? `drop-shadow(0 0 10px color-mix(in oklch, ${color} 45%, transparent))` : "none" }} />
            {lit && <circle cx={b.x + 13} cy={b.y + 17} r={4} fill={color}>
                <animate attributeName="opacity" values="1;0.35;1" dur="1.1s" repeatCount="indefinite" />
            </circle>}
            <text x={b.x + (lit ? 23 : 11)} y={b.y + 21} fontSize={13.5} fontWeight={600}
                fontFamily={mono ? "var(--font-mono)" : undefined}
                fill={lit ? color : "var(--foreground)"} style={{ transition: "all 250ms" }}>{title}</text>
            {sub && <text x={b.x + 11} y={b.y + 39} fontSize={11} fill="var(--muted-foreground)">{clip(sub, chars + 3)}</text>}
            {lines?.map((l, i) => <text key={i} x={b.x + 11} y={b.y + 39 + i * 14} fontSize={11} fill="var(--muted-foreground)">{l}</text>)}
            {extra && <text x={b.x + 11} y={b.y + b.h - 11} fontSize={11} fontFamily="var(--font-mono)"
                fill={lit ? "var(--foreground)" : "var(--muted-foreground)"}>{clip(extra, chars)}</text>}
        </g>
    );
}

function Edge({ d, lit, color = "var(--primary)", label, lx, ly }: {
    d: string; lit: boolean; color?: string; label?: string; lx?: number; ly?: number;
}) {
    return (
        <g>
            <path d={d} fill="none" stroke={lit ? color : "var(--border)"} strokeWidth={lit ? 2.4 : 1.4}
                markerEnd={`url(#${lit ? "arrow-lit" : "arrow"})`} strokeDasharray={lit ? "6 5" : undefined}
                style={{ transition: "stroke 250ms" }}>
                {lit && <animate attributeName="stroke-dashoffset" from="22" to="0" dur="0.6s" repeatCount="indefinite" />}
            </path>
            {label && <text x={lx} y={ly} fontSize={11} textAnchor="middle"
                fill={lit ? color : "var(--muted-foreground)"} fontWeight={lit ? 600 : 400}>{label}</text>}
        </g>
    );
}

export function ArchitectureDiagram({ claim, frame, steps, row, run, controller }: {
    claim: string; frame: Frame; steps: Step[]; row: Row | null; run: Row["run"] | null; controller?: string;
}) {
    const shown = stepsShown(frame);
    const maxSteps = run?.max_steps ?? 8;
    const st = useMemo(() => episodeState(steps, shown), [steps, shown]);
    const actions = useMemo(() => actionsOf(run, steps), [run, steps]);
    const uses = useMemo(() => {
        const u: Record<string, number> = {};
        for (const s of steps.slice(0, shown)) u[s.action] = (u[s.action] ?? 0) + 1;
        return u;
    }, [steps, shown]);

    // one box per action the controller can take, spread across the row
    const w = (TOOLS_X1 - TOOLS_X0 - GAP * (actions.length - 1)) / actions.length;
    const box: Record<string, Rect> = Object.fromEntries(
        actions.map((a, i) => [a, { x: TOOLS_X0 + i * (w + GAP), y: TOOL_Y, w, h: TOOL_H }]));
    const retrieval = actions.filter((a) => RETRIEVAL.includes(a));
    const first = box[retrieval[0] ?? ""], lastR = box[retrieval.at(-1) ?? ""];
    const group = first && lastR ? { x: first.x - 7, y: TOOL_Y - 9, w: lastR.x + lastR.w - first.x + 14, h: TOOL_H + 18 } : null;

    const k = frame.kind;
    const action = k === "decide" || k === "run" || k === "result" ? frame.step.action : null;
    const ctlLit = k === "choose" || k === "decide";
    const runLit = k === "run" ? action : null;
    const resultLit = k === "result" ? action : null;
    const stopLit = k === "stop";

    const ctlLine = k === "choose" ? "choosing next action…"
        : k === "decide" || k === "run" ? `→ ${argText(frame.step)}`
        : k === "stop" ? `→ ${frame.ended}` : k === "input" ? "waiting for the claim" : "reading the episode state";
    const finalVerdict = stopLit ? (row?.final.verdict ?? st.verdict) : null;
    const finalConf = stopLit ? (row?.final.confidence ?? st.confidence) : null;
    const tone = toneOf(finalVerdict).dot;
    const finalColor = !stopLit ? "var(--foreground)"
        : tone === "bg-success" ? "var(--success)" : tone === "bg-danger" ? "var(--danger)" : "var(--warning)";
    const rule = (run as { final_rule?: string } | null)?.final_rule;

    const toolExtra = (a: string) => {
        if (runLit === a) return "running…";
        if (resultLit === a && k === "result") return resultText(frame.step);
        return uses[a] ? `used ×${uses[a]}` : "not used yet";
    };
    const ordered = [...actions].sort((a, b) => Number(a === action) - Number(b === action)); // lit edges on top

    return (
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full select-none" role="img"
            aria-label={`Agent architecture, ${frameStatus(frame)}`}>
            <defs>
                {(["arrow", "arrow-lit"] as const).map((id) => (
                    <marker key={id} id={id} viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
                        <path d="M0 0L10 5L0 10z" fill={id === "arrow" ? "var(--border)" : "context-stroke"} />
                    </marker>
                ))}
            </defs>

            {group && <g>
                <rect x={group.x} y={group.y} width={group.w} height={group.h} rx={13} fill="none"
                    stroke="var(--border)" strokeDasharray="4 4" />
                <text x={group.x + 10} y={group.y + group.h + 13} fontSize={10} letterSpacing="0.1em"
                    fill="var(--muted-foreground)">RETRIEVAL</text>
            </g>}

            <Edge d={`M${CLAIM.x + CLAIM.w} ${CLAIM.y + 40} H${CTL.x - 2}`} lit={k === "input"} />
            {ordered.map((a) => (
                <Edge key={`in-${a}`} d={`M${cx(CTL)} ${CTL.y + CTL.h} V${FAN_Y} H${cx(box[a]!)} V${TOOL_Y - 2}`}
                    lit={(k === "decide" || k === "run") && action === a} color={colorOf(a)} />
            ))}
            {ordered.map((a) => (
                <Edge key={`out-${a}`} d={`M${cx(box[a]!)} ${TOOL_Y + TOOL_H} V${MERGE_Y} H${cx(STATE)} V${STATE.y - 2}`}
                    lit={resultLit === a} color={colorOf(a)} />
            ))}
            <Edge d={`M${STATE.x} ${STATE.y + 35} H12 V${BACK_Y} H${CTL.x + 28} V${CTL.y + CTL.h + 2}`}
                lit={k === "result" || k === "choose"} color={STATE_COLOR} label="feedback" lx={64} ly={BACK_Y - 6} />
            <Edge d={`M${CTL.x + CTL.w} ${CTL.y + 46} H${FINAL.x - 2}`} lit={stopLit} color={finalColor}
                label="commit / abstain" lx={(CTL.x + CTL.w + FINAL.x) / 2} ly={CTL.y + 40} />

            <Node b={CLAIM} lit={k === "input"} color="var(--primary)" title="Claim" lines={wrap(claim, 23, 3)} />
            <Node b={CTL} lit={ctlLit} color={CONTROLLER} title="Controller"
                sub={`${controller ?? "agent"} · one action per turn`} extra={ctlLine} />
            <Node b={FINAL} lit={stopLit} color={finalColor} title="Final answer" dim={!stopLit && !row}
                sub={stopLit ? pretty(finalVerdict) : rule ? `final rule ${rule}` : "verdict + citations"}
                extra={stopLit
                    ? `${finalConf != null ? `${(finalConf * 100).toFixed(0)}%` : "—"}${row ? ` · ${row.final.evidence.length} cited` : ""}`
                    : "—"} />
            {actions.map((a) => (
                <Node key={a} b={box[a]!} lit={runLit === a || resultLit === a} color={colorOf(a)} title={a} mono
                    sub={behind(a, run)} extra={toolExtra(a)} dim={!uses[a] && runLit !== a} />
            ))}
            <Node b={STATE} lit={k === "result"} color={STATE_COLOR} title="Episode state"
                sub={`${st.pages} pages opened · ${st.sentences} sentences found · ${st.subclaims} sub-claims`}
                extra={`latest verdict: ${st.verdict === undefined ? "—" : pretty(st.verdict)} · ${Math.max(0, maxSteps - shown)} of ${maxSteps} actions left`} />
        </svg>
    );
}

const btn = "grid size-7 place-items-center rounded-md text-sm text-muted-foreground transition hover:bg-muted hover:text-foreground cursor-pointer disabled:opacity-30 disabled:cursor-default";

export function PlayerControls({ player, frames, busy }: { player: ReturnType<typeof usePlayer>; frames: Frame[]; busy: boolean }) {
    const { at, last, playing, speed } = player;
    const total = frames.filter((f) => f.kind === "decide").length;
    const f = frames[at]!;
    const stepNo = f.kind === "input" ? 0 : f.i + 1;
    return (
        <div className="flex flex-wrap items-center gap-x-3 gap-y-2 border-t border-border/70 px-3 pt-2.5 text-xs text-muted-foreground">
            <div className="flex items-center gap-0.5">
                <button type="button" className={btn} title="First" onClick={() => player.seek(0)} disabled={at === 0}>⏮</button>
                <button type="button" className={btn} title="Back" onClick={() => player.seek(at - 1)} disabled={at === 0}>◀</button>
                <button type="button" className={`${btn} w-auto px-2 font-medium text-foreground`} onClick={player.toggle}>
                    {playing ? "❚❚ Pause" : at >= last && !busy ? "↻ Replay" : "▶ Play"}
                </button>
                <button type="button" className={btn} title="Forward" onClick={() => player.seek(at + 1)} disabled={at >= last}>▶|</button>
            </div>
            <input type="range" min={0} max={last} value={at} onChange={(e) => player.seek(Number(e.target.value))}
                className="h-1 min-w-24 flex-1 cursor-pointer accent-(--primary)" aria-label="Playback position" />
            <span className="tabular-nums whitespace-nowrap">step {stepNo} / {total}{busy ? "+" : ""}</span>
            <div className="flex items-center gap-0.5" role="group" aria-label="Speed">
                {[0.5, 1, 2].map((s) => (
                    <button key={s} type="button" onClick={() => player.setSpeed(s)}
                        className={`cursor-pointer rounded-md px-1.5 py-0.5 tabular-nums transition ${speed === s ? "bg-muted font-medium text-foreground" : "hover:text-foreground"}`}>
                        {s}×
                    </button>
                ))}
            </div>
        </div>
    );
}
