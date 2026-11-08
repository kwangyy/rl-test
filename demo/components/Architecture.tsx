import { useEffect, useMemo, useState } from "react";
import { pretty, toneOf, type Row, type Step } from "../app";

/** The action the controller has picked but whose result has not arrived yet (live and replay only). */
export type Pending = { t: number; action: string; args: Step["args"] & Record<string, unknown> } | null;

type Box = "retrieval" | "decomposer" | "verifier";
const BOX_OF: Record<string, Box> = {
    search: "retrieval", follow: "retrieval", open_page: "retrieval", find_pages: "retrieval", web_pages: "retrieval",
    decompose: "decomposer", verify: "verifier",
};
const isStop = (action: string) => action === "commit" || action === "abstain";

/**
 * One moment of a run. A tool step is three frames (the controller picks it, the tool runs,
 * the result lands in the shared state); commit/abstain is two (pick, then stop -> answer).
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

/** What the frame's status line says while a run is playing back. */
export function frameStatus(f: Frame): string {
    const verb: Record<string, string> = {
        search: "searching", follow: "following a link", decompose: "splitting the claim",
        verify: "judging the evidence", open_page: "opening a page", find_pages: "finding pages", web_pages: "searching the web",
    };
    switch (f.kind) {
        case "input": return "Claim received";
        case "choose": return "Controller choosing the next action…";
        case "decide": return `Controller chose ${f.step.action}`;
        case "run": return `${boxLabel[BOX_OF[f.step.action] ?? "retrieval"]}: ${verb[f.step.action] ?? f.step.action}…`;
        case "result": return "Result added to the shared state";
        case "stop": return "Final answer";
    }
}

const boxLabel: Record<Box, string> = { retrieval: "Retrieval", decomposer: "Decomposer", verifier: "Verifier" };
const boxColor: Record<Box | "controller" | "state", string> = {
    controller: "var(--arch-controller)", retrieval: "var(--primary)", decomposer: "var(--warning)",
    verifier: "var(--success)", state: "var(--muted-foreground)",
};

const clip = (s: string, n: number) => (s.length > n ? `${s.slice(0, n - 1)}…` : s);
const argText = (step: Pick<Step, "action" | "args">) => {
    const a = step.args;
    const target = a.query ?? a.title ?? a.page ?? a.target ?? a.link;
    return target ? `${step.action} “${target}”` : step.action;
};

/** Shared-state numbers after the first `n` steps. */
function sharedState(steps: Step[], n: number) {
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

const W = 770, H = 330;
const CTL = { x: 270, y: 14, w: 220, h: 92 };
const CLAIM = { x: 24, y: 14, w: 170, h: 46 };
const FINAL = { x: 586, y: 14, w: 166, h: 92 };
const TOOLS: Record<Box, { x: number; y: number; w: number; h: number }> = {
    retrieval: { x: 60, y: 152, w: 170, h: 70 },
    decomposer: { x: 295, y: 152, w: 170, h: 70 },
    verifier: { x: 530, y: 152, w: 170, h: 70 },
};
const STATE = { x: 110, y: 254, w: 540, h: 66 };
const cx = (b: { x: number; w: number }) => b.x + b.w / 2;
const FAN_Y = 130, MERGE_Y = 238;

function Node({ b, lit, color, title, sub, extra, dim }: {
    b: { x: number; y: number; w: number; h: number }; lit: boolean; color: string;
    title: string; sub?: string; extra?: string; dim?: boolean;
}) {
    return (
        <g style={{ transition: "opacity 300ms" }} opacity={dim ? 0.45 : 1}>
            <rect x={b.x} y={b.y} width={b.w} height={b.h} rx={10}
                fill={lit ? `color-mix(in oklch, ${color} 13%, var(--card))` : "var(--card)"}
                stroke={lit ? color : "var(--border)"} strokeWidth={lit ? 2.2 : 1.2}
                style={{ transition: "all 250ms", filter: lit ? `drop-shadow(0 0 10px color-mix(in oklch, ${color} 45%, transparent))` : "none" }} />
            {lit && <circle cx={b.x + 14} cy={b.y + 17} r={4} fill={color}>
                <animate attributeName="opacity" values="1;0.35;1" dur="1.1s" repeatCount="indefinite" />
            </circle>}
            <text x={b.x + (lit ? 24 : 12)} y={b.y + 21} fontSize={13.5} fontWeight={600}
                fill={lit ? color : "var(--foreground)"} style={{ transition: "all 250ms" }}>{title}</text>
            {sub && <text x={b.x + 12} y={b.y + 39} fontSize={11} fill="var(--muted-foreground)">{sub}</text>}
            {extra && <text x={b.x + 12} y={b.y + b.h - 11} fontSize={11.5} fontFamily="var(--font-mono)"
                fill={lit ? "var(--foreground)" : "var(--muted-foreground)"}>{extra}</text>}
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
                style={{ transition: "stroke 250ms", color }}>
                {lit && <animate attributeName="stroke-dashoffset" from="22" to="0" dur="0.6s" repeatCount="indefinite" />}
            </path>
            {label && <text x={lx} y={ly} fontSize={11} textAnchor="middle"
                fill={lit ? color : "var(--muted-foreground)"} fontWeight={lit ? 600 : 400}>{label}</text>}
        </g>
    );
}

export function ArchitectureDiagram({ frame, steps, row, controller, maxSteps }: {
    frame: Frame; steps: Step[]; row: Row | null; controller?: string; maxSteps: number;
}) {
    const shown = stepsShown(frame);
    const st = useMemo(() => sharedState(steps, shown), [steps, shown]);
    const uses = useMemo(() => {
        const u: Record<Box, number> = { retrieval: 0, decomposer: 0, verifier: 0 };
        for (const s of steps.slice(0, shown)) { const b = BOX_OF[s.action]; if (b) u[b]++; }
        return u;
    }, [steps, shown]);

    const k = frame.kind;
    const action = k === "decide" || k === "run" || k === "result" ? frame.step.action : null;
    const tool = action ? BOX_OF[action] : undefined;
    const ctlLit = k === "choose" || k === "decide";
    const runLit = k === "run" ? tool : undefined;
    const resultLit = k === "result" ? tool : undefined;
    const stopLit = k === "stop";

    const ctlLine = k === "choose" ? "choosing next action…"
        : k === "decide" || k === "run" ? clip(`→ ${argText(frame.step)}`, 30)
        : k === "stop" ? `→ ${frame.ended}` : k === "input" ? "waiting for the claim" : "reading the shared state";
    const ended = k === "stop" ? frame.ended : null;
    const finalVerdict = stopLit ? (row?.final.verdict ?? st.verdict) : null;
    const finalConf = stopLit ? (row?.final.confidence ?? st.confidence) : null;
    const finalColor = stopLit ? (toneOf(finalVerdict).dot === "bg-success" ? "var(--success)"
        : toneOf(finalVerdict).dot === "bg-danger" ? "var(--danger)" : "var(--warning)") : "var(--foreground)";

    const toolExtra = (b: Box) => {
        if (runLit === b) return "running…";
        if (resultLit === b && k === "result") return clip(resultText(frame.step), 24);
        return uses[b] ? `used ×${uses[b]}` : "not used yet";
    };

    const ctlX = cx(CTL);
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

            {/* claim -> controller */}
            <Edge d={`M${CLAIM.x + CLAIM.w} ${CLAIM.y + 23} H${CTL.x - 2}`} lit={k === "input"} />
            {/* controller -> each tool (the lit one last, so grey lines do not cover it) */}
            {(Object.keys(TOOLS) as Box[]).sort((a, b) => Number(a === tool) - Number(b === tool)).map((b) => (
                <Edge key={b} d={`M${ctlX} ${CTL.y + CTL.h} V${FAN_Y} H${cx(TOOLS[b])} V${TOOLS[b].y - 2}`}
                    lit={(k === "decide" || k === "run") && tool === b} color={boxColor[b]} />
            ))}
            {/* each tool -> shared state */}
            {(Object.keys(TOOLS) as Box[]).sort((a, b) => Number(a === resultLit) - Number(b === resultLit)).map((b) => (
                <Edge key={b} d={`M${cx(TOOLS[b])} ${TOOLS[b].y + TOOLS[b].h} V${MERGE_Y} H${cx(STATE)} V${STATE.y - 2}`}
                    lit={resultLit === b} color={boxColor[b]} />
            ))}
            {/* feedback: shared state -> controller, round the left */}
            <Edge d={`M${STATE.x} ${STATE.y + 28} H14 V${CTL.y + 72} H${CTL.x - 2}`} lit={k === "result" || k === "choose"}
                color="var(--muted-foreground)" label="feedback" lx={110} ly={CTL.y + 66} />
            {/* controller -> final answer */}
            <Edge d={`M${CTL.x + CTL.w} ${CTL.y + 46} H${FINAL.x - 2}`} lit={stopLit} color={finalColor}
                label="stop" lx={(CTL.x + CTL.w + FINAL.x) / 2} ly={CTL.y + 40} />

            <Node b={CLAIM} lit={k === "input"} color="var(--primary)" title="Claim" />
            <Node b={CTL} lit={ctlLit} color={boxColor.controller} title="Controller"
                sub={clip(controller ? `${controller} · one tool or stop` : "picks one tool or stops", 36)} extra={ctlLine} />
            <Node b={FINAL} lit={stopLit} color={finalColor} title="Final answer" dim={!stopLit && !row}
                sub={stopLit ? clip(`${pretty(finalVerdict)}`, 24) : "verdict + citations"}
                extra={stopLit
                    ? `${finalConf != null ? `${(finalConf * 100).toFixed(0)}%` : "—"}${row ? ` · ${row.final.evidence.length} cited` : ""}${ended === "abstain" ? " · abstain" : ""}`
                    : "—"} />
            <Node b={TOOLS.retrieval} lit={runLit === "retrieval" || resultLit === "retrieval"} color={boxColor.retrieval}
                title="Retrieval" sub="search · follow → sentences" extra={toolExtra("retrieval")} dim={!uses.retrieval && runLit !== "retrieval"} />
            <Node b={TOOLS.decomposer} lit={runLit === "decomposer" || resultLit === "decomposer"} color={boxColor.decomposer}
                title="Decomposer" sub="claim → sub-claims" extra={toolExtra("decomposer")} dim={!uses.decomposer && runLit !== "decomposer"} />
            <Node b={TOOLS.verifier} lit={runLit === "verifier" || resultLit === "verifier"} color={boxColor.verifier}
                title="Verifier" sub="claim + evidence → verdict" extra={toolExtra("verifier")} dim={!uses.verifier && runLit !== "verifier"} />
            <Node b={STATE} lit={k === "result"} color={boxColor.state} title="Shared state"
                sub={`${st.pages} pages · ${st.sentences} sentences · ${st.subclaims} sub-claims`}
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
