import { useEffect, useMemo, useState } from "react";
import { getJson, pretty, toneOf, type Claim, type Row, type Trace } from "../app";
import { fieldClass, Select } from "./ClaimForm";
import { StepTimeline } from "./StepTimeline";

type Pair = { id: string; a: Claim; b: Claim };
type Filter = "all" | "differ" | "only-a" | "only-b" | "both-wrong";

const pageSize = 25;
const filters: { value: Filter; label: string }[] = [
    { value: "all", label: "All shared" },
    { value: "differ", label: "Verdicts differ" },
    { value: "only-a", label: "Only A correct" },
    { value: "only-b", label: "Only B correct" },
    { value: "both-wrong", label: "Both wrong" },
];
const matches = (p: Pair, f: Filter) =>
    f === "all" ? true
        : f === "differ" ? p.a.verdict !== p.b.verdict
            : f === "only-a" ? p.a.correct && !p.b.correct
                : f === "only-b" ? !p.a.correct && p.b.correct
                    : p.a.gold != null && !p.a.correct && !p.b.correct;
const opsHint = "Ops = searches + decomposes + verifies. This is the cost the reward penalises.";
const label = (name: string) => name.replace(/\.jsonl$/, "");

function Verdict({ claim }: { claim: Claim }) {
    const tone = toneOf(claim.verdict);
    return (
        <span className="inline-flex items-center gap-1.5">
            <span className={`inline-flex items-center gap-1.5 rounded-full px-2 py-0.5 text-xs font-medium capitalize ring-1 ${tone.box}`}>
                {claim.verdict && <span className={`size-1.5 shrink-0 rounded-full ${tone.mark}`} />}
                {pretty(claim.verdict)}
            </span>
            {claim.gold != null && (
                <span title={claim.correct ? "correct" : "incorrect"}
                    className={`font-semibold ${claim.correct ? "text-success" : "text-danger"}`}>{claim.correct ? "✓" : "✗"}</span>
            )}
        </span>
    );
}

function FilterCard({ title, value, active, onClick }: { title: string; value: number; active: boolean; onClick: () => void }) {
    return (
        <button type="button" aria-pressed={active} onClick={onClick} title={active ? "Show all shared claims" : `Show only: ${title.toLowerCase()}`}
            className={`cursor-pointer rounded-lg px-3 py-2 text-left transition ${active ? "bg-primary/10 ring-1 ring-primary/50" : "bg-muted/40 hover:bg-muted/70"}`}>
            <span className="block text-xs text-muted-foreground">{title}</span>
            <span className="mt-0.5 block text-sm font-semibold tabular-nums">{value}</span>
        </button>
    );
}

function Stat({ title, a, b, hint }: { title: string; a: string; b?: string; hint?: string }) {
    return (
        <div className="rounded-lg bg-muted/40 px-3 py-2" title={hint}>
            <p className={`text-xs text-muted-foreground ${hint ? "w-fit cursor-help underline decoration-dotted underline-offset-2" : ""}`}>{title}</p>
            <p className="mt-0.5 flex flex-wrap items-baseline gap-x-2 text-sm tabular-nums whitespace-nowrap">
                {b == null ? <span className="font-semibold">{a}</span> : <>
                    <span className="font-semibold">A {a}</span>
                    <span className="text-muted-foreground">·</span>
                    <span className="font-semibold">B {b}</span>
                </>}
            </p>
        </div>
    );
}

function RunPanel({ tag, name, row }: { tag: "A" | "B"; name: string; row: Row }) {
    const tone = toneOf(row.final.verdict);
    const hitById = new Map(row.steps.flatMap((s) => s.result.hits ?? []).map((h) => [`${h.page}:${h.line}`, h]));
    const cited = new Set(row.final.evidence);
    const correct = row.gold == null ? null : row.final.verdict === row.gold;
    const cost = row.cost;
    return (
        <section className="min-w-0 space-y-4" aria-label={`Run ${tag}`}>
            <div className={`rounded-xl px-4 py-3 ring-1 ${tone.panel}`}>
                <p className="eyebrow" title={name}>Run {tag}</p>
                <div className="mt-1 flex items-center gap-2.5">
                    <p className="font-serif text-4xl font-semibold capitalize leading-tight">{pretty(row.final.verdict)}</p>
                    {correct != null && (
                        <span title={correct ? "correct" : "incorrect"} aria-label={correct ? "correct" : "incorrect"}
                            className={`grid size-6 shrink-0 place-items-center rounded-full text-sm font-semibold ring-1 ${correct ? "bg-success/15 text-success ring-success/30" : "bg-danger/15 text-danger ring-danger/30"}`}>
                            {correct ? "✓" : "✗"}
                        </span>
                    )}
                </div>
                <p className="mt-2 text-sm opacity-70 tabular-nums">
                    {row.final.confidence == null ? "no confidence" : `${(row.final.confidence * 100).toFixed(1)}% confidence`}
                </p>
                {cost && (
                    <div className="mt-3 flex flex-wrap items-center gap-x-1.5 gap-y-1 text-xs tabular-nums opacity-70">
                        <span>{row.steps.length} steps</span>
                        {cost.n_ops != null && <><span aria-hidden="true">·</span><span title={opsHint} className="cursor-help underline decoration-dotted underline-offset-2">{cost.n_ops} ops</span></>}
                        {cost.tokens_in != null && <><span aria-hidden="true">·</span><span>Input <strong className="font-medium">{cost.tokens_in.toLocaleString()}</strong></span></>}
                        {cost.tokens_out != null && <><span aria-hidden="true">·</span><span>Output <strong className="font-medium">{cost.tokens_out.toLocaleString()}</strong></span></>}
                        {cost.seconds != null && <><span aria-hidden="true">·</span><span>{cost.seconds.toFixed(2)}s</span></>}
                    </div>
                )}
            </div>
            {row.final.evidence.length > 0 && (
                <details open className="group text-sm">
                    <summary className="flex cursor-pointer list-none items-center gap-1.5 text-xs font-medium text-muted-foreground [&::-webkit-details-marker]:hidden">
                        <svg viewBox="0 0 20 20" className="size-3.5 shrink-0 transition group-open:rotate-90" fill="currentColor" aria-hidden>
                            <path d="M7 5l6 5-6 5V5z" />
                        </svg>
                        {row.final.evidence.length} cited {row.final.evidence.length === 1 ? "sentence" : "sentences"}
                    </summary>
                    <ul className="mt-2 space-y-2">
                        {row.final.evidence.map((id) => {
                            const hit = hitById.get(id);
                            return (
                                <li key={id} className="border-l-2 border-primary pl-3">
                                    <p className="text-xs text-muted-foreground">{hit ? <>{pretty(hit.page)} · sentence {hit.line}</> : id}</p>
                                    {hit && <p className="mt-0.5 leading-relaxed text-foreground/85">{hit.text}</p>}
                                </li>
                            );
                        })}
                    </ul>
                </details>
            )}
            <StepTimeline steps={row.steps} cited={cited} busy={false} />
        </section>
    );
}

function Detail({ pair, nameA, nameB, onBack }: { pair: Pair; nameA: string; nameB: string; onBack: () => void }) {
    const [rows, setRows] = useState<{ a: Row; b: Row } | null>(null);
    const [error, setError] = useState("");
    useEffect(() => {
        let stale = false;
        window.scrollTo({ top: 0 });
        const url = (name: string) => `/api/traces/${encodeURIComponent(name)}/${encodeURIComponent(pair.id)}`;
        Promise.all([getJson<Row>(url(nameA)), getJson<Row>(url(nameB))])
            .then(([a, b]) => !stale && setRows({ a, b }))
            .catch((e: Error) => !stale && setError(e.message));
        return () => { stale = true; };
    }, [pair.id, nameA, nameB]);

    const mismatch = rows && rows.a.run?.policy !== rows.b.run?.policy;
    return (
        <div className="min-w-0 space-y-5">
            <button type="button" onClick={onBack} className="cursor-pointer text-sm font-medium">← Back</button>
            <h2 className="relative pl-10 font-serif text-3xl leading-tight tracking-tight">
                <span aria-hidden className="absolute -top-2 left-2 select-none text-7xl leading-none text-primary/40">“</span>
                {pair.a.claim}
                <span aria-hidden className="ml-1 select-none align-[-0.3em] text-6xl leading-0 text-primary/40">”</span>
            </h2>
            {error && <p role="alert" className="rounded-xl border bg-danger/10 px-5 py-4 text-sm text-danger">{error}</p>}
            {mismatch && (
                <p className="rounded-lg bg-warning/10 px-4 py-2 text-sm text-warning ring-1 ring-warning/25">
                    These runs use different policies ({rows.a.run?.policy ?? "?"} vs {rows.b.run?.policy ?? "?"}), so cost and steps are not like for like.
                </p>
            )}
            {!rows && !error && <p className="py-10 text-center text-sm text-muted-foreground">Loading both traces…</p>}
            {rows && (
                <div className="grid items-start gap-6 xl:grid-cols-2">
                    <RunPanel tag="A" name={nameA} row={rows.a} />
                    <RunPanel tag="B" name={nameB} row={rows.b} />
                </div>
            )}
        </div>
    );
}

export function Compare({ traces, nameA, nameB, setNameA, setNameB, claimsA, claimsB, loading }: {
    traces: Trace[]; nameA: string; nameB: string; setNameA: (v: string) => void; setNameB: (v: string) => void;
    claimsA: Claim[]; claimsB: Claim[]; loading: boolean;
}) {
    const [query, setQuery] = useState("");
    const [filter, setFilter] = useState<Filter>("all");
    const [page, setPage] = useState(0);
    const [open, setOpen] = useState<Pair | null>(null);

    const pairs = useMemo<Pair[]>(() => {
        const byId = new Map(claimsB.map((c) => [String(c.id), c]));
        return claimsA.flatMap((a) => {
            const b = byId.get(String(a.id));
            return b ? [{ id: String(a.id), a, b }] : [];
        });
    }, [claimsA, claimsB]);

    if (open) return <Detail pair={open} nameA={nameA} nameB={nameB} onBack={() => setOpen(null)} />;

    const scored = pairs.filter((p) => p.a.gold != null);
    const acc = (pick: (p: Pair) => Claim) =>
        scored.length ? `${((scored.filter((p) => pick(p).correct).length / scored.length) * 100).toFixed(1)}%` : "—";
    const avgOps = (cs: Claim[]) => cs.length ? (cs.reduce((n, c) => n + (c.n_ops ?? c.n_steps), 0) / cs.length).toFixed(1) : "—";
    const counts = { differ: 0, onlyA: 0, onlyB: 0, bothWrong: 0 };
    for (const p of pairs) {
        if (p.a.verdict !== p.b.verdict) counts.differ++;
        if (matches(p, "only-a")) counts.onlyA++;
        if (matches(p, "only-b")) counts.onlyB++;
        if (matches(p, "both-wrong")) counts.bothWrong++;
    }

    const search = query.trim().toLowerCase();
    const filtered = pairs.filter((p) => matches(p, filter) && (!search || `${p.id} ${p.a.claim}`.toLowerCase().includes(search)));
    const pages = Math.max(1, Math.ceil(filtered.length / pageSize));
    const current = Math.min(page, pages - 1);
    const offset = current * pageSize;
    const visible = filtered.slice(offset, offset + pageSize);

    return (
        <section className="min-w-0 card-raised rounded-xl border bg-card" aria-label="Compare runs" aria-busy={loading}>
            <div className="space-y-4 p-5 pt-4">
                <div className="flex items-center justify-between gap-3">
                    <h2 className="text-lg font-semibold">Compare runs</h2>
                    <span className="text-sm tabular-nums text-muted-foreground">
                        {loading ? "Loading…" : <>{pairs.length} shared claims</>}
                    </span>
                </div>
                <div className="grid gap-3 md:grid-cols-2">
                    {([["A", nameA, setNameA, claimsA.length], ["B", nameB, setNameB, claimsB.length]] as const).map(([tag, name, set, n]) => (
                        <div key={tag} className="flex min-w-0 flex-col gap-1.5 text-sm font-medium">
                            <span>Run {tag} <span className="font-normal tabular-nums text-muted-foreground">· {n} rows</span></span>
                            <Select middle value={name} onChange={set} disabled={!traces.length} placeholder="No saved traces available"
                                options={traces.map((t) => ({ value: t.name, label: label(t.name) }))} />
                        </div>
                    ))}
                </div>
                {nameA && nameA === nameB && (
                    <p className="rounded-lg bg-warning/10 px-4 py-2 text-sm text-warning ring-1 ring-warning/25">Both sides are the same run. Pick two different runs.</p>
                )}
                {!loading && nameA && nameB && nameA !== nameB && pairs.length < Math.min(claimsA.length, claimsB.length) && (
                    <p className="rounded-lg bg-warning/10 px-4 py-2 text-sm text-warning ring-1 ring-warning/25">
                        Only {pairs.length} claim IDs overlap ({claimsA.length} vs {claimsB.length} rows). Differences in policy or retrieval can limit this; compare like with like.
                    </p>
                )}
                {pairs.length > 0 && (
                    <div className="space-y-2">
                        <div className="grid grid-cols-2 gap-2">
                            <Stat title="Accuracy" a={acc((p) => p.a)} b={acc((p) => p.b)} />
                            <Stat title="Avg ops per claim" hint={opsHint} a={avgOps(pairs.map((p) => p.a))} b={avgOps(pairs.map((p) => p.b))} />
                        </div>
                        <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
                            {([["differ", "Verdicts differ", counts.differ], ["only-a", "Only A correct", counts.onlyA],
                            ["only-b", "Only B correct", counts.onlyB], ["both-wrong", "Both wrong", counts.bothWrong]] as const).map(([value, title, n]) => (
                                <FilterCard key={value} title={title} value={n} active={filter === value}
                                    onClick={() => { setFilter(filter === value ? "all" : value); setPage(0); }} />
                            ))}
                        </div>
                    </div>
                )}
                <div className="flex gap-3">
                    <label className="flex min-w-0 flex-1 flex-col gap-1.5 text-sm font-medium">
                        Search
                        <input type="search" className={fieldClass} value={query} placeholder="Claim text or ID"
                            onChange={(e) => { setQuery(e.target.value); setPage(0); }} />
                    </label>
                    <div className="flex w-40 flex-col gap-1.5 text-sm font-medium" role="group" aria-label="Filter">
                        <span>Show</span>
                        <Select value={filter} onChange={(v) => { setFilter(v as Filter); setPage(0); }}
                            options={filters} />
                    </div>
                </div>
            </div>

            <div className="overflow-x-auto scrollbar-nice">
                <table className="w-full min-w-xl text-left text-sm">
                    <caption className="sr-only">Claims present in both runs. Open one to compare its traces side by side.</caption>
                    <thead className="border-y bg-muted/50 text-xs text-muted-foreground">
                        <tr>
                            {["Claim", "Gold", "A", "B"].map((h) => (
                                <th key={h} scope="col" className="whitespace-nowrap px-2.5 py-2 font-medium">{h}</th>
                            ))}
                        </tr>
                    </thead>
                    <tbody className="divide-y">
                        {loading ? <tr><td colSpan={4} className="px-5 py-10 text-center text-muted-foreground">Loading runs…</td></tr>
                            : !visible.length ? <tr><td colSpan={4} className="px-5 py-10 text-center text-muted-foreground">
                                {pairs.length ? "No claims match your search or filter." : "These runs share no claim IDs."}
                            </td></tr> : visible.map((p) => (
                                <tr key={p.id} onClick={() => setOpen(p)} className="cursor-pointer hover:bg-muted/50">
                                    <td className="w-full max-w-0 px-2.5 py-2">
                                        <button type="button" title={p.a.claim} className="block w-full cursor-pointer truncate text-left font-medium hover:underline">
                                            <span className="mr-2 inline-block rounded border bg-muted/40 px-1.5 align-[0.1em] font-mono text-[11px] font-normal tabular-nums text-muted-foreground">{p.id}</span>
                                            {p.a.claim}
                                        </button>
                                    </td>
                                    <td className="whitespace-nowrap px-2.5 py-2 capitalize text-muted-foreground">{p.a.gold == null ? "—" : pretty(p.a.gold)}</td>
                                    <td className="whitespace-nowrap px-2.5 py-2"><Verdict claim={p.a} /></td>
                                    <td className="whitespace-nowrap pl-2.5 pr-4.5 py-2"><Verdict claim={p.b} /></td>
                                </tr>
                            ))}
                    </tbody>
                </table>
            </div>

            <div className="flex flex-wrap items-center justify-between gap-3 border-t px-5 py-4 text-sm">
                <p className="text-muted-foreground" aria-live="polite">
                    {filtered.length ? `${offset + 1}-${Math.min(offset + pageSize, filtered.length)} of ${filtered.length} claims` : "0 claims"}
                </p>
                <div className="flex items-center gap-3">
                    <button type="button" disabled={loading || current === 0} onClick={() => setPage(current - 1)}
                        className="cursor-pointer rounded-lg border px-3 py-1.5 disabled:cursor-not-allowed disabled:opacity-40">Previous</button>
                    <span className="text-xs text-muted-foreground">{current + 1} / {pages}</span>
                    <button type="button" disabled={loading || current + 1 >= pages} onClick={() => setPage(current + 1)}
                        className="cursor-pointer rounded-lg border px-3 py-1.5 disabled:cursor-not-allowed disabled:opacity-40">Next</button>
                </div>
            </div>
        </section>
    );
}
