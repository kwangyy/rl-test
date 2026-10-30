import { useState } from "react";
import { pretty, toneOf, type Claim } from "../app";
import { fieldClass, Select } from "./ClaimForm";

type Outcome = "correct" | "incorrect" | "invalid" | "unscored";
const outcomeOf = (claim: Claim): Outcome =>
    !claim.verdict ? "invalid" : claim.gold == null ? "unscored" : claim.correct ? "correct" : "incorrect";
const pageSize = 25;
const outcomeMark: Record<Outcome, { icon: string; cls: string }> = {
    correct: { icon: "✓", cls: "text-success" },
    incorrect: { icon: "✕", cls: "text-danger" },
    invalid: { icon: "!", cls: "text-warning" },
    unscored: { icon: "-", cls: "text-muted-foreground" },
};

export function TraceTable({ claims, loading, busy, selectedId, onOpen }: {
    claims: Claim[];
    loading: boolean;
    busy: boolean;
    selectedId: string;
    onOpen: (claim: Claim) => void;
}) {
    const [query, setQuery] = useState("");
    const [outcome, setOutcome] = useState<Outcome | "all">("all");
    const [page, setPage] = useState(0);
    const counts = { correct: 0, incorrect: 0, invalid: 0, unscored: 0 };
    for (const claim of claims) counts[outcomeOf(claim)]++;
    const scored = counts.correct + counts.incorrect + counts.invalid;
    const search = query.trim().toLowerCase();
    const filtered = claims.filter(claim =>
        (!search || `${claim.id} ${claim.claim}`.toLowerCase().includes(search)) &&
        (outcome === "all" || outcomeOf(claim) === outcome));
    const pages = Math.max(1, Math.ceil(filtered.length / pageSize));
    const currentPage = Math.min(page, pages - 1);
    const offset = currentPage * pageSize;
    const visible = filtered.slice(offset, offset + pageSize);

    return (
        <section className="min-w-0 rounded-xl border bg-card shadow-sm" aria-label="Saved claims" aria-busy={loading}>
            <div className="space-y-3 p-5 pt-4">
                <div className="flex items-center justify-between gap-3">
                    <h2 className="text-lg font-semibold">Saved claims</h2>
                    <span className="text-sm text-muted-foreground tabular-nums">
                        {claims.length} claims{scored > 0 && <> · <span className="font-medium text-foreground">{((counts.correct / scored) * 100).toFixed(1)}%</span> correct</>}
                    </span>
                </div>
                {scored > 0 && (
                    <div className="flex h-1 overflow-hidden rounded-full bg-muted">
                        <div className="bg-success" style={{ width: `${(counts.correct / claims.length) * 100}%` }} />
                        <div className="bg-danger" style={{ width: `${(counts.incorrect / claims.length) * 100}%` }} />
                        <div className="bg-warning" style={{ width: `${(counts.invalid / claims.length) * 100}%` }} />
                    </div>
                )}
                <div className="flex gap-3">
                    <label className="flex min-w-0 flex-1 flex-col gap-1.5 text-sm font-medium">
                        Search
                        <input type="search" className={fieldClass} value={query} placeholder="Claim text or ID"
                            onChange={event => { setQuery(event.target.value); setPage(0); }} />
                    </label>
                    <div className="flex flex-col gap-1.5 text-sm font-medium w-32" role="group" aria-label="Outcome">
                        <span>Outcome</span>
                        <Select value={outcome}
                            onChange={value => { setOutcome(value as Outcome | "all"); setPage(0); }}
                            options={[
                                { value: "all", label: "All" },
                                { value: "correct", label: "Correct" },
                                { value: "incorrect", label: "Incorrect" },
                                { value: "invalid", label: "Invalid" },
                                { value: "unscored", label: "Unscored" },
                            ]} />
                    </div>
                </div>
            </div>

            <div className="overflow-x-auto scrollbar-nice">
                <table className="w-full min-w-xl text-left text-sm">
                    <caption className="sr-only">Saved claim results. Open a claim to inspect its complete trace.</caption>
                    <thead className="border-y bg-muted/50 text-xs text-muted-foreground">
                        <tr>
                            {["Claim", "Gold", "Verdict", "Confidence", "Steps", ""].map((label, i) =>
                                <th key={i} scope="col"
                                    className={`whitespace-nowrap px-2.5 py-2 font-medium ${i === 3 || i === 4 ? "text-right" : ""}`}>{label}</th>)}
                        </tr>
                    </thead>
                    <tbody className="divide-y">
                        {loading ? <tr><td colSpan={6} className="px-5 py-10 text-center text-muted-foreground">Loading saved claims…</td></tr>
                            : !visible.length ? <tr><td colSpan={6} className="px-5 py-10 text-center text-muted-foreground">
                                {claims.length ? "No claims match your search or filter." : "No saved claims available."}
                            </td></tr> : visible.map(claim => {
                                const result = outcomeOf(claim);
                                const verdictTone = toneOf(claim.verdict);
                                return (
                                    <tr key={String(claim.id)} onClick={() => !busy && onOpen(claim)}
                                        className={`cursor-pointer hover:bg-muted/50 ${selectedId === String(claim.id) ? "bg-muted/40" : ""}`}>
                                        <td className="w-full max-w-0 px-2.5 py-2">
                                            <button type="button" disabled={busy} aria-pressed={selectedId === String(claim.id)} title={claim.claim}
                                                className="block w-full cursor-pointer truncate text-left font-medium hover:underline disabled:cursor-wait">
                                                <span className="mr-2 inline-block rounded border bg-muted/40 px-1.5 align-[0.1em] font-mono text-[11px] font-normal tabular-nums text-muted-foreground">{claim.id}</span>
                                                {claim.claim}
                                            </button>
                                        </td>
                                        <td className="whitespace-nowrap px-2.5 py-2 capitalize text-muted-foreground">{claim.gold == null ? "—" : pretty(claim.gold)}</td>
                                        <td className="whitespace-nowrap px-2.5 py-2">
                                            <span className={`inline-flex items-center gap-1.5 rounded-full px-2 py-0.5 text-xs font-medium capitalize ring-1 ${verdictTone.box}`}>
                                                {claim.verdict && <span className={`size-1.5 shrink-0 rounded-full ${verdictTone.mark}`} />}
                                                {pretty(claim.verdict)}
                                            </span>
                                        </td>
                                        <td className="whitespace-nowrap px-2.5 py-2 text-center tabular-nums">{claim.confidence == null ? "—" : `${(claim.confidence * 100).toFixed(1)}%`}</td>
                                        <td className="whitespace-nowrap px-2.5 py-2 text-center tabular-nums text-muted-foreground">{claim.n_steps}</td>
                                        <td title={result} className={`w-8 whitespace-nowrap pl-2.5 pr-4.5 py-2 text-center font-semibold ${outcomeMark[result].cls}`}>{outcomeMark[result].icon}</td>
                                    </tr>
                                );
                            })}
                    </tbody>
                </table>
            </div>

            <div className="flex flex-wrap items-center justify-between gap-3 border-t px-5 py-4 text-sm">
                <p className="text-muted-foreground" aria-live="polite">
                    {filtered.length ? `${offset + 1}-${Math.min(offset + pageSize, filtered.length)} of ${filtered.length} claims` : "0 claims"}
                </p>
                <div className="flex items-center gap-3">
                    <button type="button" disabled={loading || currentPage === 0} onClick={() => setPage(currentPage - 1)}
                        className="cursor-pointer rounded-lg border px-3 py-1.5 disabled:cursor-not-allowed disabled:opacity-40">Previous</button>
                    <span className="text-xs text-muted-foreground">{currentPage + 1} / {pages}</span>
                    <button type="button" disabled={loading || currentPage + 1 >= pages} onClick={() => setPage(currentPage + 1)}
                        className="cursor-pointer rounded-lg border px-3 py-1.5 disabled:cursor-not-allowed disabled:opacity-40">Next</button>
                </div>
            </div>
        </section>
    );
}
