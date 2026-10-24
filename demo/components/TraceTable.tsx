import { useState } from "react";
import { pretty, toneOf, type Claim } from "../app";
import { fieldClass, Select } from "./ClaimForm";

type Outcome = "correct" | "incorrect" | "invalid" | "unscored";
const outcomeOf = (claim: Claim): Outcome =>
    !claim.verdict ? "invalid" : claim.gold == null ? "unscored" : claim.correct ? "correct" : "incorrect";
const pageSize = 25;

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
            <div className="space-y-4 px-5 py-4">
                <div className="flex items-center justify-between gap-3">
                    <h2 className="text-lg font-semibold">Saved claims</h2>
                    <span className="text-sm text-muted-foreground">{claims.length} claims</span>
                </div>
                <div className="flex gap-3">
                    <label className="flex min-w-0 flex-1 flex-col gap-1.5 text-sm font-medium">
                        Search
                        <input type="search" className={fieldClass} value={query} placeholder="Claim text or ID"
                            onChange={event => { setQuery(event.target.value); setPage(0); }} />
                    </label>
                    <div className="flex flex-col gap-1.5 text-sm font-medium w-44" role="group" aria-label="Outcome">
                        <span>Outcome</span>
                        <Select value={outcome}
                            onChange={value => { setOutcome(value as Outcome | "all"); setPage(0); }}
                            options={[
                                { value: "all", label: "All outcomes" },
                                { value: "correct", label: "Correct" },
                                { value: "incorrect", label: "Incorrect" },
                                { value: "invalid", label: "Invalid verdict" },
                                { value: "unscored", label: "Unscored" },
                            ]} />
                    </div>
                </div>
            </div>

            <div className="overflow-x-auto scrollbar-nice">
                <table className="w-full min-w-2xl text-left text-sm">
                    <caption className="sr-only">Saved claim results. Open a claim to inspect its complete trace.</caption>
                    <thead className="border-y bg-muted/50 text-xs text-muted-foreground">
                        <tr>
                            {["Claim", "Gold", "Verdict", "Confidence", "Outcome", "Steps"].map(label =>
                                <th key={label} scope="col" className="px-4 py-3 font-medium">{label}</th>)}
                        </tr>
                    </thead>
                    <tbody className="divide-y">
                        {loading ? <tr><td colSpan={6} className="px-5 py-10 text-center text-muted-foreground">Loading saved claims…</td></tr>
                            : !visible.length ? <tr><td colSpan={6} className="px-5 py-10 text-center text-muted-foreground">
                                {claims.length ? "No claims match your search or filter." : "No saved claims available."}
                            </td></tr> : visible.map(claim => {
                                const result = outcomeOf(claim);
                                const goldTone = toneOf(claim.gold);
                                const verdictTone = toneOf(claim.verdict);
                                return (
                                    <tr key={String(claim.id)} onClick={() => !busy && onOpen(claim)}
                                        className={`cursor-pointer align-top hover:bg-muted/50 ${selectedId === String(claim.id) ? "bg-muted/40" : ""}`}>
                                        <td className="w-full px-4 py-3">
                                            <button type="button" disabled={busy} aria-pressed={selectedId === String(claim.id)}
                                                className="cursor-pointer text-left font-medium leading-relaxed hover:underline disabled:cursor-wait">
                                                {claim.claim}
                                            </button>
                                            <p className="mt-1 text-xs text-muted-foreground">ID {claim.id}</p>
                                        </td>
                                        <td className={`px-4 py-3 capitalize ${goldTone.box} bg-transparent`}>{claim.gold == null ? "—" : pretty(claim.gold)}</td>
                                        <td className={`px-4 py-3 capitalize ${verdictTone.box} bg-transparent`}>{pretty(claim.verdict)}</td>
                                        <td className="px-4 py-3 tabular-nums">{claim.confidence == null ? "—" : `${(claim.confidence * 100).toFixed(1)}%`}</td>
                                        <td className={`px-4 py-3 capitalize ${result === "correct" ? "text-emerald-700" : result === "incorrect" ? "text-rose-700" : "text-muted-foreground"}`}>{result}</td>
                                        <td className="px-4 py-3 tabular-nums">{claim.n_steps}</td>
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
