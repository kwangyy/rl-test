import { pretty, toneOf, type Row, type Step } from "../app";
import { StepTimeline } from "./StepTimeline";
import { EvidenceList } from "./EvidenceList";

export function VerificationResult({ claim, row, steps, status, busy }: { claim: string; row: Row | null; steps: Step[]; status: string; busy: boolean }) {
    const hits = [...new Map(steps.flatMap(s => s.result.hits ?? []).map(h => [`${h.page}:${h.line}`, h])).values()];
    const cited = new Set(row?.final.evidence ?? []);
    const t = toneOf(row?.final.verdict);
    return (
        <section className="min-w-0 space-y-6" aria-busy={busy}>
            <div className="flex items-center gap-2.5 text-sm text-muted-foreground" role="status">
                {busy
                    ? <span className="size-3.5 animate-spin rounded-full border-2 bordering-primary/10 border-t-primary" />
                    : <span className="size-2 rounded-full bg-muted-foreground/50" />}
                <span className="pb-0.5">{status}</span>
            </div>

            {!claim ? (
                <div className="rounded-xl border border-dashed px-8 py-20 text-center">
                    <h2 className="font-serif text-2xl">Nothing checked yet</h2>
                    <p className="mx-auto mt-3 max-w-sm text-sm text-foreground/60">
                        Choose a claim and run it. The agent's searches, the sentences it found and the sources behind the verdict will show up here.
                    </p>
                </div>
            ) : <>
                <h2 className="max-w-3xl font-serif text-3xl tracking-tight">{claim}</h2>

                {row && (
                    <div className={`rounded-xl p-6 pt-5 ring-1 ${t.box}`}>
                        <div className="flex items-center gap-2.5">
                            <span className={`size-2.5 rounded-full ${t.dot}`} />
                            <span className="text-sm font-medium opacity-70">Verdict</span>
                        </div>
                        <p className="mt-2 text-3xl font-semibold capitalize tracking-tight">{pretty(row.final.verdict)}</p>
                        {row.final.confidence != null && (
                            <div className="mt-3 max-w-md">
                                <div className="mb-1.5 flex justify-between text-xs opacity-70">
                                    <span>Model-stated confidence</span>
                                    <span className="tabular-nums">{(row.final.confidence * 100).toFixed(1)}%</span>
                                </div>
                                <div className="h-1.5 overflow-hidden rounded-full bg-foreground/10">
                                    <div className={`h-full ${t.dot}`} style={{ width: `${row.final.confidence * 100}%` }} />
                                </div>
                            </div>
                        )}
                    </div>
                )}

                <StepTimeline steps={steps} />
                <EvidenceList hits={hits} cited={cited} />
            </>}
        </section>
    );
}
