import { pretty, toneOf, type Row, type Step } from "../app";
import { StepTimeline } from "./StepTimeline";
import { useEffect, useState } from "react";

function useCountUp(target: number, duration = 800) {
    const [value, setValue] = useState(0);

    useEffect(() => {
        if (target == null) return;

        let frame: number;
        const start = performance.now();

        const animate = (now: DOMHighResTimeStamp) => {
            // rAF can hand us a timestamp from before this effect ran, so clamp both ends.
            const progress = Math.min(Math.max((now - start) / duration, 0), 1);
            const eased = 1 - Math.pow(1 - progress, 3);

            setValue(target * eased);

            if (progress < 1) {
                frame = requestAnimationFrame(animate);
            }
        };

        frame = requestAnimationFrame(animate);

        return () => cancelAnimationFrame(frame);
    }, [target, duration]);

    return value;
}

function ConfidenceRing({ confidence, icon }: { confidence: number | null; icon: string }) {
    const animated = useCountUp(confidence == null ? 0 : confidence * 100);
    const r = 34, c = 2 * Math.PI * r;
    return (
        <span className="relative grid size-20 shrink-0 place-items-center">
            <svg viewBox="0 0 80 80" className="absolute inset-0 -rotate-90" fill="none" aria-hidden>
                <circle cx="40" cy="40" r={r} stroke="currentColor" strokeOpacity="0.15" strokeWidth="5" />
                {confidence != null && (
                    <circle cx="40" cy="40" r={r} stroke="currentColor" strokeWidth="5" strokeLinecap="round"
                        strokeDasharray={c} strokeDashoffset={c * (1 - animated / 100)} />
                )}
            </svg>
            <span className="font-serif text-3xl font-semibold leading-none">{icon}</span>
        </span>
    );
}

function Shimmer({ className, children }: { className: string; children?: React.ReactNode }) {
    return (
        <div className={`relative overflow-hidden bg-muted ${className}`}>
            <div
                className="absolute inset-y-0 left-0 w-1/2 bg-linear-to-r from-transparent via-muted-foreground/20 to-transparent"
                style={{ animation: "shimmer 2s ease-in-out infinite" }}
            />
            {children}
        </div>
    );
}

export function VerificationResult({ claim, row, steps, status, busy }: { claim: string; row: Row | null; steps: Step[]; status: string; busy: boolean }) {
    const hitById = new Map(steps.flatMap(s => s.result.hits ?? []).map(h => [`${h.page}:${h.line}`, h]));
    const citedIds = row?.final.evidence ?? [];
    const cited = new Set(citedIds);
    const t = toneOf(row?.final.verdict);
    return (
        <section className="min-w-0 space-y-6" aria-busy={busy}>
            {!claim ? (
                <div className="rounded-xl border border-dashed px-8 py-20 text-center">
                    <h2 className="font-serif text-3xl">Nothing checked yet</h2>
                    <p className="mx-auto mt-3 max-w-sm text-sm text-foreground/60">
                        Choose a claim and run it. The agent's searches, the sentences it found and the sources behind the verdict will show up here.
                    </p>
                </div>
            ) : <>
                <h2 className="relative pl-10 font-serif text-3xl leading-tight tracking-tight">
                    <span aria-hidden className="absolute -top-2 left-2 select-none text-7xl leading-none text-primary/40">“</span>
                    {claim}
                    <span aria-hidden className="ml-1 select-none align-[-0.3em] text-6xl leading-0 text-primary/40">”</span>
                </h2>

                {(row || busy) && (
                    <div
                        key={row ? "done" : "pending"}
                        className={`rounded-xl p-5 pt-4 my-8 ring-1 ${t.panel} ${row ? t.glow : ""}`}
                        style={{ animation: "verdict-in 400ms ease-out" }}
                    >
                        {row ? (
                            <>
                                <div className="flex items-center gap-5">
                                    <ConfidenceRing confidence={row.final.confidence} icon={t.icon} />
                                    <div className="min-w-0">
                                        <span className="eyebrow">Verdict</span>
                                        <p className="font-serif text-6xl font-semibold capitalize leading-tight tracking-tight">
                                            {pretty(row.final.verdict)}
                                        </p>
                                        {row.final.confidence != null && (
                                            <p className="mt-1 text-sm tabular-nums opacity-70">{(row.final.confidence * 100).toFixed(1)}% confidence</p>
                                        )}
                                    </div>
                                </div>
                                {citedIds.length > 0 && (
                                    <ul className="mt-5 space-y-3 border-t border-current/15 pt-4 text-foreground">
                                        {citedIds.map((id) => {
                                            const hit = hitById.get(id);
                                            return (
                                                <li key={id} className="text-sm">
                                                    <p className="text-xs opacity-70">
                                                        {hit ? <>{pretty(hit.page)} · sentence {hit.line}</> : id}
                                                    </p>
                                                    {hit && <p className="mt-0.5 leading-relaxed text-foreground/85">{hit.text}</p>}
                                                </li>
                                            );
                                        })}
                                    </ul>
                                )}
                            </>
                        ) : (
                            <div>
                                <div className="flex items-center gap-5">
                                    <Shimmer className="grid size-20 shrink-0 place-items-center rounded-full">
                                        <span className="relative size-5 animate-spin rounded-full border-2 border-foreground/15 border-t-foreground/60" />
                                    </Shimmer>
                                    <div className="min-w-0">
                                        <span className="eyebrow" role="status">{status}</span>
                                        {/* same slots as the verdict word (text-6xl leading-tight = 75px) and the confidence line (mt-1 + text-sm = 24px) */}
                                        <div className="flex h-18.75 items-center">
                                            <Shimmer className="h-12 w-64 rounded-md" />
                                        </div>
                                        <div className="mt-1 flex h-5 items-center">
                                            <Shimmer className="h-3 w-32 rounded" />
                                        </div>
                                    </div>
                                </div>
                            </div>
                        )}
                    </div>
                )}
                <StepTimeline steps={steps} cited={cited} busy={busy} />
            </>}
        </section>
    );
}
