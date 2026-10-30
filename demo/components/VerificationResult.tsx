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

function ConfidenceBar({ confidence, dot }: { confidence: number; dot: string; }) {
    const target = confidence * 100;
    const animated = useCountUp(target);

    return (
        <div className="mt-3 max-w-lg">
            <div className="mb-2 flex justify-between text-xs opacity-70">
                <span>Confidence</span>
                <span className="tabular-nums">
                    {animated.toFixed(1)}%
                </span>
            </div>

            <div className="relative h-2.5 overflow-hidden rounded-full bg-foreground/10">
                <div
                    className={`h-full rounded-full ${dot}`}
                    style={{
                        width: `${animated}%`,
                    }}
                />
                {[25, 50, 75].map((tick) => (
                    <span key={tick} className="absolute inset-y-0 w-px bg-background/70" style={{ left: `${tick}%` }} />
                ))}
            </div>
            <div className="relative mt-1 h-3 text-[10px] tabular-nums opacity-50">
                {[0, 25, 50, 75, 100].map((tick) => (
                    <span key={tick} className="absolute -translate-x-1/2 first:translate-x-0 last:-translate-x-full" style={{ left: `${tick}%` }}>{tick}</span>
                ))}
            </div>
        </div>
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
                    <h2 className="font-serif text-2xl">Nothing checked yet</h2>
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
                                    <span className="grid size-16 shrink-0 place-items-center rounded-full bg-current/12 ring-1 ring-current/30">
                                        <span className="text-3xl font-semibold leading-none">{t.icon}</span>
                                    </span>
                                    <div className="min-w-0">
                                        <span className="text-xs font-medium uppercase tracking-widest opacity-70">Verdict</span>
                                        <p className="text-5xl font-semibold capitalize leading-tight tracking-tight">
                                            {pretty(row.final.verdict)}
                                        </p>
                                    </div>
                                </div>
                                {row.final.confidence != null && (
                                    <ConfidenceBar
                                        confidence={row.final.confidence}
                                        dot={t.dot}
                                    />
                                )}
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
                                    <Shimmer className="grid size-16 shrink-0 place-items-center rounded-full">
                                        <span className="relative size-5 animate-spin rounded-full border-2 border-foreground/15 border-t-foreground/60" />
                                    </Shimmer>
                                    <div className="min-w-0">
                                        <span className="text-xs font-medium uppercase tracking-widest opacity-70" role="status">{status}</span>
                                        {/* same 60px slot as the verdict's text-5xl leading-tight line */}
                                        <div className="flex h-15 items-center">
                                            <Shimmer className="h-10 w-56 rounded-md" />
                                        </div>
                                    </div>
                                </div>
                                <div className="mt-3 max-w-lg">
                                    <div className="mb-2 text-xs opacity-70">Confidence</div>
                                    <div className="h-2.5 overflow-hidden rounded-full bg-foreground/10">
                                        <div
                                            className={`h-full w-1/3 rounded-full ${t.dot}`}
                                            style={{ animation: "indeterminate 2s ease-in-out infinite" }}
                                        />
                                    </div>
                                    <div className="mt-1 h-3" />
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
