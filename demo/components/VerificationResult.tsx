import { pretty, toneOf, type Row, type Step } from "../app";
import { StepTimeline } from "./StepTimeline";
import { EvidenceList } from "./EvidenceList";
import { useEffect, useState } from "react";

function useCountUp(target: number, duration = 800) {
    const [value, setValue] = useState(0);

    useEffect(() => {
        if (target == null) return;

        let frame: number;
        const start = performance.now();

        const animate = (now: DOMHighResTimeStamp) => {
            const progress = Math.min((now - start) / duration, 1);
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
        <div className="mt-3 max-w-md">
            <div className="mb-1.5 flex justify-between text-xs opacity-70">
                <span>Confidence</span>
                <span className="tabular-nums">
                    {animated.toFixed(1)}%
                </span>
            </div>

            <div className="h-1.5 overflow-hidden rounded-full bg-foreground/10">
                <div
                    className={`h-full rounded-full ${dot}`}
                    style={{
                        width: `${animated}%`,
                    }}
                />
            </div>
        </div>
    );
}

export function VerificationResult({ claim, row, steps, status, busy }: { claim: string; row: Row | null; steps: Step[]; status: string; busy: boolean }) {
    const hits = [...new Map(steps.flatMap(s => s.result.hits ?? []).map(h => [`${h.page}:${h.line}`, h])).values()];
    const cited = new Set(row?.final.evidence ?? []);
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
                <h2 className="font-serif text-3xl tracking-tight">{claim}</h2>

                {(row || busy) && (
                    <div className={`rounded-xl p-6 pt-5 ring-1 ${t.box}`}>
                        <div className="flex items-center gap-2.5">
                            {row
                                ? <span className={`size-3 rounded-full ${t.dot}`} />
                                : <span className="size-3 animate-spin rounded-full border-2 border-foreground/15 border-t-foreground/60" />}
                            <span className="text-sm mb-0.5 font-medium opacity-70" role="status">
                                {row ? "Verdict" : status}
                            </span>
                        </div>

                        {row ? (
                            <>
                                <p className="mt-2 text-3xl font-semibold capitalize tracking-tight">
                                    {pretty(row.final.verdict)}
                                </p>
                                {row.final.confidence != null && (
                                    <ConfidenceBar
                                        confidence={row.final.confidence}
                                        dot={t.dot}
                                    />
                                )}
                            </>
                        ) : (
                            <div aria-hidden="true">
                                <div className="relative mt-2 h-9 w-40 overflow-hidden rounded-md bg-muted">
                                    <div
                                        className="absolute inset-y-0 left-0 w-1/2 bg-linear-to-r from-transparent via-muted-foreground/20 to-transparent"
                                        style={{ animation: "shimmer 2s ease-in-out infinite" }}
                                    />
                                </div>
                                <div className="mt-3 max-w-md">
                                    <div className="mb-1.5 text-xs opacity-70">Confidence</div>
                                    <div className="h-1.5 overflow-hidden rounded-full bg-foreground/10">
                                        <div
                                            className={`h-full w-1/3 rounded-full ${t.dot}`}
                                            style={{ animation: "indeterminate 2s ease-in-out infinite" }}
                                        />
                                    </div>
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
