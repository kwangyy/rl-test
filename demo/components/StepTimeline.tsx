import { pretty, toneOf, type Step } from "../app"

function VerdictBadge({ verdict }: { verdict?: string | null }) {
    const t = toneOf(verdict);
    return (
        <span className={`inline-flex shrink-0 items-center gap-1.5 whitespace-nowrap rounded-full px-2.5 py-0.5 text-xs font-medium capitalize ring-1 ${t.box}`}>
            <span className={`size-1.5 rounded-full ${t.dot}`} />
            {pretty(verdict)}
        </span>
    );
}

export function StepTimeline({ steps }: { steps: Step[] }) {
    return (
        <ol className="relative space-y-3 before:absolute before:bottom-4 before:left-3 before:top-4 before:w-px before:bg-border">
            {steps.map((step) => (
                <li key={step.t} className="relative pl-10">
                    <span className="absolute left-0 top-1 z-10 grid size-7 place-items-center rounded-full border bg-card text-xs font-semibold text-muted-foreground">
                        {step.t + 1}
                    </span>
                    <div className="space-y-2 rounded-lg bg-card p-3 py-2 text-sm text-muted-foreground shadow-sm ring-1 ring-border">
                        <div className="flex min-w-0 items-center gap-2">
                            <h3 className="shrink-0 font-medium text-foreground capitalize">{step.action}</h3>
                            {"verdict" in step.result && <VerdictBadge verdict={step.result.verdict} />}
                            {step.args.query &&
                                <p title={step.args.query} className="min-w-0 flex-1 truncate">{step.args.query}</p>}
                            <span className={`${step.args.query ? "" : "ml-auto "}flex shrink-0 items-center gap-2 text-xs tabular-nums`}>
                                {step.result.hits && (
                                    <span className="whitespace-nowrap">{step.result.hits.length} hits</span>
                                )}
                                <span className="whitespace-nowrap">{step.seconds.toFixed(2)}s</span>
                            </span>
                        </div>
                        {step.result.subclaims?.length ? (
                            <ol className="space-y-1.5 rounded-md bg-muted/50 p-2 mb-1">
                                {step.result.subclaims.map((text, i) =>
                                    <li key={i} className="flex gap-2.5">
                                        <span className="inline-flex size-5 shrink-0 items-center justify-center rounded-full bg-card text-xs font-semibold text-muted-foreground ring-1 ring-border">
                                            {i + 1}
                                        </span>
                                        <p className="flex-1 leading-5">{text}</p>
                                    </li>)}
                            </ol>
                        ) : null}
                        {step.result.reasoning && (
                            <details open>
                                <summary className="cursor-pointer text-xs font-medium text-muted-foreground">Reasoning</summary>
                                <p className="mt-1 whitespace-pre-line leading-relaxed">{step.result.reasoning}</p>
                            </details>
                        )}
                    </div>
                </li>
            ))}
        </ol>
    )
}
