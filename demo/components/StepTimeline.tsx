import { pretty, toneOf, type Step } from "../app"

function VerdictBadge({ verdict }: { verdict?: string | null }) {
    const t = toneOf(verdict);
    return (
        <span className={`inline-flex px-2.5 py-1 items-center gap-1.5 rounded-full capitalize text-xs font-medium ring-1 ${t.box}`}>
            <span className={`size-1.5 rounded-full ${t.dot}`} />
            {pretty(verdict)}
        </span>
    );
}

export function StepTimeline({ steps }: { steps: Step[] }) {
    return (
        <ol className="relative space-y-4 before:absolute before:bottom-4 before:left-4 before:top-4 before:w-px before:bg-border">
            {steps.map((step) => (
                <li key={step.t} className="relative pl-12">
                    <span className="absolute left-0 top-2 z-10 grid size-8 place-items-center rounded-full bg-card text-xs font-semibold text-muted-foreground border">
                        {step.t + 1}
                    </span>
                    <div className="space-y-3 rounded-lg bg-card p-4 text-sm text-muted-foreground shadow-sm ring-1 ring-border">
                        <div className="flex items-center justify-between">
                            <h3 className="font-medium text-foreground capitalize">{step.action}</h3>
                            <span className="text-xs tabular-nums text-muted-foreground">{step.seconds.toFixed(2)}s</span>
                        </div>
                        {step.args.query && <p className="rounded-md bg-muted px-3 py-2">{step.args.query}</p>}
                        {step.result.hits && <p>Found {step.result.hits.length} evidence sentences</p>}
                        {"verdict" in step.result && <VerdictBadge verdict={step.result.verdict} />}
                        {step.result.subclaims?.map((text, i) =>
                            <div key={i} className="flex gap-2.5">
                                <span className="inline-flex size-5 shrink-0 items-center justify-center rounded-full bg-muted text-xs text-muted-foreground font-semibold">
                                    {i + 1}
                                </span>
                                <p className="flex-1 leading-5">{text}</p>
                            </div>)}
                    </div>
                </li>
            ))}
        </ol>
    );
}
