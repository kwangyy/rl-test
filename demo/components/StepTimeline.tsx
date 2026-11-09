import type { ReactNode } from "react";
import { pretty, toneOf, type Step } from "../app"

function VerdictBadge({ verdict }: { verdict?: string | null }) {
    const t = toneOf(verdict);
    return (
        <span className={`inline-flex shrink-0 items-center gap-1.5 whitespace-nowrap rounded-full px-2.5 py-0.5 text-xs font-medium capitalize ring-1 ${t.box}`}>
            <span className={`size-1.5 shrink-0 rounded-full ${t.mark}`} />
            {pretty(verdict)}
        </span>
    );
}

const svgProps = {
    viewBox: "0 0 20 20",
    className: "size-4",
    fill: "none",
    stroke: "currentColor",
    strokeWidth: 1.8,
    strokeLinecap: "round",
    strokeLinejoin: "round",
} as const;

const SearchIcon = (
    <svg {...svgProps}>
        <circle cx="11.5" cy="8.5" r="5" />
        <path d="M7.7 12.3L3.5 16.5" />
    </svg>
);

const ForkIcon = (
    <svg {...svgProps}>
        <circle cx="5" cy="4.5" r="1.8" />
        <circle cx="15" cy="4.5" r="1.8" />
        <circle cx="10" cy="15.5" r="1.8" />
        <path d="M5 6.3v1.2a2 2 0 0 0 2 2h6a2 2 0 0 0 2-2V6.3M10 9.5v4.2" />
    </svg>
);

const LinkIcon = (
    <svg {...svgProps}>
        <path d="M8.5 11.5a3 3 0 0 0 4.2 0l2.6-2.6a3 3 0 0 0-4.2-4.2l-1 1" />
        <path d="M11.5 8.5a3 3 0 0 0-4.2 0l-2.6 2.6a3 3 0 0 0 4.2 4.2l1-1" />
    </svg>
);

const PageIcon = (
    <svg {...svgProps}>
        <path d="M5.5 2.5h6l3 3v12h-9z" />
        <path d="M11.5 2.5v3h3" />
    </svg>
);

const link = "bg-primary/10 text-primary ring-primary/30";
const actionStyles: Record<string, { icon: ReactNode; num: string }> = {
    search: { icon: SearchIcon, num: "bg-primary/10 text-primary ring-primary/30" },
    decompose: { icon: ForkIcon, num: "bg-warning/10 text-warning ring-warning/30" },
    verify: { icon: "✓", num: "bg-success/10 text-success ring-success/30" },
    follow: { icon: LinkIcon, num: link },
    open_page: { icon: PageIcon, num: link },
    find_pages: { icon: SearchIcon, num: link },
    web_pages: { icon: SearchIcon, num: link },
    commit: { icon: "⚑", num: "bg-success/10 text-success ring-success/30" },
    abstain: { icon: "∅", num: "bg-muted text-muted-foreground ring-border" },
};
const fallbackStyle = { icon: "•", num: "bg-card text-muted-foreground ring-border" };

export function StepTimeline({ steps, cited, busy, active, onSelect }: {
    steps: Step[]; cited: Set<string>; busy: boolean;
    /** the step the architecture diagram is showing; clicking a step moves the diagram there */
    active?: number; onSelect?: (i: number) => void;
}) {
    return (
        <ol className="relative space-y-3">
            {steps.map((step, i) => {
                const style = actionStyles[step.action] ?? fallbackStyle;
                // what the step acted on: a query, a page title, or the page a followed link opened
                const target = (step.args.query ?? step.args.title ?? step.args.page ?? step.args.target)?.replaceAll("_", " ");
                const empty = step.result.status && step.result.status !== "ok" && !step.result.hits?.length && !step.result.pages?.length;
                return (
                    <li key={step.t} className="relative pl-10" style={{ animation: "step-in 300ms ease-out" }}
                        onClick={onSelect && ((e) => { if (!(e.target as HTMLElement).closest("summary, details[open] > *:not(summary)")) onSelect(i); })}>
                        <span className={`absolute left-0 top-1 z-10 grid size-7 place-items-center rounded-full text-sm font-semibold ring-1 ${style.num}`}>
                            {style.icon}
                        </span>
                        <div className={`space-y-2 card-raised rounded-lg bg-card p-3 py-2 text-sm text-muted-foreground transition ${active === i ? "ring-2 ring-primary/70" : "ring-1 ring-border"} ${onSelect ? "cursor-pointer hover:ring-primary/40" : ""}`}>
                            <div className="flex min-w-0 items-center gap-2">
                                <h3 className="flex items-center gap-1.5 shrink-0 font-medium text-foreground capitalize">
                                    <span className="text-xs font-normal tabular-nums text-muted-foreground">{step.t + 1}</span>
                                    {step.action}
                                </h3>
                                {"verdict" in step.result && <VerdictBadge verdict={step.result.verdict} />}
                                {target &&
                                    <p title={target} className="min-w-0 flex-1 truncate rounded-md bg-muted px-2 py-0.5 font-mono text-xs text-foreground/80">{target}</p>}
                                <span className={`${target ? "" : "ml-auto "}flex shrink-0 items-center gap-2 text-xs tabular-nums`}>
                                    <span className="whitespace-nowrap">{step.seconds.toFixed(2)}s</span>
                                </span>
                            </div>
                            {step.result.pages?.length ? (
                                <ul className="flex flex-wrap gap-1.5">
                                    {step.result.pages.map((p) => (
                                        <li key={p} className="rounded-md bg-muted px-2 py-0.5 text-xs text-foreground/80">{pretty(p)}</li>
                                    ))}
                                </ul>
                            ) : null}
                            {empty && <p className="text-xs">{step.result.status!.replaceAll("_", " ")}</p>}
                            {step.result.hits?.length ? (
                                <details className="group">
                                    <summary className="flex cursor-pointer list-none items-center gap-1.5 text-xs font-medium [&::-webkit-details-marker]:hidden">
                                        <svg viewBox="0 0 20 20" className="size-3.5 shrink-0 transition group-open:rotate-90" fill="currentColor" aria-hidden>
                                            <path d="M7 5l6 5-6 5V5z" />
                                        </svg>
                                        {step.result.hits.length} sentences retrieved
                                        {(() => {
                                            const n = step.result.hits.filter((h) => cited.has(`${h.page}:${h.line}`)).length;
                                            return n > 0 && <span className="font-normal text-primary">· {n} cited</span>;
                                        })()}
                                    </summary>
                                    <ul className="scrollbar-nice mt-2 max-h-72 space-y-2 overflow-auto">
                                        {step.result.hits.map((hit) => {
                                            const isCited = cited.has(`${hit.page}:${hit.line}`);
                                            return (
                                                <li key={`${hit.page}:${hit.line}`} className={`border-l-2 pl-3 ${isCited ? "border-primary" : "border-border"}`}>
                                                    <p className="text-xs">
                                                        <span className="font-medium text-foreground">{pretty(hit.page)}</span> · sentence {hit.line}
                                                        {isCited && <span className="ml-1.5 text-primary">cited</span>}
                                                    </p>
                                                    <p className="mt-0.5">{hit.text}</p>
                                                </li>
                                            );
                                        })}
                                    </ul>
                                </details>
                            ) : null}
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
                );
            })}
            {busy && (
                <li className="relative pl-10" style={{ animation: "step-in 300ms ease-out" }}>
                    <span className="absolute left-0 top-1 z-10 grid size-7 place-items-center rounded-full border border-dashed bg-card">
                        <span className="size-2 animate-pulse rounded-full bg-primary" />
                    </span>
                    <div className="rounded-lg border border-dashed px-3 py-2 text-sm text-muted-foreground">
                        <span className="animate-pulse">Agent is running…</span>
                    </div>
                </li>
            )}
        </ol>
    )
}
