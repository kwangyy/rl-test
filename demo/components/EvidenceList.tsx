import { pretty, type Hit } from "../app"

export function EvidenceList({ hits, cited }: { hits: Hit[]; cited: Set<string> }) {
    return (
        hits.length > 0 ? (
            <div className="space-y-3">
                <h3 className="text-base font-semibold">
                    Evidence <span className="font-normal text-muted-foreground">({hits.length} retrieved, {cited.size} cited)</span>
                </h3>
                {hits.map((hit) => {
                    const id = `${hit.page}:${hit.line}`;
                    const isCited = cited.has(id);
                    return (
                        <details
                            key={id}
                            open={isCited}
                            className={`group rounded-lg bg-card p-4 shadow-sm ring-1 ${isCited ? "ring-2 ring-primary" : "ring-border"}`}
                        >
                            <summary className="flex cursor-pointer list-none items-center gap-3 text-sm font-medium [&::-webkit-details-marker]:hidden">
                                <svg viewBox="0 0 20 20" className="h-4 w-4 shrink-0 text-muted-foreground transition group-open:rotate-90" fill="currentColor" aria-hidden>
                                    <path d="M7 5l6 5-6 5V5z" />
                                </svg>
                                <span className="min-w-0 flex-1 wrap-break-word">
                                    {pretty(hit.page)} <span className="font-normal text-muted-foreground">sentence {hit.line}</span>
                                </span>
                                {isCited && <span className="rounded-full bg-primary/10 px-2.5 py-0.5 text-xs font-medium text-primary">Cited</span>}
                            </summary>
                            <p className="mt-3 border-l-2 border-border pl-4 text-sm leading-6 text-muted-foreground">{hit.text}</p>
                        </details>
                    );
                })}
            </div>
        ) : null
    );
}
