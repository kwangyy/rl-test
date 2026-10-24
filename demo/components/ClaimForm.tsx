import { useState, useRef, useEffect } from "react";
import { type Claim, type Health, type Mode, type Trace } from "../app"

export const fieldClass =
    "w-full rounded-lg resize-none leading-relaxed border bg-background px-3.5 py-2.5 text-sm shadow-sm transition placeholder:text-muted-foreground hover:border-foreground/30 focus:outline-none focus:ring-2 focus:ring-primary/40 disabled:cursor-not-allowed disabled:opacity-50";

type Props = {
    health: Health | null; traces: Trace[]; claims: Claim[];
    trace: string; setTrace: (v: string) => void;
    claimId: string; setClaimId: (v: string) => void;
    mode: Mode; setMode: (v: Mode) => void;
    input: string; setInput: (v: string) => void;
    policy: string; setPolicy: (v: string) => void;
    busy: boolean; canRun: boolean; onStart: () => void;
};

type Option = { value: string; label: string };

export function Select({
    value,
    onChange,
    options,
    placeholder = "Select…",
    disabled,
}: {
    value: string;
    onChange: (v: string) => void;
    options: Option[];
    placeholder?: string;
    disabled?: boolean;
}) {
    const [open, setOpen] = useState(false);
    const ref = useRef<HTMLDivElement>(null);

    // close when clicking outside
    useEffect(() => {
        const close = (e: MouseEvent) => {
            if (!ref.current?.contains(e.target as Node)) setOpen(false);
        };
        document.addEventListener("mousedown", close);
        return () => document.removeEventListener("mousedown", close);
    }, []);

    const selected = options.find((o) => o.value === value);

    return (
        <div ref={ref} className="relative min-w-0">
            <button
                type="button"
                disabled={disabled}
                onClick={() => setOpen((o) => !o)}
                className="flex w-full cursor-pointer items-center justify-between gap-1.5 rounded-lg border bg-background px-3.5 py-2.5 text-left text-sm shadow-sm transition hover:border-foreground/30 focus:outline-none focus:ring-2 focus:ring-primary/40 disabled:cursor-not-allowed disabled:opacity-50"
            >
                <span className={`truncate ${selected ? "" : "text-muted-foreground"}`}>
                    {selected?.label ?? placeholder}
                </span>
                <svg
                    className={`h-4 w-4 shrink-0 text-muted-foreground transition-transform ${open ? "rotate-180" : ""}`}
                    viewBox="0 0 20 20"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                >
                    <path d="M5 8l5 5 5-5" />
                </svg>
            </button>

            {open && (
                <ul className="absolute z-20 scrollbar-nice mt-1.5 max-h-60 w-full overflow-auto rounded-lg border bg-background p-1 shadow-lg">
                    {options.length === 0 && (
                        <li className="px-3 py-2 text-sm text-muted-foreground">Nothing to select</li>
                    )}
                    {options.map((o) => (
                        <li
                            key={o.value}
                            onClick={() => { onChange(o.value); setOpen(false); }}
                            className={`flex cursor-pointer items-center justify-between gap-2 rounded-md px-3 py-2 text-sm transition hover:bg-muted ${o.value === value ? "bg-muted font-medium" : ""}`}
                        >
                            <span className="truncate">{o.label}</span>
                            {o.value === value && <span className="text-primary">✓</span>}
                        </li>
                    ))}
                </ul>
            )}
        </div>
    );
}

export function ClaimForm({ health, traces, claims, trace, setTrace, claimId, setClaimId, mode, setMode, input, setInput, policy, setPolicy, busy, canRun, onStart }: Props) {
    return (
        <form
            className="min-w-0 space-y-4 rounded-xl p-6 shadow-sm border lg:sticky lg:top-8"
            onSubmit={(e) => { e.preventDefault(); onStart(); }}
        >
            <h2 className="text-sm font-medium">Mode</h2>
            <div className="-mt-2 grid grid-cols-3 p-1 gap-1 bg-secondary rounded-lg">
                {(["browse", "replay", "live"] as const).map((m) => (
                    <button
                        key={m}
                        type="button"
                        disabled={busy || (m === "live" && !health?.live)}
                        aria-pressed={mode === m}
                        onClick={() => setMode(m)}
                        className={`rounded-md px-3 py-1.5 text-sm font-medium capitalize transition cursor-pointer disabled:cursor-not-allowed ${mode === m ? "bg-background shadow-sm" : "text-foreground/60 hover:text-foreground"}`}
                    >
                        {m}
                    </button>
                ))}
            </div>

            {mode !== "live" ? (
                <>
                    <div className="flex min-w-0 flex-col gap-2 text-sm font-medium">
                        Experiment
                        <Select
                            value={trace}
                            onChange={setTrace}
                            disabled={busy || !traces.length}
                            placeholder="No saved traces available"
                            options={traces.map((t) => ({
                                value: t.name,
                                label: t.name.replace(/\.jsonl$/, ""),
                            }))}
                        />
                    </div>
                    {mode === "replay" && <div className="flex min-w-0 flex-col gap-2 text-sm font-medium">
                        Claim
                        <Select
                            value={claimId}
                            onChange={setClaimId}
                            disabled={busy || !claims.length}
                            placeholder="Select an experiment"
                            options={claims.map((c) => ({ value: String(c.id), label: c.claim }))}
                        />
                    </div>}
                </>
            ) : (
                <>
                    <label className="flex flex-col gap-2 text-sm font-medium">
                        Claim
                        <textarea
                            className={fieldClass}
                            rows={4}
                            value={input}
                            disabled={busy}
                            onChange={(e) => setInput(e.target.value)}
                            placeholder="Enter a claim"
                        />
                    </label>
                    <div className="flex flex-col gap-2 text-sm font-medium">
                        Policy
                        <Select
                            value={policy}
                            onChange={setPolicy}
                            disabled={busy}
                            options={(health?.policies ?? []).map((p) => ({
                                value: p,
                                label: p === "fixed" ? "Fixed: search, then verify" : "Prompted: agent chooses steps",
                            }))}
                        />
                    </div>
                </>
            )}

            {mode !== "browse" && <button
                className="w-full px-5 py-3 cursor-pointer rounded-lg bg-primary text-sm font-medium text-primary-foreground shadow-sm transition hover:bg-primary/90 focus-visible:outline-none focus-visible:ring-4 focus-visible:ring-primary/25 disabled:cursor-not-allowed disabled:opacity-40"
                disabled={!canRun}
            >
                {busy ? "Running…" : mode === "replay" ? "Replay" : "Verify"}
            </button>}
        </form>
    );
}
