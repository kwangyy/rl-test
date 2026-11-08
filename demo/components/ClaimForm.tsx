import { useState, useRef, useEffect } from "react";
import { policyLabel, type Claim, type Health, type Mode } from "../app"

export const fieldClass =
    "w-full rounded-lg resize-none leading-relaxed border bg-background px-3 py-2 text-sm shadow-sm transition placeholder:text-muted-foreground hover:border-foreground/30 focus:outline-none focus:ring-2 focus:ring-primary/40 disabled:cursor-not-allowed disabled:opacity-50";

type Props = {
    health: Health | null; claims: Claim[];
    claimId: string; setClaimId: (v: string) => void;
    mode: Mode;
    input: string; setInput: (v: string) => void;
    policy: string; setPolicy: (v: string) => void;
    busy: boolean; canRun: boolean; onStart: () => void;
};

type Option = { value: string; label: string };

function MiddleTruncate({ text, tail = 20 }: { text: string; tail?: number }) {
    if (text.length <= tail + 8) return <span className="block truncate">{text}</span>;
    return (
        <span className="flex">
            <span className="truncate">{text.slice(0, -tail)}</span>
            <span className="shrink-0 whitespace-nowrap">{text.slice(-tail)}</span>
        </span>
    );
}

export function Select({
    value,
    onChange,
    options,
    placeholder = "Select…",
    disabled,
    middle,
}: {
    value: string;
    onChange: (v: string) => void;
    options: Option[];
    placeholder?: string;
    disabled?: boolean;
    middle?: boolean; // cut long names in the middle so the distinguishing suffix stays visible
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
                className="flex w-full cursor-pointer items-center justify-between gap-1.5 rounded-lg border bg-background px-3 py-2 text-left text-sm shadow-sm transition hover:border-foreground/30 focus:outline-none focus:ring-2 focus:ring-primary/40 disabled:cursor-not-allowed disabled:opacity-50"
            >
                <span className={`min-w-0 flex-1 ${selected ? "" : "text-muted-foreground"}`} title={selected?.label}>
                    {middle && selected ? <MiddleTruncate text={selected.label} /> : <span className="block truncate">{selected?.label ?? placeholder}</span>}
                </span>
                <svg
                    className={`size-3.5 shrink-0 text-muted-foreground transition-transform ${open ? "rotate-180" : ""}`}
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
                <ul className="absolute z-20 scrollbar-nice mt-0.5 max-h-60 w-full overflow-auto rounded-lg border bg-background p-1 shadow-lg">
                    {options.length === 0 && (
                        <li className="px-3 py-2 text-sm text-muted-foreground">Nothing to select</li>
                    )}
                    {options.map((o) => (
                        <li
                            key={o.value}
                            onClick={() => { onChange(o.value); setOpen(false); }}
                            className={`flex cursor-pointer items-center justify-between gap-2 rounded-md px-3 py-2 text-sm transition hover:bg-muted ${o.value === value ? "bg-muted font-medium" : ""}`}
                        >
                            <span className="min-w-0 flex-1" title={o.label}>{middle ? <MiddleTruncate text={o.label} /> : <span className="block truncate">{o.label}</span>}</span>
                            {o.value === value && <span className="text-primary">✓</span>}
                        </li>
                    ))}
                </ul>
            )}
        </div>
    );
}

const modes: Mode[] = ["live", "replay"];

export function ModeTabs({ mode, setMode, busy, liveAvailable }: {
    mode: Mode; setMode: (v: Mode) => void; busy: boolean; liveAvailable: boolean;
}) {
    return (
        <nav aria-label="Mode" className="relative grid grid-cols-2 rounded-lg bg-secondary p-1">
            <span
                aria-hidden
                className="absolute inset-y-1 left-1 w-[calc((100%-0.5rem)/2)] rounded-md bg-background shadow-sm transition-transform duration-300 ease-out"
                style={{ transform: `translateX(${modes.indexOf(mode) * 100}%)` }}
            />
            {modes.map((m) => (
                <button
                    key={m}
                    type="button"
                    disabled={busy || (m === "live" && !liveAvailable)}
                    aria-pressed={mode === m}
                    onClick={() => setMode(m)}
                    className={`relative z-10 rounded-md px-3 py-1.5 text-sm font-medium capitalize transition-colors cursor-pointer disabled:cursor-not-allowed disabled:opacity-50 ${mode === m ? "" : "text-foreground/60 hover:text-foreground"}`}
                >
                    {m}
                </button>
            ))}
        </nav>
    );
}

export function ClaimForm({ health, claims, claimId, setClaimId, mode, input, setInput, policy, setPolicy, busy, canRun, onStart }: Props) {
    return (
        <form
            className="min-w-0 space-y-4 card-raised rounded-xl p-5 pt-4 border bg-card lg:sticky lg:top-8"
            onSubmit={(e) => { e.preventDefault(); onStart(); }}
        >
            {mode !== "live" ? (
                <>
                    <p className="text-xs leading-relaxed text-muted-foreground">
                        Saved runs of the 2B imitation controller on FEVER dev claims, played back step by step.
                    </p>
                    <fieldset className="flex min-w-0 flex-col gap-2 text-sm font-medium" disabled={busy}>
                        <legend className="mb-2">Claim</legend>
                        {claims.length === 0 && <p className="text-sm font-normal text-muted-foreground">Loading saved runs…</p>}
                        {claims.map((c) => (
                            <label key={c.id} className={`flex cursor-pointer gap-2.5 rounded-lg border px-3 py-2 text-sm font-normal leading-snug transition has-disabled:cursor-not-allowed ${String(c.id) === claimId ? "border-primary/60 bg-primary/5" : "hover:border-foreground/30"}`}>
                                <input type="radio" name="claim" className="mt-0.5 accent-(--primary)" value={String(c.id)}
                                    checked={String(c.id) === claimId} onChange={() => setClaimId(String(c.id))} />
                                <span className="min-w-0 flex-1">{c.claim}</span>
                            </label>
                        ))}
                    </fieldset>
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
                                label: policyLabel(p),
                            }))}
                        />
                    </div>
                </>
            )}

            <button
                className="flex w-full items-center justify-center gap-2 px-5 py-3 cursor-pointer rounded-lg bg-primary text-sm font-medium text-primary-foreground shadow-sm transition hover:bg-primary/90 focus-visible:outline-none focus-visible:ring-4 focus-visible:ring-primary/25 disabled:cursor-not-allowed disabled:opacity-40"
                disabled={!canRun}
            >
                {busy
                    ? <><span className="size-3.5 animate-spin rounded-full border-2 border-primary-foreground/30 border-t-primary-foreground" />Running…</>
                    : mode === "replay" ? "Replay" : "Verify"}
            </button>
        </form>
    );
}
