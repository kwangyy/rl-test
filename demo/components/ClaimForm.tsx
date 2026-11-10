import type { Claim, Mode } from "../app";

export const fieldClass =
    "w-full rounded-lg resize-none leading-relaxed border bg-background px-3 py-2 text-sm shadow-sm transition placeholder:text-muted-foreground hover:border-foreground/30 focus:outline-none focus:ring-2 focus:ring-primary/40 disabled:cursor-not-allowed disabled:opacity-50";

type Props = {
    claims: Claim[];
    claimId: string; setClaimId: (v: string) => void;
    mode: Mode;
    input: string; setInput: (v: string) => void;
    busy: boolean; canRun: boolean; onStart: () => void;
};

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

export function ClaimForm({ claims, claimId, setClaimId, mode, input, setInput, busy, canRun, onStart }: Props) {
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
                    <p className="text-xs leading-relaxed text-muted-foreground">
                        The 2B imitation controller checks your claim live against FEVER's Wikipedia pages.
                    </p>
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
