import { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { ClaimForm } from "./components/ClaimForm";
import { VerificationResult } from "./components/VerificationResult";

export type Health = { live: boolean; policies: string[] };
export type Trace = { name: string; rows: number };
export type Claim = { id: string | number; claim: string };
export type Hit = { page: string; line: number; text: string };
export type Step = {
    t: number;
    action: string;
    args: { query?: string };
    result: { hits?: Hit[]; verdict?: string | null; subclaims?: string[] };
    seconds: number;
};
export type Row = {
    steps: Step[];
    final: { verdict: string | null; evidence: string[]; confidence: number | null };
};

export const getJson = async <T,>(path: string): Promise<T> => {
    const res = await fetch(path);
    if (!res.ok) throw new Error((await res.json()).detail ?? "Request failed.");
    return res.json();
};

export const pretty = (s?: string | null) => s?.replaceAll("_", " ").toLowerCase() ?? "no valid verdict";

export const tones = {
    good: { box: "bg-emerald-50 text-emerald-900 ring-emerald-200", dot: "bg-emerald-500" },
    bad: { box: "bg-rose-50 text-rose-900 ring-rose-200", dot: "bg-rose-500" },
    unsure: { box: "bg-amber-50 text-amber-900 ring-amber-200", dot: "bg-amber-500" },
    none: { box: "bg-slate-100 text-slate-700 ring-slate-200", dot: "bg-slate-400" },
};
export const toneOf = (v?: string | null) =>
    !v ? tones.none : /support/i.test(v) ? tones.good : /refut/i.test(v) ? tones.bad : tones.unsure;

function App() {
    const [health, setHealth] = useState<Health | null>(null);
    const [traces, setTraces] = useState<Trace[]>([]);
    const [claims, setClaims] = useState<Claim[]>([]);
    const [trace, setTrace] = useState("");
    const [claimId, setClaimId] = useState("");
    const [mode, setMode] = useState<"replay" | "live">("replay");
    const [input, setInput] = useState("");
    const [policy, setPolicy] = useState("fixed");

    const [claim, setClaim] = useState("");
    const [steps, setSteps] = useState<Step[]>([]);
    const [row, setRow] = useState<Row | null>(null);
    const [busy, setBusy] = useState(false);
    const [status, setStatus] = useState("Ready");
    const [error, setError] = useState("");
    const source = useRef<EventSource | null>(null);

    useEffect(() => {
        Promise.all([getJson<Health>("/api/health"), getJson<Trace[]>("/api/traces")])
            .then(([h, t]) => { setHealth(h); setTraces(t); setTrace(t[0]?.name ?? ""); })
            .catch((e: Error) => setError(e.message));
        return () => source.current?.close();
    }, []);

    useEffect(() => {
        if (!trace) return;
        let stale = false;
        setClaims([]);
        setClaimId("");
        getJson<{ items: Claim[] }>(`/api/traces/${encodeURIComponent(trace)}?limit=10000`)
            .then(({ items }) => {
                if (stale) return;
                setClaims(items);
                setClaimId(String(items[0]?.id ?? ""));
            })
            .catch((e: Error) => !stale && setError(e.message));
        return () => { stale = true; };
    }, [trace]);

    function start() {
        source.current?.close();
        setSteps([]);
        setRow(null);
        setError("");
        setBusy(true);
        setStatus("Connecting…");

        const isLive = mode === "live";
        setClaim(isLive ? input.trim() : claims.find((c) => String(c.id) === claimId)?.claim ?? "");

        const url = isLive
            ? `/api/live?${new URLSearchParams({ claim: input.trim(), policy })}`
            : `/api/traces/${encodeURIComponent(trace)}/${encodeURIComponent(claimId)}/replay?delay=0.3`;
        const events = new EventSource(url);
        source.current = events;

        const finish = (message?: string) => {
            if (message) { setError(message); setStatus("Stopped"); }
            setBusy(false);
            events.close();
        };

        const handlers: Record<string, (data: any) => void> = {
            queued: (d) => setStatus(d.message),
            status: (d) => setStatus(d.message),
            run: (d) => { setClaim(d.claim); setStatus(d.replay ? "Replaying saved run…" : "Checking claim…"); },
            action: (d) => setStatus(`${d.action}…`),
            step: (d) => setSteps((prev) => [...prev, d.step]),
            final: (d) => { setRow(d.row); setSteps(d.row.steps); setStatus("Complete"); finish(); },
            error: (d) => finish(d.message),
        };

        for (const [name, handle] of Object.entries(handlers)) {
            events.addEventListener(name, (event) => {
                if (source.current !== events) return;
                // A plain Event (no data) means the connection itself failed.
                if (!(event instanceof MessageEvent)) return finish("The connection closed before completion.");
                try { handle(JSON.parse(event.data)); }
                catch { finish("The backend sent an unreadable event."); }
            });
        }
    }

    const canRun = !busy && !!health && (mode === "replay" ? !!claimId : !!input.trim());

    return (
        <div className="min-h-screen">
            <main className="p-4 lg:p-8">
                <header className="mb-8 flex items-center justify-between gap-4">
                    <div className="flex items-center gap-3">
                        <span className="grid h-10 w-10 place-items-center rounded-xl">
                            <svg viewBox="0 0 24 24" className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
                                <path d="m5 12.5 4.5 4.5L19 7.5" />
                            </svg>
                        </span>
                        <h1 className="text-xl font-semibold tracking-tight">Fact verifier</h1>
                    </div>
                    <span className="inline-flex items-center gap-2 rounded-full px-3.5 py-1.5 text-xs border">
                        <span className={`h-2 w-2 rounded-full ${health ? "bg-emerald-500" : "bg-rose-500"}`} />
                        {!health ? "Backend offline" : health.live ? "Live" : "Replay only"}
                    </span>
                </header>

                {error && (
                    <p role="alert" className="mb-6 px-5 py-4 rounded-xl bg-destructive/9 text-sm text-destructive/80 border">{error}</p>
                )}

                <div className="grid items-start gap-8 lg:grid-cols-[340px_1fr]">
                    <ClaimForm
                        health={health}
                        traces={traces}
                        claims={claims}
                        trace={trace}
                        setTrace={setTrace}
                        claimId={claimId}
                        setClaimId={setClaimId}
                        mode={mode}
                        setMode={setMode}
                        input={input}
                        setInput={setInput}
                        policy={policy}
                        setPolicy={setPolicy}
                        busy={busy}
                        canRun={canRun}
                        onStart={start}
                    />
                    <VerificationResult claim={claim} row={row} steps={steps} status={status} busy={busy} />
                </div>
            </main>
        </div>
    );
}

createRoot(document.getElementById("root")!).render(<App />);
