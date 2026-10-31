import { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { ClaimForm, ModeTabs } from "./components/ClaimForm";
import { ThemeToggle } from "./components/ThemeToggle";
import { TraceTable } from "./components/TraceTable";
import { Compare } from "./components/Compare";
import { VerificationResult } from "./components/VerificationResult";

export type Health = { live: boolean; policies: string[] };
export type Trace = { name: string; rows: number };
export type Claim = {
    id: string | number;
    claim: string;
    gold: string | null;
    verdict: string | null;
    confidence: number | null;
    correct: boolean;
    n_steps: number;
    n_ops?: number;
};
export type Mode = "browse" | "replay" | "live" | "compare";
export type Hit = { page: string; line: number; text: string };
export type Step = {
    t: number;
    action: string;
    args: { query?: string };
    result: { hits?: Hit[]; verdict?: string | null; subclaims?: string[]; reasoning?: string | null };
    seconds: number;
};
export type Row = {
    claim: string;
    gold?: string | null;
    run?: { policy?: string; verifier?: string };
    cost?: { n_ops?: number; tokens_in?: number; tokens_out?: number; seconds?: number };
    steps: Step[];
    final: { verdict: string | null; evidence: string[]; confidence: number | null };
};

export const getJson = async <T,>(path: string): Promise<T> => {
    const res = await fetch(path);
    if (!res.ok) throw new Error((await res.json()).detail ?? "Request failed.");
    return res.json();
};

export const pretty = (s?: string | null) => s?.replaceAll("_", " ").toLowerCase() ?? "no valid verdict";

const tones = {
    good: { box: "bg-success/10 text-success ring-success/25", dot: "bg-success", mark: "bg-success", panel: "bg-success/7 text-success ring-success/30", glow: "shadow-[0_0_24px_-4px_color-mix(in_srgb,var(--success)_40%,transparent)]", icon: "✓" },
    bad: { box: "bg-danger/10 text-danger ring-danger/25", dot: "bg-danger", mark: "bg-danger", panel: "bg-danger/7 text-danger ring-danger/30", glow: "shadow-[0_0_24px_-4px_color-mix(in_srgb,var(--danger)_40%,transparent)]", icon: "✗" },
    unsure: { box: "bg-warning/10 text-warning ring-warning/25", dot: "bg-warning", mark: "bg-transparent ring-1 ring-inset ring-warning", panel: "bg-warning/7 text-warning ring-warning/30", glow: "shadow-[0_0_24px_-4px_color-mix(in_srgb,var(--warning)_40%,transparent)]", icon: "?" },
    none: { box: "bg-muted/50 text-muted-foreground ring-border/50", dot: "bg-muted-foreground/50", mark: "bg-transparent ring-1 ring-inset ring-muted-foreground/60", panel: "bg-card text-muted-foreground ring-border", glow: "", icon: "-" },
};
export const toneOf = (v?: string | null) =>
    !v ? tones.none : /support/i.test(v) ? tones.good : /refut/i.test(v) ? tones.bad : tones.unsure;

function App() {
    const [health, setHealth] = useState<Health | null>(null);
    const [traces, setTraces] = useState<Trace[]>([]);
    const [claims, setClaims] = useState<Claim[]>([]);
    const [loadingClaims, setLoadingClaims] = useState(false);
    const [browseDetail, setBrowseDetail] = useState(false);
    const [trace, setTrace] = useState("");
    const [traceB, setTraceB] = useState("");
    const [claimsB, setClaimsB] = useState<Claim[]>([]);
    const [loadingB, setLoadingB] = useState(false);
    const [claimId, setClaimId] = useState("");
    const [mode, setMode] = useState<Mode>("browse");
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
            .then(([h, t]) => { setHealth(h); setTraces(t); setTrace(t[0]?.name ?? ""); setTraceB(t[1]?.name ?? ""); })
            .catch((e: Error) => setError(e.message));
        return () => source.current?.close();
    }, []);

    useEffect(() => {
        if (!trace) return;
        let stale = false;
        setClaims([]);
        setClaimId("");
        setLoadingClaims(true);
        setBrowseDetail(false);
        setClaim("");
        setRow(null);
        setSteps([]);
        getJson<{ items: Claim[] }>(`/api/traces/${encodeURIComponent(trace)}?limit=10000`)
            .then(({ items }) => {
                if (stale) return;
                setClaims(items);
                setClaimId(String(items[0]?.id ?? ""));
            })
            .catch((e: Error) => !stale && setError(e.message))
            .finally(() => !stale && setLoadingClaims(false));
        return () => { stale = true; };
    }, [trace]);

    // Compare's second run: only its summary list is needed, nothing else resets.
    useEffect(() => {
        if (!traceB) return;
        let stale = false;
        setClaimsB([]);
        setLoadingB(true);
        getJson<{ items: Claim[] }>(`/api/traces/${encodeURIComponent(traceB)}?limit=10000`)
            .then(({ items }) => !stale && setClaimsB(items))
            .catch((e: Error) => !stale && setError(e.message))
            .finally(() => !stale && setLoadingB(false));
        return () => { stale = true; };
    }, [traceB]);

    useEffect(() => { setBrowseDetail(false); }, [mode]);

    async function start(savedClaimId = claimId) {
        if (busy) return;
        source.current?.close();
        source.current = null;
        setSteps([]);
        setRow(null);
        setError("");
        setBusy(true);
        setStatus("Connecting…");

        const isLive = mode === "live";
        setClaim(isLive ? input.trim() : claims.find((c) => String(c.id) === savedClaimId)?.claim ?? "");

        const savedUrl = `/api/traces/${encodeURIComponent(trace)}/${encodeURIComponent(savedClaimId)}`;
        if (mode === "browse") {
            setClaimId(savedClaimId);
            setBrowseDetail(true);
            window.scrollTo({ top: 0 });
            setStatus("Loading saved trace…");
            try {
                const saved = await getJson<Row>(savedUrl);
                setClaim(saved.claim);
                setRow(saved);
                setSteps(saved.steps);
                setStatus("Saved trace loaded");
            } catch (e) {
                setError(e instanceof Error ? e.message : "Could not load the saved trace.");
                setStatus("Stopped");
            } finally {
                setBusy(false);
            }
            return;
        }

        const url = isLive
            ? `/api/live?${new URLSearchParams({ claim: input.trim(), policy })}`
            : `${savedUrl}/replay?delay=0.3`;
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
            final: (d) => {
                setRow(d.row); setSteps(d.row.steps); setStatus("Complete"); finish();
                // --save-live may have written a new trace file: offer it in Browse without a reload
                if (isLive) getJson<Trace[]>("/api/traces").then(setTraces).catch(() => undefined);
            },
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

    const backendState = !health && !error ? "connecting" : !health ? "offline" : health.live ? "live" : "saved";

    const statusUI = {
        connecting: { label: "Connecting…", dot: "bg-muted-foreground animate-pulse" },
        offline: { label: "Backend offline", dot: "bg-danger" },
        live: { label: "Live", dot: "bg-success" },
        saved: { label: "Saved traces only", dot: "bg-warning" },
    }[backendState];

    const showForm = mode === "replay" || mode === "live";
    const canRun = !busy && !!health && (mode === "live" ? !!input.trim() : !!claimId);

    return (
        <div className="min-h-screen">
            <main className="p-4 lg:p-8">
                <header className="mb-4 flex flex-wrap items-center justify-between gap-x-4 gap-y-3 lg:mb-8">
                    <h1 className="flex items-center gap-3 font-serif text-4xl font-semibold tracking-tight">
                        Fact verifier
                    </h1>
                    <div className="order-last w-full lg:order-0 lg:w-96">
                        <ModeTabs mode={mode} setMode={setMode} busy={busy} liveAvailable={!!health?.live} />
                    </div>
                    <div className="flex items-center gap-3">
                        <span className="inline-flex items-center gap-2 rounded-full border px-3 py-1 text-xs text-muted-foreground">
                            <span className={`size-1.5 rounded-full ${statusUI.dot}`} />
                            {statusUI.label}
                        </span>
                        <ThemeToggle />
                    </div>
                </header>

                {error && (
                    <p role="alert" className="mb-6 px-5 py-4 rounded-xl bg-danger/10 text-sm text-danger border">{error}</p>
                )}

                <div className={`grid items-start gap-8 ${showForm ? "lg:grid-cols-[340px_1fr]" : ""}`}>
                    {showForm && <ClaimForm
                        health={health}
                        traces={traces}
                        claims={claims}
                        trace={trace}
                        setTrace={setTrace}
                        claimId={claimId}
                        setClaimId={setClaimId}
                        mode={mode}
                        input={input}
                        setInput={setInput}
                        policy={policy}
                        setPolicy={setPolicy}
                        busy={busy}
                        canRun={canRun}
                        onStart={start}
                    />}
                    {mode === "compare" ? (
                        <Compare key={`${trace}|${traceB}`} traces={traces} nameA={trace} nameB={traceB}
                            setNameA={setTrace} setNameB={setTraceB}
                            claimsA={claims} claimsB={claimsB} loading={loadingClaims || loadingB} />
                    ) : mode === "browse" ? (
                        <div className="min-w-0 space-y-5">
                            <div hidden={browseDetail}>
                                <TraceTable key={trace} traces={traces} trace={trace} setTrace={setTrace} claims={claims} loading={loadingClaims} busy={busy}
                                    selectedId={claimId} onOpen={(selected) => { void start(String(selected.id)); }} />
                            </div>
                            {browseDetail && <>
                                <button type="button" disabled={busy} onClick={() => setBrowseDetail(false)}
                                    className="cursor-pointer text-sm font-medium disabled:opacity-40">
                                    ← Back
                                </button>
                                <VerificationResult claim={claim} row={row} steps={steps} status={status} busy={busy} />
                            </>}
                        </div>
                    ) : <VerificationResult claim={claim} row={row} steps={steps} status={status} busy={busy} />}
                </div>
            </main>
        </div>
    );
}

createRoot(document.getElementById("root")!).render(<App />);
