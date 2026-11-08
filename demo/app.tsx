import { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { ClaimForm, ModeTabs } from "./components/ClaimForm";
import { ThemeToggle } from "./components/ThemeToggle";
import { VerificationResult } from "./components/VerificationResult";
import type { Pending } from "./components/Architecture";

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
export type Mode = "replay" | "live";

/**
 * Replay offers a handful of saved runs of the live controller (2B imitation, follow tools,
 * dev 900), chosen to show the different paths it takes. All are correct verdicts.
 */
export const FEATURED_TRACE =
    "fever_dev_prompted-sft-direct-from-base-qwen3p5-2b-5895-samples_run1_ctl-local-sft-controller-from-base-qwen3p5-2b-flash-follow-18291-samples_run1_follow_bm25-p5-k100_fever-dev-900-ids-ac9cfead_n900.jsonl";
export const FEATURED_IDS = [
    "8740",   // supports in one search
    "137968", // refutes in one search
    "224186", // supports after following a link (two hops)
    "8834",   // refutes after a follow and a second search
    "227372", // refutes after two follows
    "134748", // not enough info: searches twice, then abstains
];
export type Hit = { page: string; line: number; text: string };
export type Step = {
    t: number;
    action: string;
    args: { query?: string; title?: string; target?: string; link?: string; page?: string | null };
    result: { hits?: Hit[]; verdict?: string | null; subclaims?: string[]; reasoning?: string | null; status?: string; pages?: string[] };
    seconds: number;
};
export type Row = {
    claim: string;
    gold?: string | null;
    run?: { policy?: string; verifier?: string; controller?: string; tools?: string; max_steps?: number };
    cost?: { n_ops?: number; tokens_in?: number; tokens_out?: number; seconds?: number };
    steps: Step[];
    final: { verdict: string | null; evidence: string[]; confidence: number | null };
};

export const getJson = async <T,>(path: string): Promise<T> => {
    const res = await fetch(path);
    if (!res.ok) throw new Error((await res.json()).detail ?? "Request failed.");
    return res.json();
};

/** "prompted-bm25" -> "Prompted" etc. Unknown policies fall back to their raw name. */
export const policyLabel = (p: string) =>
    p.startsWith("fixed") ? "Fixed: search, then verify"
    : p === "controller" ? "Controller: 2B imitation model"
    : p === "prompted" ? "Prompted: Qwen flash chooses steps"
    : p.startsWith("prompted") ? "Prompted: agent chooses steps"
    : `${p.replaceAll(/[-_]/g, " ").replace(/^./, (c) => c.toUpperCase())}: agent chooses steps`;

/** Short model name for display: drop the org prefix and any path. */
export const shortModel = (s?: string) => s?.split("/").filter(Boolean).pop();

/** Readable controller name: the local adapters by what they are, API models by short name. */
export const controllerName = (c?: string) => {
    if (!c) return undefined;
    if (/sft-controller/.test(c)) return "2B imitation";
    if (/grpo/.test(c)) return "2B GRPO";
    return shortModel(c.replace(/^local:/, "").split("+")[0]);
};

/** The controller (and tool preset) behind a run, or null for the fixed loop. */
export const controllerLabel = (run?: Row["run"]) =>
    run?.controller ? `${controllerName(run.controller)}${run.tools && run.tools !== "full" ? ` · ${run.tools} tools` : ""}` : null;

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
    const [trace, setTrace] = useState("");
    const [claimId, setClaimId] = useState("");
    const [mode, setMode] = useState<Mode>("live");
    const [input, setInput] = useState("");
    const [policy, setPolicy] = useState("fixed");

    const [claim, setClaim] = useState("");
    const [steps, setSteps] = useState<Step[]>([]);
    const [row, setRow] = useState<Row | null>(null);
    const [busy, setBusy] = useState(false);
    const [status, setStatus] = useState("Ready");
    const [error, setError] = useState("");
    const [pending, setPending] = useState<Pending>(null);
    const [runInfo, setRunInfo] = useState<Row["run"] | null>(null);
    const [runKey, setRunKey] = useState(0);
    const source = useRef<EventSource | null>(null);

    useEffect(() => {
        Promise.all([getJson<Health>("/api/health"), getJson<Trace[]>("/api/traces")])
            .then(([h, t]) => {
                setHealth(h);
                if (h.policies.includes("controller")) setPolicy("controller");
                if (!h.live) setMode("replay");
                setTraces(t);
                setTrace(t.some((x) => x.name === FEATURED_TRACE) ? FEATURED_TRACE : t[0]?.name ?? "");
            })
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
                // the featured claims, in their listed order; any other trace offers all of its claims
                const byId = new Map(items.map((c) => [String(c.id), c]));
                const featured = FEATURED_IDS.flatMap((id) => byId.get(id) ?? []);
                const shown = trace === FEATURED_TRACE && featured.length ? featured : items;
                setClaims(shown);
                setClaimId(String(shown[0]?.id ?? ""));
            })
            .catch((e: Error) => !stale && setError(e.message));
        return () => { stale = true; };
    }, [trace]);

    async function start(savedClaimId = claimId) {
        if (busy) return;
        source.current?.close();
        source.current = null;
        setSteps([]);
        setRow(null);
        setPending(null);
        setRunInfo(null);
        setRunKey((k) => k + 1);
        setError("");
        setBusy(true);
        setStatus("Connecting…");

        const isLive = mode === "live";
        setClaim(isLive ? input.trim() : claims.find((c) => String(c.id) === savedClaimId)?.claim ?? "");
        const savedUrl = `/api/traces/${encodeURIComponent(trace)}/${encodeURIComponent(savedClaimId)}`;

        const url = isLive
            ? `/api/live?${new URLSearchParams({ claim: input.trim(), policy })}`
            : `${savedUrl}/replay?delay=0.05`; // the diagram sets the pace
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
            run: (d) => { setClaim(d.claim); setRunInfo(d.run ?? null); setStatus(d.replay ? "Replaying saved run…" : "Checking claim…"); },
            action: (d) => { setStatus(`${d.action}…`); setPending({ t: d.t, action: d.action, args: d.args ?? {} }); },
            step: (d) => { setPending(null); setSteps((prev) => [...prev, d.step]); },
            final: (d) => {
                setRow(d.row); setSteps(d.row.steps); setStatus("Complete"); finish();
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

    const result = { claim, row, steps, status, busy, pending, runKey, run: runInfo ?? row?.run ?? null, animate: true };

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

                <div className="grid items-start gap-8 lg:grid-cols-[340px_1fr]">
                    <ClaimForm
                        health={health}
                        claims={claims}
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
                    />
                    <VerificationResult {...result} />
                </div>
            </main>
        </div>
    );
}

createRoot(document.getElementById("root")!).render(<App />);
