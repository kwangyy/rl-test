import { useEffect, useState } from "react";

const isDark = () => document.documentElement.classList.contains("dark");

/**
 * Light/dark switch. The initial class is set by the inline script in index.html
 * so the first paint already matches; this only keeps it in sync afterwards.
 */
export function ThemeToggle() {
    const [dark, setDark] = useState(isDark);

    useEffect(() => {
        document.documentElement.classList.toggle("dark", dark);
        localStorage.setItem("theme", dark ? "dark" : "light");
    }, [dark]);

    const label = dark ? "Switch to light theme" : "Switch to dark theme";

    return (
        <button
            type="button"
            onClick={() => setDark((d) => !d)}
            aria-pressed={dark}
            aria-label={label}
            title={label}
            className="grid size-8 shrink-0 cursor-pointer place-items-center rounded-full border bg-background text-muted-foreground shadow-sm transition hover:border-foreground/30 hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40"
        >
            <svg viewBox="0 0 24 24" className="size-4" fill="none" stroke="currentColor" strokeWidth="1.75"
                strokeLinecap="round" strokeLinejoin="round" aria-hidden>
                {dark ? (
                    <>
                        <circle cx="12" cy="12" r="4" />
                        <path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" />
                    </>
                ) : (
                    <path d="M20 14.5A8.5 8.5 0 1 1 9.5 4a7 7 0 0 0 10.5 10.5z" />
                )}
            </svg>
        </button>
    );
}
