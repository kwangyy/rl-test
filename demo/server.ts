import { rm } from "node:fs/promises";
import index from "./index.html";
import tailwind from "bun-plugin-tailwind";

if (Bun.argv.includes("--build")) {
    const outdir = `${import.meta.dir}/dist`;
    const result = await Bun.build({
        entrypoints: [`${import.meta.dir}/index.html`],
        outdir,
        target: "browser",
        minify: true,
        naming: { chunk: "app.[ext]" },
        define: { "process.env.NODE_ENV": '"production"' },
        plugins: [tailwind],
    });
    if (!result.success) {
        for (const log of result.logs) console.error(log);
        process.exit(1);
    }
    const outputs = new Set(result.outputs.map((output) => output.path));
    for await (const path of new Bun.Glob("**/*").scan({ cwd: outdir, absolute: true, onlyFiles: true })) {
        if (!outputs.has(path)) await rm(path);
    }
    console.log("Built demo/dist");
} else {
    const backend = new URL(process.env.API_URL ?? "http://127.0.0.1:8000");
    const server = Bun.serve({
        hostname: "127.0.0.1",
        port: Number(process.env.PORT ?? 3000),
        idleTimeout: 0,
        routes: {
            "/": index,
            "/api/*": async (request) => {
                const url = new URL(request.url);
                try {
                    return await fetch(new URL(url.pathname + url.search, backend), {
                        signal: request.signal,
                    });
                } catch {
                    return Response.json({ detail: "Cannot reach the verification backend." }, { status: 502 });
                }
            },
        },
        fetch() {
            return new Response("Not found", { status: 404 });
        },
    });
    console.log(`Fact verifier: ${server.url}`);
}
