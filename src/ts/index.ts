// index.ts
import { serve } from "bun";
import fs from "fs/promises";
import path from "path";

// Helper: determine MIME type based on file extension
function getMimeType(ext: string): string {
    switch (ext.toLowerCase()) {
        case ".jpg":
        case ".jpeg":
            return "image/jpeg";
        case ".png":
            return "image/png";
        case ".gif":
            return "image/gif";
        case ".bmp":
            return "image/bmp";
        case ".webp":
            return "image/webp";
        default:
            return "application/octet-stream";
    }
}

// GET /images?sourceDir=...
async function handleGetImages(request: Request): Promise<Response> {
    try {
        const url = new URL(request.url);
        const sourceDir = url.searchParams.get("sourceDir");
        const destDir = url.searchParams.get("destDir");
        if (!sourceDir || !destDir) {
            return new Response(JSON.stringify({ error: "sourceDir and destDir query parameter is required" }), { status: 400 });
        }
        const files = await fs.readdir(sourceDir);
        const imageFiles = files.filter((file) => {
            const ext = path.extname(file).toLowerCase();
            return [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"].includes(ext);
        });
        console.log(destDir);
        const dirs = await fs.readdir(destDir);
        console.log(dirs);
        const destDirs = dirs.filter((folder) => {
            return !folder.includes(".");
        });
        console.log(destDirs);
        return new Response(JSON.stringify({ images: imageFiles, destDirs: destDirs }), {
            headers: { "Content-Type": "application/json" }
        });
    } catch (err: any) {
        return new Response(JSON.stringify({ error: err.message }), { status: 500 });
    }
}

// GET /image?sourceDir=...&filename=...
async function handleGetImage(request: Request): Promise<Response> {
    try {
        const url = new URL(request.url);
        const sourceDir = url.searchParams.get("sourceDir");
        const filename = url.searchParams.get("filename");
        if (!sourceDir || !filename) {
            return new Response("Missing sourceDir or filename", { status: 400 });
        }
        const filePath = path.join(sourceDir, filename);
        const fileBuffer = await fs.readFile(filePath);
        const mimeType = getMimeType(path.extname(filename));
        return new Response(fileBuffer, { headers: { "Content-Type": mimeType } });
    } catch (err: any) {
        return new Response(JSON.stringify({ error: err.message }), { status: 500 });
    }
}

// POST /move-image
// Request JSON: { sourceDir, destinationDir, filename, targetFolder }
async function handleMoveImage(request: Request): Promise<Response> {
    try {
        const { sourceDir, destinationDir, filename, targetFolder } = await request.json();
        if (!sourceDir || !destinationDir || !filename || !targetFolder) {
            return new Response(JSON.stringify({ error: "Missing required fields" }), { status: 400 });
        }
        const sourcePath = path.join(sourceDir, filename);
        const targetDir = path.join(destinationDir, targetFolder);
        await fs.mkdir(targetDir, { recursive: true });
        const targetPath = path.join(targetDir, filename);
        await fs.rename(sourcePath, targetPath);
        return new Response(JSON.stringify({ success: true }), {
            headers: { "Content-Type": "application/json" }
        });
    } catch (err: any) {
        return new Response(JSON.stringify({ error: err.message }), { status: 500 });
    }
}

serve({
    async fetch(request) {
        const url = new URL(request.url);
        // Serve the static HTML file at the root.
        if (url.pathname === "/") {
            return new Response(Bun.file("./public/index.html"), {
                headers: { "Content-Type": "text/html" }
            });
        }
        // Bun supports TypeScript on the fly—serve main.ts as JavaScript.
        if (url.pathname === "/main.js") {
            return new Response(Bun.file("./public/main.js"), {
                headers: { "Content-Type": "application/javascript" }
            });
        }
        if (url.pathname === "/images" && request.method === "GET") {
            return handleGetImages(request);
        }
        if (url.pathname === "/image" && request.method === "GET") {
            return handleGetImage(request);
        }
        if (url.pathname === "/move-image" && request.method === "POST") {
            return handleMoveImage(request);
        }
        return new Response("Not Found", { status: 404 });
    },
    port: 3000
});
