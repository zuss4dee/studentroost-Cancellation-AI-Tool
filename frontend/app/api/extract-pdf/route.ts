import { extractPDFContent } from "../../../lib/pdfReader";

export const runtime = "nodejs";
export const maxDuration = 300;

function isPdfBuffer(buf: Buffer): boolean {
  if (buf.length < 5) return false;
  return buf.subarray(0, 5).toString("ascii") === "%PDF-";
}

export async function POST(req: Request): Promise<Response> {
  try {
    const ct = req.headers.get("content-type") ?? "";
    if (!ct.includes("multipart/form-data")) {
      return Response.json({ error: "Expected multipart form data" }, { status: 400 });
    }

    const form = await req.formData();
    const file = form.get("file");

    if (!file || !(file instanceof Blob)) {
      return Response.json({ error: "Missing file field" }, { status: 400 });
    }

    const ab = await file.arrayBuffer();
    const buffer = Buffer.from(ab);

    if (!isPdfBuffer(buffer)) {
      return Response.json({ error: "Not a PDF file" }, { status: 400 });
    }

    const result = await extractPDFContent(buffer);
    return Response.json(result);
  } catch (e) {
    const message = e instanceof Error ? e.message : "Extraction failed";
    return Response.json({ error: message }, { status: 500 });
  }
}
