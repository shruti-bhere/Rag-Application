const getBase = (base) => (base || "").replace(/\/$/, "");

export async function uploadPdf(apiBase, file) {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${getBase(apiBase)}/upload`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || res.statusText || "Upload failed");
  }
  return res.json();
}

export async function* askStream(apiBase, question) {
  const res = await fetch(`${getBase(apiBase)}/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || res.statusText || "Ask failed");
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";
    for (const line of lines) {
      if (line.startsWith("data: ")) {
        const data = line.slice(6);
        if (data === "[DONE]") return;
        try {
          const parsed = JSON.parse(data);
          yield parsed;
        } catch (_) {}
      }
    }
  }
  if (buffer.startsWith("data: ")) {
    const data = buffer.slice(6);
    if (data !== "[DONE]") {
      try {
        yield JSON.parse(data);
      } catch (_) {}
    }
  }
}
