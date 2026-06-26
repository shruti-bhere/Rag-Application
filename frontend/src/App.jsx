import { useState, useRef, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { uploadPdf, askStream } from "./api";
import "./App.css";

const API_BASE = "/api";

function App() {
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [uploadResult, setUploadResult] = useState(null);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [streamContent, setStreamContent] = useState("");
  const [isDragging, setIsDragging] = useState(false);
  const messagesEndRef = useRef(null);
  const fileInputRef = useRef(null);

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, []);

  const startUpload = async (selected) => {
    if (!selected) return;
    setUploading(true);
    setUploadResult(null);
    try {
      const res = await uploadPdf(API_BASE, selected);
      setUploadResult(res);
      setFile(null);
      if (fileInputRef.current) fileInputRef.current.value = "";
    } catch (err) {
      setUploadResult({ error: err.message || "Upload failed." });
    } finally {
      setUploading(false);
    }
  };

  const onFileChange = async (e) => {
    const f = e.target.files?.[0];
    if (f?.name?.toLowerCase().endsWith(".pdf")) {
      setFile(f);
      setUploadResult(null);
      await startUpload(f);
    } else if (f) {
      setFile(null);
      setUploadResult({ error: "Please select a PDF file." });
    }
  };

  const handleDrop = async (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    const f = e.dataTransfer?.files?.[0];
    if (f?.name?.toLowerCase().endsWith(".pdf")) {
      setFile(f);
      setUploadResult(null);
      await startUpload(f);
    } else if (f) {
      setFile(null);
      setUploadResult({ error: "Please select a PDF file." });
    }
  };

  const handleUpload = async () => {
    if (!file) return;
    await startUpload(file);
  };

  const handleAsk = async () => {
    const q = input.trim();
    if (!q || streaming) return;
    setInput("");
    setMessages((prev) => [...prev, { role: "user", content: q }]);
    setStreaming(true);
    setStreamContent("");

    try {
      let full = "";
      for await (const chunk of askStream(API_BASE, q)) {
        if (chunk.type === "token" && chunk.content) {
          full += chunk.content;
          setStreamContent(full);
          scrollToBottom();
        }
        if (chunk.type === "error") {
          setMessages((prev) => [...prev, { role: "assistant", content: `Error: ${chunk.content}` }]);
          setStreamContent("");
          return;
        }
      }
      if (full) {
        setMessages((prev) => [...prev, { role: "assistant", content: full }]);
      }
    } catch (err) {
      setMessages((prev) => [...prev, { role: "assistant", content: `Error: ${err.message}` }]);
    } finally {
      setStreamContent("");
      setStreaming(false);
      scrollToBottom();
    }
  };

  return (
    <div className="app">
      <motion.header
        className="header"
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4 }}
      >
        <div className="logo">
          <span className="logo-icon">◇</span>
          <h1>RAG PDF Q&A</h1>
        </div>
        <p className="tagline">Upload PDFs · Semantic search · Ask in natural language</p>
      </motion.header>

      <main className="main">
        <motion.section
          className="upload-section"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4, delay: 0.1 }}
        >
          <div
            className={`upload-zone${isDragging ? " dragging" : ""}`}
            onClick={() => fileInputRef.current?.click()}
            onKeyDown={(e) => e.key === "Enter" && fileInputRef.current?.click()}
            onDragEnter={(e) => {
              e.preventDefault();
              e.stopPropagation();
              setIsDragging(true);
            }}
            onDragOver={(e) => {
              e.preventDefault();
              e.stopPropagation();
            }}
            onDragLeave={(e) => {
              e.preventDefault();
              e.stopPropagation();
              setIsDragging(false);
            }}
            onDrop={handleDrop}
            role="button"
            tabIndex={0}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept=".pdf"
              onChange={onFileChange}
              className="upload-input"
            />
            <span className="upload-icon">📄</span>
            <span className="upload-text">
              {file ? file.name : "Drop PDF or click to upload"}
            </span>
            <motion.button
              className="btn btn-init"
              disabled={uploading}
              onClick={(e) => {
                e.stopPropagation();
                if (!file) {
                  fileInputRef.current?.click();
                } else {
                  handleUpload();
                }
              }}
              whileHover={!uploading ? { scale: 1.02 } : {}}
              whileTap={!uploading ? { scale: 0.98 } : {}}
            >
              <span className="btn-init-icon" aria-hidden="true" />
              <span className="btn-init-label">
                {uploading ? "UPLOADING…" : "UPLOAD PDF"}
              </span>
            </motion.button>
          </div>
          <AnimatePresence>
            {uploadResult && (
              <motion.div
                className={`upload-result ${uploadResult.error ? "error" : "success"}`}
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: "auto" }}
                exit={{ opacity: 0, height: 0 }}
                transition={{ duration: 0.25 }}
              >
                {uploadResult.error
                  ? uploadResult.error
                  : `Indexed ${uploadResult.chunks_indexed} chunks from ${uploadResult.filename}`}
              </motion.div>
            )}
          </AnimatePresence>
        </motion.section>

        <motion.section
          className="chat-section"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4, delay: 0.2 }}
        >
          <div className="chat-messages">
            <AnimatePresence initial={false}>
              {messages.map((msg, i) => (
                <motion.div
                  key={i}
                  className={`message message-${msg.role}`}
                  initial={{ opacity: 0, x: msg.role === "user" ? 20 : -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.25 }}
                >
                  <span className="message-role">{msg.role === "user" ? "You" : "Assistant"}</span>
                  <div className="message-content">{msg.content}</div>
                </motion.div>
              ))}
            </AnimatePresence>
            {streaming && (
              <motion.div
                className="message message-assistant streaming"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
              >
                <span className="message-role">Assistant</span>
                <div className="message-content">
                  {streamContent ? (
                    <>
                      {streamContent}
                      <span className="cursor" />
                    </>
                  ) : (
                    <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
                      <div className="radar-spinner" />
                      <span className="thinking">Thinking…</span>
                    </div>
                  )}
                </div>
              </motion.div>
            )}
            <div ref={messagesEndRef} />
          </div>

          <motion.div
            className="chat-input-wrap"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.3 }}
          >
            <input
              type="text"
              className="chat-input"
              placeholder="Ask anything about your documents…"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && handleAsk()}
              disabled={streaming}
            />
            <motion.button
              className="btn btn-send"
              onClick={handleAsk}
              disabled={!input.trim() || streaming}
              whileHover={input.trim() && !streaming ? { scale: 1.03 } : {}}
              whileTap={input.trim() && !streaming ? { scale: 0.97 } : {}}
            >
              {streaming ? "…" : "Send"}
            </motion.button>
          </motion.div>
        </motion.section>
      </main>
    </div>
  );
}

export default App;
