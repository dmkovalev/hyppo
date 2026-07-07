import React from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import "./styles.css";

class ErrorBoundary extends React.Component<
  { children: React.ReactNode }, { err: Error | null; info: string }> {
  constructor(props: { children: React.ReactNode }) {
    super(props);
    this.state = { err: null, info: "" };
  }
  static getDerivedStateFromError(err: Error) { return { err, info: "" }; }
  componentDidCatch(err: Error, info: { componentStack: string }) {
    this.setState({ err, info: info.componentStack });
  }
  render() {
    if (this.state.err) {
      return (
        <div style={{ padding: 40, fontFamily: "IBM Plex Mono, monospace", color: "#b0324f" }}>
          <h2>Ошибка рендера</h2>
          <p style={{ fontSize: 16, fontWeight: 600 }}>{String(this.state.err?.message || this.state.err)}</p>
          <pre style={{ whiteSpace: "pre-wrap", fontSize: 12, color: "#5f5c55" }}>{this.state.info}</pre>
        </div>
      );
    }
    return this.props.children;
  }
}

createRoot(document.getElementById("root")!).render(
  <React.StrictMode><ErrorBoundary><App /></ErrorBoundary></React.StrictMode>
);
