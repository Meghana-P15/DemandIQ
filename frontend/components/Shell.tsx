"use client";
import { ReactNode } from "react";
import { usePathname } from "next/navigation";
import Link from "next/link";
import { Brand } from "./Brand";

const nav = [
  ["/dashboard", "Overview", "⌂"],
  ["/forecast", "Demand Forecast", "↗"],
  ["/batch", "Batch Analysis", "⇧"],
  ["/replenishment", "Replenishment", "◆"],
  ["/anomalies", "Anomalies", "!"],
  ["/segments", "Segments", "◫"],
  ["/performance", "Performance", "⌁"],
  ["/explainability", "Explainability", "✦"],
  ["/architecture", "Architecture", "⌘"],
];

export function Shell({ children, title, eyebrow }: { children: ReactNode; title: string; eyebrow?: string }) {
  const path = usePathname();
  return <div className="appShell">
    <aside className="sidebar">
      <div className="sideBrand"><Brand /></div>
      <div className="sideLabel">INTELLIGENCE</div>
      <nav>{nav.map(([href,label,icon]) => <Link key={href} href={href} className={`navItem ${path===href?"active":""}`}><span className="navIcon">{icon}</span><span>{label}</span></Link>)}</nav>
      <div className="sidebarFooter">
        <div className="demoStatus"><span className="statusDot"></span><div><strong>Demo mode</strong><small>Authentic project data</small></div></div>
        <Link href="/" className="backLink">← Product site</Link>
      </div>
    </aside>
    <main className="mainPanel">
      <header className="topbar">
        <div><span className="eyebrow">{eyebrow || "DEMAND INTELLIGENCE"}</span><h1>{title}</h1></div>
        <div className="topbarActions"><span className="livePill"><span className="statusDot"></span>Live model</span><a href="http://localhost:8000/docs" target="_blank" rel="noreferrer" className="iconButton">API ↗</a></div>
      </header>
      <div className="pageContent">{children}</div>
    </main>
  </div>;
}
