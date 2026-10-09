"use client";

import type { ReactNode } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { Brand } from "./Brand";

const navSections = [
  {
    label: "WORKSPACE",
    items: [
      ["/dashboard", "Overview", "▦"],
      ["/forecast", "Demand Forecast", "⌁"],
      ["/batch", "Batch Analysis", "⇧"],
    ],
  },
  {
    label: "INTELLIGENCE",
    items: [
      ["/replenishment", "Replenishment", "◇"],
      ["/anomalies", "Anomalies", "!"],
      ["/segments", "Segments", "◫"],
    ],
  },
  {
    label: "MODEL",
    items: [
      ["/performance", "Performance", "◉"],
      ["/explainability", "Explainability", "✦"],
      ["/architecture", "Architecture", "⌘"],
    ],
  },
];

export function Shell({
  children,
  title,
  eyebrow,
}: {
  children: ReactNode;
  title: string;
  eyebrow?: string;
}) {
  const pathname = usePathname();

  const apiRoot = (
    process.env.NEXT_PUBLIC_API_URL ||
    "http://localhost:8000/api/v1"
  ).replace(/\/api\/v1\/?$/, "");

  const docsUrl = `${apiRoot}/docs`;

  const isActive = (href: string) =>
    pathname === href ||
    (href !== "/dashboard" && pathname.startsWith(href));

  return (
    <div className="appShell">
      <aside className="sidebar">
        <div className="sideBrand">
          <Brand />

          <span className="productEdition">
            RETAIL INTELLIGENCE
          </span>
        </div>

        <div className="navGroups">
          {navSections.map((section) => (
            <div className="navGroup" key={section.label}>
              <div className="navGroupLabel">
                {section.label}
              </div>

              <nav>
                {section.items.map(([href, label, icon]) => (
                  <Link
                    key={href}
                    href={href}
                    className={`navItem ${
                      isActive(href) ? "active" : ""
                    }`}
                  >
                    <span className="navIcon">
                      {icon}
                    </span>

                    <span className="navText">
                      {label}
                    </span>

                    {isActive(href) && (
                      <span className="activeIndicator" />
                    )}
                  </Link>
                ))}
              </nav>
            </div>
          ))}
        </div>

        <div className="sidebarFooter">
          <div className="modelStatusCard">
            <div className="modelStatusTop">
              <span className="statusDot" />

              <span>MODEL ONLINE</span>
            </div>

            <strong>XGBoost Production</strong>

            <small>
              Demand forecasting & risk intelligence
            </small>
          </div>

          <Link href="/" className="backLink">
            ← Back to product site
          </Link>
        </div>
      </aside>

      <main className="mainPanel">
        <header className="topbar">
          <div className="topbarTitle">
            <span className="eyebrow">
              {eyebrow || "DEMAND INTELLIGENCE"}
            </span>

            <div className="topbarTitleRow">
              <h1>{title}</h1>

              <span className="topbarDivider" />

              <span className="workspaceLabel">
                Retail intelligence workspace
              </span>
            </div>
          </div>

          <div className="topbarActions">
            <span className="livePill">
              <span className="statusDot" />
              Live model
            </span>

            <a
              href={docsUrl}
              target="_blank"
              rel="noreferrer"
              className="apiButton"
            >
              API Docs
              <span>↗</span>
            </a>
          </div>
        </header>

        <div className="pageContent">
          {children}
        </div>
      </main>
    </div>
  );
}