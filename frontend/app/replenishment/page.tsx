"use client";

import { useEffect, useState } from "react";
import { Shell } from "@/components/Shell";
import { PriorityBadge } from "@/components/PriorityBadge";
import { HorizontalBars } from "@/components/Charts";
import { apiGet } from "@/lib/api";
import { number, percent } from "@/lib/format";

export default function Replenishment() {
  const [priority, setPriority] = useState("");
  const [data, setData] = useState<any>(null);
  const [error, setError] = useState("");

  const load = () =>
    apiGet<any>(
      `/replenishment?limit=100${
        priority ? `&priority=${priority}` : ""
      }`
    )
      .then(setData)
      .catch((e) => setError(e.message));

  useEffect(() => {
    void load();
  }, [priority]);

  const bars = (data?.items || [])
    .slice(0, 8)
    .map((r: any) => ({
      label: `S${r.store_nbr} · ${r.family}`,
      value: r.replenishment_priority_score,
    }));

  return (
    <Shell
      title="Replenishment intelligence"
      eyebrow="DECISION SUPPORT"
    >
      <div className="notice">
        Priority score = 35% demand + 30% high-demand risk + 15% volatility +
        10% anomaly signal + 10% promotion. Thresholds preserve the original
        Week 12 rules.
      </div>

      <div
        className="contentGrid"
        style={{ marginTop: 18 }}
      >
        <section className="panel">
          <div className="panelHeader">
            <div>
              <h2>Highest priority actions</h2>
              <p>Sorted by genuine replenishment priority score</p>
            </div>
          </div>

          <HorizontalBars data={bars} />
        </section>

        <section className="panel">
          <div className="panelHeader">
            <div>
              <h2>Decision thresholds</h2>
              <p>Transparent, deterministic action layer</p>
            </div>
          </div>

          <div className="bars">
            <div className="clusterCard">
              <PriorityBadge value="CRITICAL" />
              <p>≥ 0.75 · Review / replenish first</p>
            </div>

            <div className="clusterCard">
              <PriorityBadge value="HIGH" />
              <p>≥ 0.55 · Prioritize replenishment</p>
            </div>

            <div className="clusterCard">
              <PriorityBadge value="MEDIUM" />
              <p>≥ 0.35 · Monitor closely</p>
            </div>

            <div className="clusterCard">
              <PriorityBadge value="LOW" />
              <p>&lt; 0.35 · Normal monitoring</p>
            </div>
          </div>
        </section>
      </div>

      <section className="panel">
        <div className="sectionToolbar">
          <p>
            {data
              ? `${data.total.toLocaleString()} matching recommendations`
              : "Loading…"}
          </p>

          <div className="filters">
            <select
              value={priority}
              onChange={(e) => setPriority(e.target.value)}
            >
              <option value="">All priorities</option>
              <option>CRITICAL</option>
              <option>HIGH</option>
              <option>MEDIUM</option>
              <option>LOW</option>
            </select>
          </div>
        </div>

        {error && (
          <div className="notice error">
            {error}
          </div>
        )}

        <div className="dataTableWrap">
          <table className="dataTable">
            <thead>
              <tr>
                <th>Store</th>
                <th>Family</th>
                <th>Forecast</th>
                <th>Risk</th>
                <th>Volatility</th>
                <th>Anomaly</th>
                <th>Score</th>
                <th>Priority</th>
                <th>Action</th>
              </tr>
            </thead>

            <tbody>
              {data?.items.map((r: any, i: number) => (
                <tr key={i}>
                  <td>#{r.store_nbr}</td>
                  <td>{r.family}</td>
                  <td>{number(r.predicted_demand, 0)}</td>
                  <td>{percent(r.high_demand_probability)}</td>
                  <td>{number(r.demand_volatility, 0)}</td>
                  <td>{r.anomaly_signal ? "Flagged" : "—"}</td>
                  <td>{number(r.replenishment_priority_score, 3)}</td>
                  <td>
                    <PriorityBadge value={r.priority} />
                  </td>
                  <td>{r.recommended_action}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </Shell>
  );
}