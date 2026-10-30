import { ReactNode } from "react";

export function MetricCard({ label, value, detail, icon, tone = "teal" }: { label: string; value: ReactNode; detail?: ReactNode; icon?: string; tone?: "teal"|"indigo"|"amber"|"red" }) {
  return <article className="metricCard">
    <div className={`metricIcon ${tone}`}>{icon || "↗"}</div>
    <div className="metricLabel">{label}</div>
    <div className="metricValue">{value}</div>
    {detail && <div className="metricDetail">{detail}</div>}
  </article>;
}
