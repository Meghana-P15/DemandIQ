"use client";
import { ReactNode } from "react";

type Point = { label: string; value: number };

export function MiniLineChart({ data }: { data: Point[] }) {
  if (!data.length) return <div className="emptyChart">No data</div>;
  const w=700,h=230,p=24; const min=Math.min(...data.map(d=>d.value)); const max=Math.max(...data.map(d=>d.value)); const range=max-min||1;
  const pts=data.map((d,i)=>`${p+(i/(Math.max(1,data.length-1)))*(w-2*p)},${h-p-((d.value-min)/range)*(h-2*p)}`).join(" ");
  return <div className="chartWrap"><svg viewBox={`0 0 ${w} ${h}`} role="img" aria-label="Demand trend chart">
    <defs><linearGradient id="area" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stopColor="var(--teal)" stopOpacity=".24"/><stop offset="1" stopColor="var(--teal)" stopOpacity="0"/></linearGradient></defs>
    {[0,1,2,3].map(i=><line key={i} x1={p} x2={w-p} y1={p+i*(h-2*p)/3} y2={p+i*(h-2*p)/3} className="gridLine"/>)}
    <polygon points={`${p},${h-p} ${pts} ${w-p},${h-p}`} fill="url(#area)"/>
    <polyline points={pts} fill="none" stroke="var(--teal)" strokeWidth="4" strokeLinecap="round" strokeLinejoin="round"/>
  </svg><div className="chartLabels"><span>{data[0]?.label}</span><span>{data[data.length-1]?.label}</span></div></div>;
}

export function HorizontalBars({ data, maxItems=7 }: { data: Point[]; maxItems?: number }) {
  const rows=data.slice(0,maxItems); const max=Math.max(...rows.map(d=>d.value),1);
  return <div className="bars">{rows.map((d,i)=><div className="barRow" key={`${d.label}-${i}`}><div className="barMeta"><span>{d.label}</span><strong>{d.value.toLocaleString(undefined,{maximumFractionDigits:2})}</strong></div><div className="barTrack"><span style={{width:`${Math.max(2,d.value/max*100)}%`}}/></div></div>)}</div>;
}

export function Scatter({ data }: { data: {x:number;y:number;cluster:number;label:string}[] }) {
  if (!data.length) return <div className="emptyChart">No data</div>; const w=700,h=360,p=30; const xs=data.map(d=>d.x), ys=data.map(d=>d.y); const xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=Math.min(...ys),ymax=Math.max(...ys);
  const sx=(x:number)=>p+(x-xmin)/(xmax-xmin||1)*(w-2*p); const sy=(y:number)=>h-p-(y-ymin)/(ymax-ymin||1)*(h-2*p);
  return <div className="chartWrap"><svg viewBox={`0 0 ${w} ${h}`} role="img" aria-label="Cluster PCA scatter plot">{data.map((d,i)=><circle key={i} cx={sx(d.x)} cy={sy(d.y)} r="6" className={`clusterPoint cluster${Math.abs(d.cluster)%4}`}><title>{d.label} · cluster {d.cluster}</title></circle>)}</svg></div>;
}
