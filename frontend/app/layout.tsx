import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "DemandIQ | AI-Powered Demand Intelligence",
  description: "Forecast retail demand, detect anomalies and prioritize replenishment with machine learning.",
  openGraph: { title: "DemandIQ", description: "Forecast Demand. Detect Risk. Replenish Smarter.", type: "website" },
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body>{children}</body></html>;
}
