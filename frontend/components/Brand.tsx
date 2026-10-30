import Link from "next/link";

export function Brand({ compact = false }: { compact?: boolean }) {
  return (
    <Link href="/" className="brand" aria-label="DemandIQ home">
      <span className="brandMark" aria-hidden="true"><span></span><span></span><span></span></span>
      {!compact && <span>Demand<span>IQ</span></span>}
    </Link>
  );
}
