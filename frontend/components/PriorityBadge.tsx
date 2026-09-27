export function PriorityBadge({ value }: { value: string }) {
  const key=value.toLowerCase(); return <span className={`badge priority ${key}`}>{value}</span>;
}
