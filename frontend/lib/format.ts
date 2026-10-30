export function compactNumber(value: number): string {
  return new Intl.NumberFormat("en", { notation: "compact", maximumFractionDigits: 1 }).format(value);
}
export function number(value: number, digits = 2): string {
  return new Intl.NumberFormat("en", { maximumFractionDigits: digits }).format(value);
}
export function percent(value: number, digits = 1): string {
  return `${(value * 100).toFixed(digits)}%`;
}
