import { describe, expect, it } from "vitest";
import { compactNumber, number, percent } from "../lib/format";

describe("format helpers", () => {
  it("formats percentages from probabilities", () => {
    expect(percent(0.6599999, 1)).toBe("66.0%");
  });

  it("formats decimals with requested precision", () => {
    expect(number(0.963665, 3)).toBe("0.964");
  });

  it("uses compact notation for large counts", () => {
    expect(compactNumber(98702)).toMatch(/98.7K|98.7k/);
  });
});
