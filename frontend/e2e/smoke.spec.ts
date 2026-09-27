import { expect, test } from "@playwright/test";

test("landing page communicates the product", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByText("Forecast demand.")).toBeVisible();
  await expect(page.getByRole("link", { name: /open live demo/i })).toBeVisible();
});

test("dashboard loads authentic KPI surface", async ({ page }) => {
  await page.goto("/dashboard");
  await expect(page.getByText("Demand command center")).toBeVisible();
  await expect(page.getByText("Temporal R²")).toBeVisible();
});

test("forecast form is usable on mobile", async ({ page }) => {
  await page.goto("/forecast");
  await expect(page.getByRole("button", { name: /generate forecast/i })).toBeVisible();
});
