// webui/tests/overview.spec.ts
import { test, expect } from "@playwright/test";

test("overview shows demo project and lifecycle", async ({ page }) => {
  await page.goto("http://127.0.0.1:8787/");
  await expect(page.getByText("norne-brugge")).toBeVisible();
  await expect(page.getByText("История итераций")).toBeVisible();
});
