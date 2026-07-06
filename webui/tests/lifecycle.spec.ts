// webui/tests/lifecycle.spec.ts
import { test, expect } from "@playwright/test";

test("full lifecycle on demo: run → results", async ({ page }) => {
  await page.goto("http://127.0.0.1:8787/");
  await page.getByText("norne-brugge").click();
  await page.getByRole("button", { name: "Запуски" }).click();
  await page.getByRole("button", { name: "Запустить итерацию" }).click();
  await expect(page.locator("table tbody tr").first()).toBeVisible();
});
