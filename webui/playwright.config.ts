// webui/playwright.config.ts
import { defineConfig } from "@playwright/test";
export default defineConfig({ testDir: "./tests", timeout: 30000 });
