/**
 * @vitest-environment jsdom
 */
import React from "react";
import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ProviderSubagentJournalUnavailable } from "./provider-subagent-journal-unavailable";

const { theme } = vi.hoisted(() => ({
  theme: { colors: { foregroundMuted: "#aaaaaa" } },
}));

vi.mock("react-native-unistyles", () => ({
  StyleSheet: {
    create: (factory: unknown) => (typeof factory === "function" ? factory(theme) : factory),
  },
}));

const UNAVAILABLE_COPY =
  "The provider does not share this task's journal. Only its status is shown.";

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string) => (key === "subagents.providerLogUnavailable" ? UNAVAILABLE_COPY : key),
  }),
}));

afterEach(() => {
  cleanup();
});

describe("ProviderSubagentJournalUnavailable", () => {
  it("says the journal is unavailable instead of showing an empty transcript", () => {
    // A child of a provider-owned tree has no addressable turn. Rendering nothing
    // would read as a broken chat, so the panel has to say what is missing.
    render(<ProviderSubagentJournalUnavailable />);

    expect(screen.getByTestId("provider-subagent-panel-unsupported")).toBeTruthy();
    expect(screen.getByText(UNAVAILABLE_COPY)).toBeTruthy();
  });
});
