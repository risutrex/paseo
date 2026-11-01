// @vitest-environment jsdom

import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { ProviderQueueTrack } from "./provider-queue";

const LABEL = "In the provider's queue - not editable here";

const identityStatus = (status: string): string => status;
const statusOrQueued = (status: string): string => (status ? status : "queued");

afterEach(cleanup);

describe("ProviderQueueTrack", () => {
  it("shows a provider queue position apart from the composer queue", () => {
    // The composer already has a queue of its own. A provider position must be
    // recognisable at a glance, or the user reads two sources of truth as one.
    render(
      <ProviderQueueTrack
        items={[{ itemId: "item-1", text: "second thought", status: "queued", createdAt: null }]}
        label={LABEL}
        statusLabel={identityStatus}
      />,
    );

    expect(screen.getByText("second thought")).toBeTruthy();
    expect(screen.getByText(LABEL)).toBeTruthy();
    expect(screen.getByText("1 · queued")).toBeTruthy();
    // The provider exposes no edit or promote over ACP from here, so the row must
    // not offer either: a button that cannot work is worse than no button.
    expect(screen.queryByLabelText("Edit queued message")).toBeNull();
    expect(screen.queryByLabelText("Send queued message now")).toBeNull();
  });

  it("renders nothing for an empty or unsupported queue", () => {
    const { container } = render(
      <ProviderQueueTrack items={[]} label={LABEL} statusLabel={identityStatus} />,
    );

    expect(container.childElementCount).toBe(0);
  });

  it("keeps a row whose text the provider did not report", () => {
    // An unknown position must not break the composer: the id alone is still
    // worth showing, because it is the only handle the provider gives.
    render(
      <ProviderQueueTrack
        items={[{ itemId: "item-2", text: "", status: "", createdAt: null }]}
        label={LABEL}
        statusLabel={statusOrQueued}
      />,
    );

    expect(screen.getByTestId("composer-provider-queue-item")).toBeTruthy();
    expect(screen.getByText("1 · queued")).toBeTruthy();
  });
});
