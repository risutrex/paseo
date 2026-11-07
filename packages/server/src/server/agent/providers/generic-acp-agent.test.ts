import { describe, expect, test, vi } from "vitest";

import { createTestLogger } from "../../../test-utils/test-logger.js";

const mockState = vi.hoisted(() => ({
  superConstructorOptions: [] as unknown[],
}));

vi.mock("./acp-agent.js", () => ({
  DEFAULT_ACP_CAPABILITIES: {
    supportsStreaming: true,
    supportsSessionPersistence: true,
    supportsDynamicModes: true,
    supportsMcpServers: true,
    supportsReasoningStream: true,
    supportsToolInvocations: true,
    supportsRewindConversation: false,
    supportsRewindFiles: false,
    supportsRewindBoth: false,
  },
  ACPAgentClient: class ACPAgentClient {
    readonly provider: string;

    constructor(options: unknown) {
      this.provider = "acp";
      mockState.superConstructorOptions.push(options);
    }
  },
  isMiniMaxCodeProvider: (providerId: string | undefined, command: readonly string[] | undefined) =>
    providerId === "minimax-code" || (command ?? []).some((part) => /mcode|minimax/i.test(part)),
  MCODE_CLIENT_CAPABILITY_META: {
    "minimax-code/extensions": { version: 1, notifications: true },
  },
  mergeACPClientCapabilityMeta: (
    base: Record<string, unknown> | undefined,
    override: Record<string, unknown> | undefined,
  ) => (base ? { ...base, ...override } : override),
}));

import { GenericACPAgentClient } from "./generic-acp-agent.js";

describe("GenericACPAgentClient", () => {
  test("passes the custom command only as defaultCommand", () => {
    const _client = new GenericACPAgentClient({
      logger: createTestLogger(),
      command: ["hermes", "acp"],
      env: {
        HERMES_LOG: "info",
      },
    });
    void _client;

    expect(mockState.superConstructorOptions).toEqual([
      {
        provider: "acp",
        logger: expect.any(Object),
        runtimeSettings: {
          env: {
            HERMES_LOG: "info",
          },
        },
        defaultCommand: ["hermes", "acp"],
        capabilities: {
          supportsStreaming: true,
          supportsSessionPersistence: true,
          supportsDynamicModes: true,
          supportsMcpServers: true,
          supportsReasoningStream: true,
          supportsToolInvocations: true,
          supportsRewindConversation: false,
          supportsRewindFiles: false,
          supportsRewindBoth: false,
        },
        waitForInitialCommands: true,
      },
    ]);
  });
  test("keeps the vendor extension request next to the user's own metadata", () => {
    // The vendor request asks MiniMax Code for its extension notifications.
    // A user's own metadata must travel beside it, not replace it: replacing
    // it would turn the extensions off without a word.
    mockState.superConstructorOptions.length = 0;
    const _client = new GenericACPAgentClient({
      logger: createTestLogger(),
      command: ["minimax-code", "acp"],
      providerId: "minimax-code",
      clientCapabilityMeta: { "user/key": { enabled: true } },
    });
    void _client;

    const options = mockState.superConstructorOptions[0] as {
      clientCapabilityMeta?: Record<string, unknown>;
    };
    expect(options.clientCapabilityMeta).toEqual({
      "minimax-code/extensions": { version: 1, notifications: true },
      "user/key": { enabled: true },
    });
  });
});
