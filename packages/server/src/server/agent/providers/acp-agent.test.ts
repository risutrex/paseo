import { type ChildProcess, type ChildProcessWithoutNullStreams } from "node:child_process";
import { EventEmitter } from "node:events";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { createRequire } from "node:module";
import { tmpdir } from "node:os";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import {
  AgentSideConnection,
  ClientSideConnection,
  PROTOCOL_VERSION,
  RequestError,
  ndJsonStream,
  type Agent,
  type Client as ACPClient,
  type CreateTerminalRequest,
  PermissionOption,
  PromptResponse,
  RequestPermissionRequest,
  SessionConfigOption,
  SessionUpdate,
} from "@agentclientprotocol/sdk";

import {
  ACPAgentClient,
  ACPAgentSession,
  MCODE_CLIENT_CAPABILITY_META,
  type SpawnedACPProcess,
  type SessionStateResponse,
  buildACPClientCapabilities,
  createLoggedNdJsonStream,
  deriveModelDefinitionsFromACP,
  deriveModesFromACP,
  isMiniMaxCodeProvider,
  mapACPUsage,
  mergeACPClientCapabilityMeta,
  resolveACPModeSelection,
  resolveACPModelSelection,
  summarizeACPRequestError,
} from "./acp-agent.js";
import { createIdleRuntimeCleanup } from "../idle-runtime-cleanup/index.js";
import type { ProcessTerminator, TreeKillTarget } from "../../../utils/tree-kill.js";
import {
  COPILOT_AGENT_FEATURE_OPTION,
  COPILOT_ALLOW_ALL_MODE_ID,
  COPILOT_MODES,
  CopilotACPAgentClient,
  beforeCopilotModeWriter,
  transformCopilotConfigOptions,
  transformCopilotModeId,
  transformCopilotSessionResponse,
  writeCopilotProviderMode,
} from "./copilot-acp-agent.js";
import { GenericACPAgentClient } from "./generic-acp-agent.js";
import { parseKiroExtensionCommands } from "./kiro-acp-agent.js";
import { transformPiModels } from "./pi/agent.js";
import type { AgentStreamEvent } from "../agent-sdk-types.js";
import type {
  AgentCapabilityFlags,
  AgentPersistenceHandle,
  AgentSession,
  ProviderRefreshContext,
} from "../agent-sdk-types.js";
import { buildProviderRegistry } from "../provider-registry.js";
import { createTestLogger } from "../../../test-utils/test-logger.js";
import { buildStringCommandShellInvocation } from "../../../utils/string-command-shell.js";
import { asInternals } from "../../test-utils/class-mocks.js";
import {
  cancelInFlightImportProcess,
  countInFlightImportProcesses,
} from "../import-session-cancellation.js";
import * as spawnUtils from "../../../utils/spawn.js";

describe("buildACPClientCapabilities", () => {
  test("enables terminal execution on the host while keeping filesystem operations with the agent by default", () => {
    expect(buildACPClientCapabilities()).toEqual({
      fs: {
        readTextFile: false,
        writeTextFile: false,
      },
      terminal: true,
    });
  });

  test("applies provider capability overrides without dropping metadata", () => {
    expect(
      buildACPClientCapabilities(
        { source: "provider" },
        {
          fs: {
            readTextFile: true,
          },
          terminal: false,
        },
      ),
    ).toEqual({
      fs: {
        readTextFile: true,
        writeTextFile: false,
      },
      terminal: false,
      _meta: { source: "provider" },
    });
  });
});

interface ACPSessionInternals {
  sessionId: string | null;
  connection: {
    prompt: (...args: unknown[]) => Promise<PromptResponse>;
    cancel?: (...args: unknown[]) => Promise<void>;
    extMethod?: (
      method: string,
      params: Record<string, unknown>,
    ) => Promise<Record<string, unknown>>;
  };
  agentMeta: Record<string, unknown> | null;
  activeForegroundTurnId: string | null;
  configOptions: SessionConfigOption[];
  translateSessionUpdate(update: SessionUpdate): AgentStreamEvent[];
  acpMcpServers(): unknown[];
  handleProviderExit(
    code: number | null,
    signal: NodeJS.Signals | null,
    stderrChunks: string[],
  ): void;
}

interface ACPModelSelectionInternals {
  sessionId: string | null;
  connection: {
    setSessionConfigOption: (input: {
      sessionId: string;
      configId: string;
      value: string;
    }) => Promise<unknown>;
  };
  configOptions: SessionConfigOption[];
}

interface ACPConfiguredOverrideInternals {
  sessionId: string | null;
  connection: {
    setSessionMode: (input: { sessionId: string; modeId: string }) => Promise<void>;
    setSessionConfigOption: (input: {
      sessionId: string;
      configId: string;
      value: string;
    }) => Promise<unknown>;
    unstable_setSessionModel?: (input: { sessionId: string; modelId: string }) => Promise<void>;
  };
  configOptions: SessionConfigOption[];
  availableModes: Array<{ id: string; label: string; description?: string }>;
  availableModels: Array<{ modelId: string; name: string; description?: string | null }> | null;
  currentMode: string | null;
  currentModel: string | null;
  applyConfiguredOverrides(): Promise<void>;
}

function createSession(
  options: {
    terminateProcess?: ProcessTerminator;
    launchEnv?: Record<string, string>;
  } = {},
): ACPAgentSession {
  const { terminateProcess, launchEnv } = options;
  return new ACPAgentSession(
    {
      provider: "claude-acp",
      cwd: "/tmp/paseo-acp-test",
    },
    {
      provider: "claude-acp",
      logger: createTestLogger(),
      defaultCommand: ["claude", "--acp"],
      defaultModes: [],
      capabilities: {
        supportsStreaming: true,
        supportsSessionPersistence: true,
        supportsDynamicModes: true,
        supportsMcpServers: true,
        supportsReasoningStream: true,
        supportsToolInvocations: true,
      },
      ...(terminateProcess ? { terminateProcess } : {}),
      ...(launchEnv ? { launchEnv } : {}),
    },
  );
}

// Typed substitute for the real tree-kill terminator. Records which child
// processes it was asked to terminate, so tests assert on observable state
// instead of spying on the production function. In "deferred" mode the
// terminations hang until releaseAll(), letting tests observe parallelism.
class FakeTerminator {
  readonly terminated: TreeKillTarget[] = [];
  private readonly pending: Array<() => void> = [];

  constructor(private readonly mode: "immediate" | "deferred" = "immediate") {}

  readonly terminate: ProcessTerminator = async (child) => {
    this.terminated.push(child);
    if (this.mode === "deferred") {
      await new Promise<void>((resolve) => this.pending.push(resolve));
    }
    return "terminated";
  };

  releaseAll(): void {
    for (const resolve of this.pending.splice(0)) {
      resolve();
    }
  }
}

function createSessionWithConfig(
  config: {
    provider?: string;
    modeId?: string | null;
    model?: string | null;
    featureValues?: Record<string, unknown>;
  } = {},
  logger: ReturnType<typeof createTestLogger> = createTestLogger(),
): ACPAgentSession {
  return new ACPAgentSession(
    {
      provider: config.provider ?? "claude-acp",
      cwd: "/tmp/paseo-acp-test",
      modeId: config.modeId ?? undefined,
      model: config.model ?? undefined,
      featureValues: config.featureValues,
    },
    {
      provider: config.provider ?? "claude-acp",
      logger,
      defaultCommand: ["claude", "--acp"],
      defaultModes: [],
      capabilities: {
        supportsStreaming: true,
        supportsSessionPersistence: true,
        supportsDynamicModes: true,
        supportsMcpServers: true,
        supportsReasoningStream: true,
        supportsToolInvocations: true,
      },
    },
  );
}

function createKiroSession(
  options: { waitForInitialCommands?: boolean; initialCommandsWaitTimeoutMs?: number } = {},
  logger: ReturnType<typeof createTestLogger> = createTestLogger(),
): ACPAgentSession {
  return new ACPAgentSession(
    {
      provider: "kiro",
      cwd: "/tmp/paseo-acp-test",
    },
    {
      provider: "kiro",
      logger,
      defaultCommand: ["kiro-cli", "acp"],
      defaultModes: [],
      capabilities: {
        supportsStreaming: true,
        supportsSessionPersistence: true,
        supportsDynamicModes: true,
        supportsMcpServers: true,
        supportsReasoningStream: true,
        supportsToolInvocations: true,
      },
      extensionCommandsParser: parseKiroExtensionCommands,
      waitForInitialCommands: options.waitForInitialCommands ?? false,
      initialCommandsWaitTimeoutMs: options.initialCommandsWaitTimeoutMs,
    },
  );
}

function createTerminalChildStub(): ChildProcess {
  const child = new EventEmitter() as ChildProcess;
  child.stdout = new EventEmitter() as ChildProcess["stdout"];
  child.stderr = new EventEmitter() as ChildProcess["stderr"];
  child.kill = vi.fn(() => true) as ChildProcess["kill"];
  return child;
}

function createDestroyableStream(): { destroyed: boolean; destroy: () => void } {
  const stream = {
    destroyed: false,
    destroy() {
      stream.destroyed = true;
    },
  };
  return stream;
}

function createProbeChildStub(): ChildProcessWithoutNullStreams {
  const child = new EventEmitter() as ChildProcessWithoutNullStreams;
  child.stdin = createDestroyableStream() as unknown as ChildProcessWithoutNullStreams["stdin"];
  child.stdout = createDestroyableStream() as unknown as ChildProcessWithoutNullStreams["stdout"];
  child.stderr = createDestroyableStream() as unknown as ChildProcessWithoutNullStreams["stderr"];
  child.kill = vi.fn(() => true) as ChildProcessWithoutNullStreams["kill"];
  return child;
}

function selectConfigOption(
  category: "mode" | "model" | "thought_level",
  values: string[],
  currentValue = values[0] ?? "",
): SessionConfigOption {
  return {
    id: `${category}-option`,
    name: selectConfigOptionName(category),
    category,
    type: "select",
    currentValue,
    options: values.map((value) => ({ value, name: value })),
  };
}

function createCopilotSessionWithConfig(
  modeId?: string | null,
  featureValues?: Record<string, unknown>,
): ACPAgentSession {
  return new ACPAgentSession(
    {
      provider: "copilot",
      cwd: "/tmp/paseo-acp-test",
      modeId: modeId ?? undefined,
      ...(featureValues ? { featureValues } : {}),
    },
    {
      provider: "copilot",
      logger: createTestLogger(),
      defaultCommand: ["copilot", "--acp"],
      defaultModes: COPILOT_MODES,
      sessionResponseTransformer: transformCopilotSessionResponse,
      configOptionsTransformer: transformCopilotConfigOptions,
      configFeatureOptions: [COPILOT_AGENT_FEATURE_OPTION],
      modeIdTransformer: transformCopilotModeId,
      providerModeWriter: writeCopilotProviderMode,
      beforeModeWriter: beforeCopilotModeWriter,
      capabilities: {
        supportsStreaming: true,
        supportsSessionPersistence: true,
        supportsDynamicModes: true,
        supportsMcpServers: true,
        supportsReasoningStream: true,
        supportsToolInvocations: true,
      },
    },
  );
}

function copilotModeConfigOption(currentValue: string): SessionConfigOption {
  return {
    id: "mode",
    name: "Mode",
    category: "mode",
    type: "select",
    currentValue,
    options: [
      {
        value: "https://agentclientprotocol.com/protocol/session-modes#agent",
        name: "Agent",
      },
      {
        value: "https://agentclientprotocol.com/protocol/session-modes#plan",
        name: "Plan",
      },
      {
        value: "https://agentclientprotocol.com/protocol/session-modes#autopilot",
        name: "Autopilot",
      },
    ],
  };
}

function copilotAllowAllConfigOption(currentValue: "on" | "off"): SessionConfigOption {
  return {
    id: "allow_all",
    name: "Allow All",
    category: "permissions",
    type: "select",
    currentValue,
    options: [
      { value: "on", name: "On" },
      { value: "off", name: "Off" },
    ],
  };
}

function copilotAgentConfigOption(currentValue: string): SessionConfigOption {
  return {
    id: "agent",
    name: "Agent",
    category: "_agent",
    type: "select",
    currentValue,
    options: [
      {
        value: "",
        name: "",
      },
      {
        value: "Probe Agent",
        name: "Probe Agent",
        description: "Temporary probe agent",
      },
    ],
  };
}

function selectConfigOptionName(category: "mode" | "model" | "thought_level"): string {
  if (category === "mode") {
    return "Mode";
  }
  if (category === "model") {
    return "Model";
  }
  return "Thinking";
}

function prepareConfiguredOverrideSession(
  session: ACPAgentSession,
  options: {
    currentMode?: string | null;
    availableModes?: Array<{ id: string; label: string; description?: string }>;
    currentModel?: string | null;
    availableModels?: Array<{ modelId: string; name: string; description?: string | null }> | null;
    configOptions?: SessionConfigOption[];
    connection?: Partial<ACPConfiguredOverrideInternals["connection"]>;
  } = {},
): {
  internals: ACPConfiguredOverrideInternals;
  setSessionMode: ReturnType<typeof vi.fn>;
  unstableSetSessionModel: ReturnType<typeof vi.fn>;
  setSessionConfigOption: ReturnType<typeof vi.fn>;
} {
  const setSessionMode = vi.fn(async () => undefined);
  const unstableSetSessionModel = vi.fn(async () => undefined);
  const setSessionConfigOption = vi.fn(async () => ({
    configOptions: options.configOptions ?? [],
  }));
  const internals = asInternals<ACPConfiguredOverrideInternals>(session);
  internals.sessionId = "session-1";
  internals.connection = {
    setSessionMode,
    setSessionConfigOption,
    unstable_setSessionModel: unstableSetSessionModel,
    ...options.connection,
  };
  internals.availableModes = options.availableModes ?? [];
  internals.availableModels = options.availableModels ?? null;
  internals.configOptions = options.configOptions ?? [];
  internals.currentMode = options.currentMode ?? null;
  internals.currentModel = options.currentModel ?? null;

  return { internals, setSessionMode, unstableSetSessionModel, setSessionConfigOption };
}

test("ACP setModel only uses config-option fallback when the matching select choice contains the model", async () => {
  const logger = createTestLogger();
  const childLogger = { trace: vi.fn(), warn: vi.fn() };
  vi.spyOn(logger, "child").mockReturnValue(asInternals<typeof logger>(childLogger));
  const session = createSessionWithConfig({}, logger);
  const setSessionConfigOption = vi.fn(async () => ({
    configOptions: [
      {
        id: "model-option",
        name: "Model",
        category: "model",
        type: "select",
        currentValue: "sonnet",
        options: [{ value: "sonnet", name: "Sonnet" }],
      },
    ],
  }));
  const internals = asInternals<ACPModelSelectionInternals>(session);
  internals.sessionId = "session-1";
  internals.connection = { setSessionConfigOption };
  internals.configOptions = [
    {
      id: "model-option",
      name: "Model",
      category: "model",
      type: "select",
      currentValue: "sonnet",
      options: [{ value: "sonnet", name: "Sonnet" }],
    },
  ];

  await session.setModel("sonnet");

  expect(setSessionConfigOption).toHaveBeenCalledWith({
    sessionId: "session-1",
    configId: "model-option",
    value: "sonnet",
  });

  setSessionConfigOption.mockClear();

  await expect(session.setModel("new-provider-model")).resolves.toBeUndefined();
  expect(childLogger.warn).toHaveBeenCalledWith(
    { value: "new-provider-model" },
    expect.stringContaining("is not a valid claude-acp model config option"),
  );
  expect(setSessionConfigOption).not.toHaveBeenCalled();
});

describe("createLoggedNdJsonStream", () => {
  test("routes malformed ACP stdout through the provider logger instead of console.error", async () => {
    const input = new TransformStream<Uint8Array, Uint8Array>();
    const output = new TransformStream<Uint8Array, Uint8Array>();
    const logger = {
      warn: vi.fn(),
    };
    const consoleError = vi.spyOn(console, "error").mockImplementation(() => {});

    const stream = createLoggedNdJsonStream(output.writable, input.readable, {
      logger: asInternals<ReturnType<typeof createTestLogger>>(logger),
      provider: "gemini",
    });
    const reader = stream.readable.getReader();
    const writer = input.writable.getWriter();

    await writer.write(
      new TextEncoder().encode(
        'Please visit the following URL to authorize the application:\n{"jsonrpc":"2.0","method":"ok","params":{}}\n',
      ),
    );

    const parsed = await reader.read();

    expect(parsed.value).toEqual({ jsonrpc: "2.0", method: "ok", params: {} });
    expect(logger.warn).toHaveBeenCalledWith(
      expect.objectContaining({
        err: {
          type: "SyntaxError",
          message: "ACP stdout line was not valid JSON",
        },
        provider: "gemini",
      }),
      "ACP agent emitted non-JSON stdout; ignoring line",
    );
    expect(logger.warn.mock.calls[0]?.[0]).not.toHaveProperty("linePreview");
    expect(consoleError).not.toHaveBeenCalled();

    await writer.close();
    reader.releaseLock();
    consoleError.mockRestore();
  });

  test("normalizes stringified numeric ACP response ids", async () => {
    const input = new TransformStream<Uint8Array, Uint8Array>();
    const output = new TransformStream<Uint8Array, Uint8Array>();
    const logger = {
      warn: vi.fn(),
    };

    const stream = createLoggedNdJsonStream(output.writable, input.readable, {
      logger: asInternals<ReturnType<typeof createTestLogger>>(logger),
      provider: "deepseek-tui",
    });
    const reader = stream.readable.getReader();
    const writer = input.writable.getWriter();

    await writer.write(
      new TextEncoder().encode('{"jsonrpc":"2.0","id":"0","result":{"ok":true}}\n'),
    );

    const parsed = await reader.read();

    expect(parsed.value).toEqual({ jsonrpc: "2.0", id: 0, result: { ok: true } });
    expect(logger.warn).not.toHaveBeenCalled();

    await writer.close();
    reader.releaseLock();
  });

  test("does not log terminal control sequences from malformed ACP stdout", async () => {
    const input = new TransformStream<Uint8Array, Uint8Array>();
    const output = new TransformStream<Uint8Array, Uint8Array>();
    const logger = {
      warn: vi.fn(),
    };

    const stream = createLoggedNdJsonStream(output.writable, input.readable, {
      logger: asInternals<ReturnType<typeof createTestLogger>>(logger),
      provider: "gemini",
    });
    const reader = stream.readable.getReader();
    const writer = input.writable.getWriter();

    await writer.write(new TextEncoder().encode('\u001b[1G\u001b[0JEn\n{"ok":true}\n'));

    const parsed = await reader.read();

    expect(parsed.value).toEqual({ ok: true });
    expect(JSON.stringify(logger.warn.mock.calls)).not.toContain("\u001b");
    expect(JSON.stringify(logger.warn.mock.calls)).not.toContain("[1G");
    expect(logger.warn.mock.calls[0]?.[0]).toEqual({
      err: {
        type: "SyntaxError",
        message: "ACP stdout line was not valid JSON",
      },
      provider: "gemini",
    });

    await writer.close();
    reader.releaseLock();
  });
});

describe("isMiniMaxCodeProvider", () => {
  test("recognizes the provider's own executable, not a path that mentions it", () => {
    expect(isMiniMaxCodeProvider("minimax-code", ["minimax-code", "acp"])).toBe(true);
    expect(isMiniMaxCodeProvider(undefined, ["/opt/tools/mcode", "acp"])).toBe(true);

    // A custom ACP agent that merely lives under a directory called
    // "minimax-experiments" is not MiniMax Code: sending it the vendor
    // handshake announces extensions it does not understand.
    expect(isMiniMaxCodeProvider("custom-acp", ["/Users/x/minimax-experiments/agent", "acp"])).toBe(
      false,
    );
  });
});

describe("mergeACPClientCapabilityMeta", () => {
  test("adds the vendor handshake to the user's own client metadata", () => {
    // The provider answers its extension notifications only when the
    // handshake carries the vendor key, while a user-supplied metadata object
    // may still be needed. One must not replace the other.
    expect(mergeACPClientCapabilityMeta(MCODE_CLIENT_CAPABILITY_META, { source: "user" })).toEqual({
      "minimax-code/extensions": { version: 1, notifications: true },
      source: "user",
    });

    // For the same key the user's fields win, and the rest of the vendor
    // request survives.
    expect(
      mergeACPClientCapabilityMeta(MCODE_CLIENT_CAPABILITY_META, {
        "minimax-code/extensions": { notifications: false },
      }),
    ).toEqual({ "minimax-code/extensions": { version: 1, notifications: false } });
  });
});

describe("ACPAgentSession terminal tools", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  test("runs single-string terminal commands through the platform shell", async () => {
    const child = createTerminalChildStub();
    const spawn = vi.spyOn(spawnUtils, "spawnProcess").mockReturnValue(child);
    const session = createSession();
    const shell = buildStringCommandShellInvocation({
      command: "git -C /repo status --short",
      windowsShell: "cmd",
    });

    await session.createTerminal({
      sessionId: "session-1",
      command: "git -C /repo status --short",
      cwd: "/repo",
    });

    expect(spawn).toHaveBeenCalledWith(
      shell.shell,
      shell.args,
      expect.objectContaining({
        cwd: "/repo",
        envOverlay: expect.objectContaining({ BASH_ENV: undefined }),
        shell: false,
      }),
    );
  });

  test("preserves cmd semantics for single-string terminal commands on Windows", async () => {
    const originalPlatform = process.platform;
    Object.defineProperty(process, "platform", {
      value: "win32",
      configurable: true,
    });
    try {
      const child = createTerminalChildStub();
      const spawn = vi.spyOn(spawnUtils, "spawnProcess").mockReturnValue(child);
      const session = createSession();

      await session.createTerminal({
        sessionId: "session-1",
        command: "echo %TEMP% && echo ok",
        cwd: "C:\\repo",
      });

      expect(spawn).toHaveBeenCalledWith(
        "cmd.exe",
        ["/c", "echo %TEMP% && echo ok"],
        expect.objectContaining({
          cwd: "C:\\repo",
          envOverlay: expect.objectContaining({ BASH_ENV: undefined }),
          shell: false,
        }),
      );
    } finally {
      Object.defineProperty(process, "platform", {
        value: originalPlatform,
        configurable: true,
      });
    }
  });

  test("preserves explicit terminal argv", async () => {
    const child = createTerminalChildStub();
    const spawn = vi.spyOn(spawnUtils, "spawnProcess").mockReturnValue(child);
    const session = createSession();

    await session.createTerminal({
      sessionId: "session-1",
      command: "git",
      args: ["status", "--short"],
      cwd: "/repo",
    });

    expect(spawn).toHaveBeenCalledWith(
      "git",
      ["status", "--short"],
      expect.objectContaining({ cwd: "/repo" }),
    );
  });

  // Terminals the daemon opens for an agent are siblings of the agent process, so
  // they inherit nothing from it. They carry the agent's identity only if the
  // session puts its launch environment on them, the way its transport spawn does.
  describe("agent launch environment", () => {
    afterEach(() => {
      vi.unstubAllEnvs();
    });

    const readLaunchIdentityArgs = [
      "-e",
      "process.stdout.write(JSON.stringify({ id: process.env.PASEO_AGENT_ID ?? null, cwd: process.env.PASEO_AGENT_CWD ?? null, extra: process.env.PASEO_TEST_EXTRA ?? null }))",
    ];

    async function readLaunchIdentity(
      session: ACPAgentSession,
      params: Partial<CreateTerminalRequest> = {},
    ): Promise<{ id: string | null; cwd: string | null; extra: string | null }> {
      const terminal = await session.createTerminal({
        sessionId: "session-1",
        command: process.execPath,
        args: readLaunchIdentityArgs,
        cwd: process.cwd(),
        ...params,
      });
      await session.waitForTerminalExit({
        sessionId: "session-1",
        terminalId: terminal.terminalId,
      });
      const { output } = await session.terminalOutput({
        sessionId: "session-1",
        terminalId: terminal.terminalId,
      });
      return JSON.parse(output.trim());
    }

    test("gives the terminal the agent's identity", async () => {
      // The daemon itself does not run inside a Paseo agent.
      vi.stubEnv("PASEO_AGENT_ID", undefined);
      vi.stubEnv("PASEO_AGENT_CWD", undefined);
      const session = createSession({
        launchEnv: {
          PASEO_AGENT_ID: "agent-1",
          PASEO_AGENT_CWD: "/repo",
        },
      });

      await expect(readLaunchIdentity(session)).resolves.toMatchObject({
        id: "agent-1",
        cwd: "/repo",
      });
    });

    test("prefers the agent's identity over the daemon's own environment", async () => {
      // A daemon started from inside another Paseo agent carries that agent's id.
      vi.stubEnv("PASEO_AGENT_ID", "daemon-host-agent");
      vi.stubEnv("PASEO_AGENT_CWD", "/elsewhere");
      const session = createSession({
        launchEnv: {
          PASEO_AGENT_ID: "agent-1",
          PASEO_AGENT_CWD: "/repo",
        },
      });

      await expect(readLaunchIdentity(session)).resolves.toMatchObject({
        id: "agent-1",
        cwd: "/repo",
      });
    });

    test("lets the requested terminal environment win over the launch environment", async () => {
      const session = createSession({
        launchEnv: {
          PASEO_AGENT_ID: "agent-1",
          PASEO_TEST_EXTRA: "from-launch",
        },
      });

      await expect(
        readLaunchIdentity(session, {
          env: [{ name: "PASEO_TEST_EXTRA", value: "from-request" }],
        }),
      ).resolves.toMatchObject({ id: "agent-1", extra: "from-request" });
    });

    test("carries the agent's identity into single-string shell commands", async () => {
      const child = createTerminalChildStub();
      const spawn = vi.spyOn(spawnUtils, "spawnProcess").mockReturnValue(child);
      const session = createSession({ launchEnv: { PASEO_AGENT_ID: "agent-1" } });

      await session.createTerminal({
        sessionId: "session-1",
        command: "paseo heartbeat create --every 5m",
        cwd: "/repo",
      });

      expect(spawn).toHaveBeenCalledWith(
        expect.any(String),
        expect.any(Array),
        expect.objectContaining({
          envOverlay: expect.objectContaining({
            PASEO_AGENT_ID: "agent-1",
            BASH_ENV: undefined,
          }),
        }),
      );
    });
  });

  test("surfaces spawn errors through terminal output and waitForTerminalExit", async () => {
    const child = createTerminalChildStub();
    vi.spyOn(spawnUtils, "spawnProcess").mockReturnValue(child);
    const session = createSession();

    const terminal = await session.createTerminal({
      sessionId: "session-1",
      command: "missing-command",
    });
    child.emit("error", new Error("spawn missing-command ENOENT"));

    await expect(
      session.waitForTerminalExit({
        sessionId: "session-1",
        terminalId: terminal.terminalId,
      }),
    ).rejects.toThrow("spawn missing-command ENOENT");
    await expect(
      session.terminalOutput({
        sessionId: "session-1",
        terminalId: terminal.terminalId,
      }),
    ).resolves.toMatchObject({
      output: "spawn missing-command ENOENT\n",
      truncated: false,
    });
  });

  test("trims large terminal output without rescanning the retained suffix", async () => {
    const session = createSession();
    const terminal = await session.createTerminal({
      sessionId: "session-1",
      command: process.execPath,
      args: ["-e", 'process.stdout.write("x".repeat(1024 * 1024))'],
      cwd: process.cwd(),
      outputByteLimit: 1024,
    });
    await session.waitForTerminalExit({ sessionId: "session-1", terminalId: terminal.terminalId });

    await expect(
      session.terminalOutput({ sessionId: "session-1", terminalId: terminal.terminalId }),
    ).resolves.toEqual({
      output: "x".repeat(1024),
      truncated: true,
      exitStatus: { exitCode: 0, signal: null },
    });
  }, 1000);

  test("trims terminal output at a UTF-8 character boundary", async () => {
    const session = createSession();
    const terminal = await session.createTerminal({
      sessionId: "session-1",
      command: process.execPath,
      args: ["-e", 'process.stdout.write("😀ab")'],
      cwd: process.cwd(),
      outputByteLimit: 3,
    });
    await session.waitForTerminalExit({ sessionId: "session-1", terminalId: terminal.terminalId });

    await expect(
      session.terminalOutput({ sessionId: "session-1", terminalId: terminal.terminalId }),
    ).resolves.toEqual({
      output: "ab",
      truncated: true,
      exitStatus: { exitCode: 0, signal: null },
    });
  });
});

describe("mapACPUsage", () => {
  test("maps ACP usage fields into Paseo usage", () => {
    expect(
      mapACPUsage({
        inputTokens: 11,
        outputTokens: 7,
        totalTokens: 18,
        cachedReadTokens: 5,
      }),
    ).toEqual({
      inputTokens: 11,
      outputTokens: 7,
      cachedInputTokens: 5,
    });
  });
});

describe("ACP context-window usage", () => {
  async function emitUsageUpdate(update: {
    used: number;
    size: number;
  }): Promise<{ events: unknown[] }> {
    const session = createSessionWithConfig({ provider: "dsh" });
    asInternals<ACPSessionInternals>(session).sessionId = "session-1";
    const events: unknown[] = [];
    session.subscribe((event) => {
      if (event.type === "usage_updated") events.push(event);
    });
    await session.sessionUpdate({
      sessionId: "session-1",
      update: { sessionUpdate: "usage_update", used: update.used, size: update.size },
    });
    return { events };
  }

  test("forwards usage_update as context-window usage state", async () => {
    const { events } = await emitUsageUpdate({ used: 13_759, size: 1_000_000 });

    expect(events).toEqual([
      {
        type: "usage_updated",
        provider: "dsh",
        usage: { contextWindowMaxTokens: 1_000_000, contextWindowUsedTokens: 13_759 },
      },
    ]);
  });

  test("emits nothing when size and used cannot both drive a meter", async () => {
    await expect(
      emitUsageUpdate({ used: -1, size: 0 }).then((result) => result.events),
    ).resolves.toEqual([]);
    await expect(
      emitUsageUpdate({ used: 13_759, size: 0 }).then((result) => result.events),
    ).resolves.toEqual([]);
  });
});

describe("deriveModesFromACP", () => {
  test("prefers explicit ACP mode state", () => {
    const result = deriveModesFromACP(
      [{ id: "fallback", label: "Fallback" }],
      {
        availableModes: [
          { id: "default", name: "Always Ask", description: "Prompt before tools" },
          { id: "plan", name: "Plan", description: "Read only" },
        ],
        currentModeId: "plan",
      },
      [],
    );

    expect(result).toEqual({
      modes: [
        { id: "default", label: "Always Ask", description: "Prompt before tools" },
        { id: "plan", label: "Plan", description: "Read only" },
      ],
      currentModeId: "plan",
    });
  });

  test("falls back to config options when explicit mode state is absent", () => {
    const result = deriveModesFromACP([{ id: "fallback", label: "Fallback" }], null, [
      {
        id: "mode",
        name: "Mode",
        category: "mode",
        type: "select",
        currentValue: "acceptEdits",
        options: [
          { value: "default", name: "Always Ask" },
          { value: "acceptEdits", name: "Accept File Edits" },
        ],
      },
    ]);

    expect(result).toEqual({
      modes: [
        { id: "default", label: "Always Ask", description: undefined },
        { id: "acceptEdits", label: "Accept File Edits", description: undefined },
      ],
      currentModeId: "acceptEdits",
    });
  });

  test("returns an empty mode list when fallback modes are empty and config only exposes thought levels", () => {
    const result = deriveModesFromACP([], null, [
      {
        id: "thought_level",
        name: "Thinking",
        category: "thought_level",
        type: "select",
        currentValue: "medium",
        options: [
          { value: "low", name: "Low" },
          { value: "medium", name: "Medium" },
          { value: "high", name: "High" },
        ],
      },
    ]);

    expect(result).toEqual({
      modes: [],
      currentModeId: null,
    });
  });
});

describe("ACP selection validity helpers", () => {
  test("classifies advertised ACP modes and select config option choices", () => {
    const result = resolveACPModeSelection({
      modeId: "plan",
      availableModes: [
        { id: "default", label: "Always Ask" },
        { id: "plan", label: "Plan" },
      ],
      configOptions: [
        {
          id: "mode",
          name: "Mode",
          category: "mode",
          type: "select",
          currentValue: "default",
          options: [{ value: "default", name: "Always Ask" }],
        },
      ],
    });

    expect(result).toMatchObject({
      availableMode: { id: "plan", label: "Plan" },
      configChoice: null,
      hasAvailableModes: true,
    });
    expect(result.configOption?.id).toBe("mode");
  });

  test("classifies model select config option choices separately from advertised models", () => {
    const result = resolveACPModelSelection({
      modelId: "opus",
      availableModels: [{ modelId: "sonnet", name: "Sonnet", description: null }],
      configOptions: [
        {
          id: "model",
          name: "Model",
          category: "model",
          type: "select",
          currentValue: "sonnet",
          options: [
            {
              group: "Anthropic",
              options: [{ value: "opus", name: "Opus", description: "Deep" }],
            },
          ],
        },
      ],
    });

    expect(result).toMatchObject({
      availableModel: null,
      configChoice: {
        value: "opus",
        name: "Opus",
        description: "Deep",
        group: "Anthropic",
      },
      hasAvailableModels: true,
    });
    expect(result.configOption?.id).toBe("model");
  });
});

describe("ACPAgentSession Zed parity", () => {
  test("applies valid stored mode/model values, routes current_mode_update, and skips invalid Cursor-style stored values with warnings", async () => {
    const validSession = createSessionWithConfig({ modeId: "plan", model: "sonnet" });
    const valid = prepareConfiguredOverrideSession(validSession, {
      currentMode: "default",
      availableModes: [
        { id: "default", label: "Always Ask" },
        { id: "plan", label: "Plan" },
      ],
      currentModel: "haiku",
      availableModels: [
        { modelId: "haiku", name: "Haiku", description: null },
        { modelId: "sonnet", name: "Sonnet", description: null },
      ],
    });

    await valid.internals.applyConfiguredOverrides();
    expect(valid.setSessionMode).toHaveBeenCalledWith({ sessionId: "session-1", modeId: "plan" });
    expect(valid.unstableSetSessionModel).toHaveBeenCalledWith({
      sessionId: "session-1",
      modelId: "sonnet",
    });

    const modeEvents = asInternals<ACPSessionInternals>(validSession).translateSessionUpdate({
      sessionUpdate: "current_mode_update",
      currentModeId: "default",
    });
    expect(modeEvents).toEqual([
      {
        type: "mode_changed",
        provider: "claude-acp",
        currentModeId: "default",
        availableModes: [
          { id: "default", label: "Always Ask" },
          { id: "plan", label: "Plan" },
        ],
      },
    ]);
    expect(await validSession.getCurrentMode()).toBe("default");

    const logger = createTestLogger();
    const childLogger = { trace: vi.fn(), warn: vi.fn() };
    vi.spyOn(logger, "child").mockReturnValue(asInternals<typeof logger>(childLogger));
    const invalidSession = createSessionWithConfig(
      { modeId: "acceptEdits", model: "opus" },
      logger,
    );
    const invalid = prepareConfiguredOverrideSession(invalidSession, {
      currentMode: "default",
      availableModes: [
        { id: "default", label: "Always Ask" },
        { id: "plan", label: "Plan" },
      ],
      currentModel: "sonnet",
      availableModels: [{ modelId: "sonnet", name: "Sonnet", description: null }],
    });

    await expect(invalid.internals.applyConfiguredOverrides()).resolves.toBeUndefined();
    expect(invalid.setSessionMode).not.toHaveBeenCalled();
    expect(invalid.unstableSetSessionModel).not.toHaveBeenCalled();
    expect(childLogger.warn).toHaveBeenCalledWith(
      { value: expect.stringContaining("acceptEdits") },
      expect.stringContaining("not valid"),
    );
    expect(childLogger.warn).toHaveBeenCalledWith(
      { value: expect.stringContaining("opus") },
      expect.stringContaining("not a valid"),
    );
  });

  test("does not use config-option fallback when Cursor-style availableModes omit the stored mode", async () => {
    const session = createSessionWithConfig({ modeId: "acceptEdits" });
    const { internals, setSessionConfigOption } = prepareConfiguredOverrideSession(session, {
      currentMode: "default",
      availableModes: [
        { id: "default", label: "Always Ask" },
        { id: "plan", label: "Plan" },
      ],
      configOptions: [selectConfigOption("mode", ["default", "acceptEdits"], "default")],
      connection: { unstable_setSessionModel: undefined },
    });

    await expect(internals.applyConfiguredOverrides()).resolves.toBeUndefined();
    expect(setSessionConfigOption).not.toHaveBeenCalled();
  });

  test("does not fail session start when configured model cannot be applied by ACP", async () => {
    const logger = createTestLogger();
    const childLogger = { trace: vi.fn(), warn: vi.fn() };
    vi.spyOn(logger, "child").mockReturnValue(asInternals<typeof logger>(childLogger));
    const session = createSessionWithConfig(
      { provider: "deepseek-tui", model: "deepseek/v4" },
      logger,
    );
    const { internals, setSessionConfigOption, unstableSetSessionModel } =
      prepareConfiguredOverrideSession(session, {
        currentModel: null,
        availableModels: null,
        configOptions: [],
        connection: { unstable_setSessionModel: undefined },
      });

    await expect(internals.applyConfiguredOverrides()).resolves.toBeUndefined();
    expect(unstableSetSessionModel).not.toHaveBeenCalled();
    expect(setSessionConfigOption).not.toHaveBeenCalled();
    expect(childLogger.warn).toHaveBeenCalledWith(
      { value: "deepseek/v4" },
      "deepseek-tui does not expose ACP model selection; using provider default model",
    );
  });

  test("routes config_option_update and refreshes derived mode, model, and thinking state", async () => {
    const session = createSession();
    const internals = asInternals<ACPSessionInternals>(session);

    const events = internals.translateSessionUpdate({
      sessionUpdate: "config_option_update",
      configOptions: [
        selectConfigOption("mode", ["default", "plan"], "plan"),
        selectConfigOption("model", ["sonnet", "opus"], "opus"),
        selectConfigOption("thought_level", ["low", "high"], "high"),
      ],
    });

    expect(events).toMatchObject([
      {
        type: "mode_changed",
        provider: "claude-acp",
        currentModeId: "plan",
        availableModes: [
          { id: "default", label: "default" },
          { id: "plan", label: "plan" },
        ],
      },
      {
        type: "model_changed",
        provider: "claude-acp",
        runtimeInfo: expect.objectContaining({
          model: "opus",
          thinkingOptionId: "high",
          modeId: "plan",
        }),
      },
      {
        type: "thinking_option_changed",
        provider: "claude-acp",
        thinkingOptionId: "high",
      },
    ]);
    expect(internals.configOptions).toEqual([
      selectConfigOption("mode", ["default", "plan"], "plan"),
      selectConfigOption("model", ["sonnet", "opus"], "opus"),
      selectConfigOption("thought_level", ["low", "high"], "high"),
    ]);
    expect(await session.getAvailableModes()).toEqual([
      { id: "default", label: "default" },
      { id: "plan", label: "plan" },
    ]);
    expect(await session.getCurrentMode()).toBe("plan");
    await expect(session.getRuntimeInfo()).resolves.toMatchObject({
      model: "opus",
      thinkingOptionId: "high",
      modeId: "plan",
    });
  });

  test("keeps pushed mode when a later config_option_update has no mode payload", async () => {
    const session = createSession();
    const internals = asInternals<ACPSessionInternals>(session);

    internals.translateSessionUpdate({
      sessionUpdate: "current_mode_update",
      currentModeId: "plan",
    });
    const events = internals.translateSessionUpdate({
      sessionUpdate: "config_option_update",
      configOptions: [selectConfigOption("model", ["sonnet"], "sonnet")],
    });

    expect(events.map((event) => event.type)).toEqual(["model_changed"]);
    expect(await session.getCurrentMode()).toBe("plan");
    await expect(session.getRuntimeInfo()).resolves.toMatchObject({
      model: "sonnet",
      modeId: "plan",
    });
  });

  test("uses last writer when current_mode_update and config_option_update both include a mode", async () => {
    const session = createSession();
    const internals = asInternals<ACPSessionInternals>(session);

    const configEvents = internals.translateSessionUpdate({
      sessionUpdate: "config_option_update",
      configOptions: [selectConfigOption("mode", ["default", "plan"], "plan")],
    });
    const modeEvents = internals.translateSessionUpdate({
      sessionUpdate: "current_mode_update",
      currentModeId: "default",
    });

    expect(configEvents).toMatchObject([{ type: "mode_changed", currentModeId: "plan" }]);
    expect(modeEvents).toMatchObject([{ type: "mode_changed", currentModeId: "default" }]);
    expect(await session.getCurrentMode()).toBe("default");
  });

  test("uses canonical mode returned by setSessionConfigOption response", async () => {
    const session = createSession();
    const internals = asInternals<ACPModelSelectionInternals>(session);
    const events: AgentStreamEvent[] = [];
    const unsubscribe = session.subscribe((event) => events.push(event));
    internals.sessionId = "session-1";
    internals.configOptions = [selectConfigOption("mode", ["ask", "default"], "ask")];
    internals.connection = {
      setSessionConfigOption: vi.fn(async () => ({
        configOptions: [selectConfigOption("mode", ["ask", "default"], "default")],
      })),
    };

    await session.setMode("ask");
    unsubscribe();

    expect(await session.getCurrentMode()).toBe("default");
    expect(events).toMatchObject([
      {
        type: "mode_changed",
        provider: "claude-acp",
        currentModeId: "default",
        availableModes: [
          { id: "ask", label: "ask" },
          { id: "default", label: "default" },
        ],
      },
    ]);
  });

  test("uses canonical model returned by setSessionConfigOption response", async () => {
    const session = createSession();
    const internals = asInternals<ACPModelSelectionInternals>(session);
    const events: AgentStreamEvent[] = [];
    const unsubscribe = session.subscribe((event) => events.push(event));
    internals.sessionId = "session-1";
    internals.configOptions = [selectConfigOption("model", ["claude-sonnet", "sonnet"], "sonnet")];
    internals.connection = {
      setSessionConfigOption: vi.fn(async () => ({
        configOptions: [selectConfigOption("model", ["claude-sonnet", "sonnet"], "sonnet")],
      })),
    };

    await session.setModel("claude-sonnet");
    unsubscribe();

    await expect(session.getRuntimeInfo()).resolves.toMatchObject({ model: "sonnet" });
    expect(events).toContainEqual({
      type: "model_changed",
      provider: "claude-acp",
      runtimeInfo: expect.objectContaining({ model: "sonnet" }),
    });
  });

  test("uses canonical thinking option returned by setSessionConfigOption response", async () => {
    const session = createSession();
    const internals = asInternals<ACPModelSelectionInternals>(session);
    const events: AgentStreamEvent[] = [];
    const unsubscribe = session.subscribe((event) => events.push(event));
    internals.sessionId = "session-1";
    internals.configOptions = [
      selectConfigOption("thought_level", ["think-hard", "high"], "think-hard"),
    ];
    internals.connection = {
      setSessionConfigOption: vi.fn(async () => ({
        configOptions: [selectConfigOption("thought_level", ["think-hard", "high"], "high")],
      })),
    };

    await session.setThinkingOption("think-hard");
    unsubscribe();

    await expect(session.getRuntimeInfo()).resolves.toMatchObject({ thinkingOptionId: "high" });
    expect(events).toContainEqual({
      type: "thinking_option_changed",
      provider: "claude-acp",
      thinkingOptionId: "high",
    });
  });

  test("passes generic ACP permission requests through to the user", async () => {
    const session = createSessionWithConfig({
      provider: "cursor-acp",
      modeId: "https://agentclientprotocol.com/protocol/session-modes#agent",
    });
    const events: AgentStreamEvent[] = [];
    const permissionOptions: PermissionOption[] = [
      { optionId: "allow-once", name: "Allow", kind: "allow_once" },
      { optionId: "reject-once", name: "Reject", kind: "reject_once" },
    ];

    asInternals<ACPSessionInternals>(session).sessionId = "session-1";
    session.subscribe((event) => {
      events.push(event);
    });

    const permission = session.requestPermission({
      sessionId: "session-1",
      toolCall: {
        toolCallId: "tool-1",
        title: "Edit file",
        kind: "edit",
        status: "pending",
      },
      options: permissionOptions,
    } satisfies RequestPermissionRequest);

    await Promise.resolve();

    const requested = events.find((event) => event.type === "permission_requested");
    expect(requested).toMatchObject({
      type: "permission_requested",
      request: {
        actions: [
          { id: "allow-once", label: "Allow", behavior: "allow" },
          { id: "reject-once", label: "Reject", behavior: "deny" },
        ],
      },
    });
    if (requested?.type !== "permission_requested") {
      throw new Error("Expected permission request");
    }

    await session.respondToPermission(requested.request.id, { behavior: "allow" });
    await expect(permission).resolves.toEqual({
      outcome: { outcome: "selected", optionId: "allow-once" },
    });
  });

  test("preserves ACP chooser actions and returns the selected option", async () => {
    const session = createSessionWithConfig({
      provider: "kimi-acp",
      modeId: "https://agentclientprotocol.com/protocol/session-modes#agent",
    });
    const events: AgentStreamEvent[] = [];

    asInternals<ACPSessionInternals>(session).sessionId = "session-1";
    session.subscribe((event) => events.push(event));

    const permission = session.requestPermission({
      sessionId: "session-1",
      toolCall: {
        toolCallId: "question-1",
        title: "AskUserQuestion",
        status: "pending",
        content: [
          {
            type: "content",
            content: {
              type: "text",
              text: "Which path should Paseo take?",
            },
          },
        ],
      },
      options: [
        { optionId: "q0_opt_0", name: "Narrow fix", kind: "allow_once" },
        { optionId: "q0_opt_1", name: "Protocol fix", kind: "allow_once" },
        { optionId: "q0_skip", name: "Skip", kind: "reject_once" },
      ],
    } satisfies RequestPermissionRequest);

    await Promise.resolve();

    const requested = events.find((event) => event.type === "permission_requested");
    expect(requested).toMatchObject({
      type: "permission_requested",
      request: {
        detail: {
          type: "plain_text",
          label: "AskUserQuestion",
          text: "Which path should Paseo take?",
        },
        actions: [
          { id: "q0_opt_0", label: "Narrow fix", behavior: "allow" },
          { id: "q0_opt_1", label: "Protocol fix", behavior: "allow" },
          { id: "q0_skip", label: "Skip", behavior: "deny" },
        ],
      },
    });
    if (requested?.type !== "permission_requested") {
      throw new Error("Expected permission request");
    }

    await session.respondToPermission(requested.request.id, {
      behavior: "allow",
      selectedActionId: "q0_opt_1",
    });

    await expect(permission).resolves.toEqual({
      outcome: { outcome: "selected", optionId: "q0_opt_1" },
    });
  });

  test("preserves ACP permission requests after invalid selected actions", async () => {
    const session = createSessionWithConfig({ provider: "generic-acp" });
    const events: AgentStreamEvent[] = [];

    asInternals<ACPSessionInternals>(session).sessionId = "session-1";
    session.subscribe((event) => events.push(event));

    const permission = session.requestPermission({
      sessionId: "session-1",
      toolCall: {
        toolCallId: "tool-1",
        title: "Edit file",
        kind: "edit",
        status: "pending",
      },
      options: [
        { optionId: "allow-once", name: "Allow", kind: "allow_once" },
        { optionId: "reject-once", name: "Reject", kind: "reject_once" },
      ],
    } satisfies RequestPermissionRequest);

    await Promise.resolve();

    const requested = events.find((event) => event.type === "permission_requested");
    if (requested?.type !== "permission_requested") {
      throw new Error("Expected permission request");
    }

    await expect(
      session.respondToPermission(requested.request.id, {
        behavior: "allow",
        selectedActionId: "",
      }),
    ).rejects.toThrow("does not exist");
    expect(session.getPendingPermissions()).toHaveLength(1);

    await expect(
      session.respondToPermission(requested.request.id, {
        behavior: "deny",
        selectedActionId: "allow-once",
      }),
    ).rejects.toThrow("does not match 'deny' behavior");
    expect(session.getPendingPermissions()).toHaveLength(1);

    await session.respondToPermission(requested.request.id, {
      behavior: "deny",
      selectedActionId: "reject-once",
    });
    await expect(permission).resolves.toEqual({
      outcome: { outcome: "selected", optionId: "reject-once" },
    });
  });

  test("auto-accepts ACP permission requests when the shared feature is enabled", async () => {
    const session = createSessionWithConfig({
      provider: "cursor-acp",
      featureValues: { auto_accept: true },
    });
    const events: Array<{ type: string }> = [];

    asInternals<ACPSessionInternals>(session).sessionId = "session-1";
    session.subscribe((event) => events.push(event as { type: string }));

    await expect(
      session.requestPermission({
        sessionId: "session-1",
        toolCall: {
          toolCallId: "tool-1",
          title: "Edit file",
          kind: "edit",
          status: "pending",
        },
        options: [
          { optionId: "allow-once", name: "Allow", kind: "allow_once" },
          { optionId: "allow-always", name: "Always allow", kind: "allow_always" },
          { optionId: "reject-once", name: "Reject", kind: "reject_once" },
        ],
      } satisfies RequestPermissionRequest),
    ).resolves.toEqual({
      outcome: { outcome: "selected", optionId: "allow-once" },
    });
    expect(events).not.toContainEqual(expect.objectContaining({ type: "permission_requested" }));
    expect(session.getPendingPermissions()).toEqual([]);
  });

  test("does not auto-accept ACP chooser requests", async () => {
    const session = createSessionWithConfig({
      provider: "kimi-acp",
      featureValues: { auto_accept: true },
    });
    const events: AgentStreamEvent[] = [];

    asInternals<ACPSessionInternals>(session).sessionId = "session-1";
    session.subscribe((event) => events.push(event));

    const permission = session.requestPermission({
      sessionId: "session-1",
      toolCall: {
        toolCallId: "question-1",
        title: "AskUserQuestion",
        status: "pending",
      },
      options: [
        { optionId: "q0_opt_0", name: "Narrow fix", kind: "allow_once" },
        { optionId: "q0_opt_1", name: "Protocol fix", kind: "allow_once" },
        { optionId: "q0_skip", name: "Skip", kind: "reject_once" },
      ],
    } satisfies RequestPermissionRequest);

    await Promise.resolve();

    const requested = events.find((event) => event.type === "permission_requested");
    expect(requested).toMatchObject({
      type: "permission_requested",
      request: {
        actions: [
          { id: "q0_opt_0", label: "Narrow fix", behavior: "allow" },
          { id: "q0_opt_1", label: "Protocol fix", behavior: "allow" },
          { id: "q0_skip", label: "Skip", behavior: "deny" },
        ],
      },
    });
    if (requested?.type !== "permission_requested") {
      throw new Error("Expected permission request");
    }

    await session.respondToPermission(requested.request.id, {
      behavior: "allow",
      selectedActionId: "q0_opt_0",
    });
    await expect(permission).resolves.toEqual({
      outcome: { outcome: "selected", optionId: "q0_opt_0" },
    });
  });

  test("starts auto-accepting permissions when the shared feature is toggled on", async () => {
    const session = createSessionWithConfig({ provider: "generic-acp" });
    const internals = asInternals<ACPSessionInternals>(session);
    internals.sessionId = "session-1";
    internals.connection = {};

    await session.setFeature("auto_accept", true);

    expect(session.features).toContainEqual(
      expect.objectContaining({ type: "toggle", id: "auto_accept", value: true }),
    );
    await expect(
      session.requestPermission({
        sessionId: "session-1",
        toolCall: {
          toolCallId: "tool-1",
          title: "Run command",
          kind: "execute",
          status: "pending",
        },
        options: [{ optionId: "allow-always", name: "Always allow", kind: "allow_always" }],
      } satisfies RequestPermissionRequest),
    ).resolves.toEqual({
      outcome: { outcome: "selected", optionId: "allow-always" },
    });
  });

  test("surfaces an ACP permission when auto-accept has no allow option", async () => {
    const session = createSessionWithConfig({ featureValues: { auto_accept: true } });
    const events: Array<{ type: string; request?: { id: string } }> = [];
    asInternals<ACPSessionInternals>(session).sessionId = "session-1";
    session.subscribe((event) => events.push(event as { type: string; request?: { id: string } }));

    const permission = session.requestPermission({
      sessionId: "session-1",
      toolCall: {
        toolCallId: "tool-1",
        title: "Unavailable approval",
        kind: "other",
        status: "pending",
      },
      options: [{ optionId: "reject-once", name: "Reject", kind: "reject_once" }],
    } satisfies RequestPermissionRequest);
    await Promise.resolve();

    const requested = events.find((event) => event.type === "permission_requested");
    expect(requested?.request?.id).toEqual(expect.any(String));
    await session.respondToPermission(requested!.request!.id, { behavior: "deny" });
    await expect(permission).resolves.toEqual({
      outcome: { outcome: "selected", optionId: "reject-once" },
    });
  });

  test("maps Copilot Allow All mode to allow_all ACP config on session start", async () => {
    const setSessionConfigOption = vi.fn(async () => ({
      configOptions: [
        copilotModeConfigOption("https://agentclientprotocol.com/protocol/session-modes#agent"),
        copilotAllowAllConfigOption("on"),
      ],
    }));
    const setSessionMode = vi.fn(async () => undefined);
    const session = createCopilotSessionWithConfig(COPILOT_ALLOW_ALL_MODE_ID);
    const { internals } = prepareConfiguredOverrideSession(session, {
      currentMode: "https://agentclientprotocol.com/protocol/session-modes#agent",
      availableModes: COPILOT_MODES,
      configOptions: [
        copilotModeConfigOption("https://agentclientprotocol.com/protocol/session-modes#agent"),
        copilotAllowAllConfigOption("off"),
      ],
      connection: { setSessionConfigOption, setSessionMode },
    });
    const events: AgentStreamEvent[] = [];
    const unsubscribe = session.subscribe((event) => events.push(event));
    await internals.applyConfiguredOverrides();
    unsubscribe();

    expect(setSessionConfigOption).toHaveBeenCalledWith({
      sessionId: "session-1",
      configId: "allow_all",
      value: "on",
    });
    // Allow All is agent mode with permissions pre-granted, so it has to leave
    // whatever mode the session is in, not just flip the flag.
    expect(setSessionMode).toHaveBeenCalledWith({
      sessionId: "session-1",
      modeId: "https://agentclientprotocol.com/protocol/session-modes#agent",
    });
    await expect(session.getCurrentMode()).resolves.toBe(COPILOT_ALLOW_ALL_MODE_ID);
    expect(events.some((event) => event.type === "permission_requested")).toBe(false);
  });

  test("accepts Copilot's legacy autopilot mode ID as Allow All", async () => {
    const setSessionConfigOption = vi.fn(async () => ({
      configOptions: [
        copilotModeConfigOption("https://agentclientprotocol.com/protocol/session-modes#agent"),
        copilotAllowAllConfigOption("on"),
      ],
    }));
    const setSessionMode = vi.fn(async () => undefined);
    const session = createCopilotSessionWithConfig();
    prepareConfiguredOverrideSession(session, {
      currentMode: "https://agentclientprotocol.com/protocol/session-modes#agent",
      availableModes: COPILOT_MODES,
      configOptions: [
        copilotModeConfigOption("https://agentclientprotocol.com/protocol/session-modes#agent"),
        copilotAllowAllConfigOption("off"),
      ],
      connection: { setSessionConfigOption, setSessionMode },
    });

    await session.setMode("https://agentclientprotocol.com/protocol/session-modes#autopilot");

    expect(setSessionConfigOption).toHaveBeenCalledWith({
      sessionId: "session-1",
      configId: "allow_all",
      value: "on",
    });
    // Allow All is agent mode with permissions pre-granted, so it has to leave
    // whatever mode the session is in, not just flip the flag.
    expect(setSessionMode).toHaveBeenCalledWith({
      sessionId: "session-1",
      modeId: "https://agentclientprotocol.com/protocol/session-modes#agent",
    });
    await expect(session.getCurrentMode()).resolves.toBe(COPILOT_ALLOW_ALL_MODE_ID);
  });

  test("switching Copilot away from Allow All turns allow_all off before setting the ACP mode", async () => {
    const setSessionConfigOption = vi.fn(async (input: { value: string }) => ({
      configOptions: [
        copilotModeConfigOption("https://agentclientprotocol.com/protocol/session-modes#agent"),
        copilotAllowAllConfigOption(input.value === "on" ? "on" : "off"),
      ],
    }));
    const setSessionMode = vi.fn(async () => undefined);
    const session = createCopilotSessionWithConfig(COPILOT_ALLOW_ALL_MODE_ID);
    prepareConfiguredOverrideSession(session, {
      currentMode: COPILOT_ALLOW_ALL_MODE_ID,
      availableModes: COPILOT_MODES,
      configOptions: [
        copilotModeConfigOption(COPILOT_ALLOW_ALL_MODE_ID),
        copilotAllowAllConfigOption("on"),
      ],
      connection: { setSessionConfigOption, setSessionMode },
    });

    await session.setMode("https://agentclientprotocol.com/protocol/session-modes#agent");

    expect(setSessionConfigOption).toHaveBeenCalledWith({
      sessionId: "session-1",
      configId: "allow_all",
      value: "off",
    });
    expect(setSessionMode).toHaveBeenCalledWith({
      sessionId: "session-1",
      modeId: "https://agentclientprotocol.com/protocol/session-modes#agent",
    });
  });

  test("trusts Copilot allow_all config updates as the current mode source", async () => {
    const session = createCopilotSessionWithConfig();
    const internals = asInternals<ACPSessionInternals>(session);

    const events = internals.translateSessionUpdate({
      sessionUpdate: "config_option_update",
      configOptions: [
        copilotModeConfigOption("https://agentclientprotocol.com/protocol/session-modes#agent"),
        copilotAllowAllConfigOption("on"),
      ],
    });

    expect(events).toMatchObject([
      {
        type: "mode_changed",
        provider: "copilot",
        currentModeId: COPILOT_ALLOW_ALL_MODE_ID,
        availableModes: expect.arrayContaining([
          expect.objectContaining({ id: COPILOT_ALLOW_ALL_MODE_ID, label: "Allow All" }),
        ]),
      },
    ]);
    await expect(session.getCurrentMode()).resolves.toBe(COPILOT_ALLOW_ALL_MODE_ID);
  });

  test("exposes Copilot custom agents as a select feature", () => {
    const session = createCopilotSessionWithConfig();
    const internals = asInternals<ACPSessionInternals>(session);
    internals.configOptions = [copilotAgentConfigOption("")];

    expect(session.features).toEqual([
      expect.objectContaining({
        type: "toggle",
        id: "auto_accept",
        value: false,
      }),
      {
        type: "select",
        id: "agent",
        label: "Agent",
        description: "Use a Copilot custom agent profile",
        tooltip: "Select Copilot agent",
        icon: undefined,
        value: "",
        options: [
          {
            id: "",
            label: "Default",
            description: undefined,
            isDefault: true,
            metadata: undefined,
          },
          {
            id: "Probe Agent",
            label: "Probe Agent",
            description: "Temporary probe agent",
            isDefault: false,
            metadata: undefined,
          },
        ],
      },
    ]);
  });

  test("applies configured Copilot custom agent before the first turn", async () => {
    const setSessionConfigOption = vi.fn(async () => ({
      configOptions: [copilotAgentConfigOption("Probe Agent")],
    }));
    const session = createCopilotSessionWithConfig(null, { agent: "Probe Agent" });
    const { internals } = prepareConfiguredOverrideSession(session, {
      configOptions: [copilotAgentConfigOption("")],
      connection: { setSessionConfigOption },
    });

    await internals.applyConfiguredOverrides();

    expect(setSessionConfigOption).toHaveBeenCalledWith({
      sessionId: "session-1",
      configId: "agent",
      value: "Probe Agent",
    });
    expect(session.features).toEqual([
      expect.objectContaining({
        id: "auto_accept",
        value: false,
      }),
      expect.objectContaining({
        id: "agent",
        value: "Probe Agent",
      }),
    ]);
  });

  test("sets Copilot custom agent through ACP config options", async () => {
    const setSessionConfigOption = vi.fn(async () => ({
      configOptions: [copilotAgentConfigOption("Probe Agent")],
    }));
    const session = createCopilotSessionWithConfig();
    prepareConfiguredOverrideSession(session, {
      configOptions: [copilotAgentConfigOption("")],
      connection: { setSessionConfigOption },
    });

    await session.setFeature("agent", "Probe Agent");

    expect(setSessionConfigOption).toHaveBeenCalledWith({
      sessionId: "session-1",
      configId: "agent",
      value: "Probe Agent",
    });
    expect(session.features).toEqual([
      expect.objectContaining({
        id: "auto_accept",
        value: false,
      }),
      expect.objectContaining({
        id: "agent",
        value: "Probe Agent",
      }),
    ]);
  });
});

describe("deriveModelDefinitionsFromACP", () => {
  test("attaches shared thinking options to ACP model state", () => {
    const result = deriveModelDefinitionsFromACP(
      "claude-acp",
      {
        availableModels: [
          { modelId: "haiku", name: "Haiku", description: "Fast" },
          { modelId: "sonnet", name: "Sonnet", description: "Balanced" },
        ],
        currentModelId: "haiku",
      },
      [
        {
          id: "reasoning",
          name: "Reasoning",
          category: "thought_level",
          type: "select",
          currentValue: "medium",
          options: [
            { value: "low", name: "Low" },
            { value: "medium", name: "Medium" },
            { value: "high", name: "High" },
          ],
        },
      ],
    );

    expect(result).toEqual([
      {
        provider: "claude-acp",
        id: "haiku",
        label: "Haiku",
        description: "Fast",
        isDefault: true,
        thinkingOptions: [
          {
            id: "low",
            label: "Low",
            description: undefined,
            isDefault: false,
            metadata: undefined,
          },
          {
            id: "medium",
            label: "Medium",
            description: undefined,
            isDefault: true,
            metadata: undefined,
          },
          {
            id: "high",
            label: "High",
            description: undefined,
            isDefault: false,
            metadata: undefined,
          },
        ],
        defaultThinkingOptionId: "medium",
      },
      {
        provider: "claude-acp",
        id: "sonnet",
        label: "Sonnet",
        description: "Balanced",
        isDefault: false,
        thinkingOptions: [
          {
            id: "low",
            label: "Low",
            description: undefined,
            isDefault: false,
            metadata: undefined,
          },
          {
            id: "medium",
            label: "Medium",
            description: undefined,
            isDefault: true,
            metadata: undefined,
          },
          {
            id: "high",
            label: "High",
            description: undefined,
            isDefault: false,
            metadata: undefined,
          },
        ],
        defaultThinkingOptionId: "medium",
      },
    ]);
  });
});

describe("ACPAgentClient modelTransformer", () => {
  test("applies modelTransformer after deriving ACP models", async () => {
    class TestACPAgentClient extends ACPAgentClient {
      protected override async spawnProcess(): Promise<SpawnedACPProcess> {
        return {
          child: { kill: vi.fn(), exitCode: 0, signalCode: null, once: vi.fn() },
          connection: {
            newSession: vi.fn().mockResolvedValue({
              models: {
                availableModels: [
                  {
                    modelId: "openrouter/openai/gpt-4.1-mini",
                    name: "openrouter/openai/gpt-4.1-mini",
                    description: null,
                  },
                ],
                currentModelId: "openrouter/openai/gpt-4.1-mini",
              },
              configOptions: [],
            }),
          },
          initialize: { agentCapabilities: {} },
        } as SpawnedACPProcess;
      }

      protected override async closeProbe(): Promise<void> {}
    }

    const client = new TestACPAgentClient({
      provider: "pi",
      logger: createTestLogger(),
      defaultCommand: ["test-acp"],
      modelTransformer: transformPiModels,
    });

    await expect(
      client.fetchCatalog({ scope: "workspace", cwd: "/tmp/acp-models", force: false }),
    ).resolves.toEqual({
      models: [
        {
          provider: "pi",
          id: "openrouter/openai/gpt-4.1-mini",
          label: "gpt-4.1-mini",
          description: "openrouter/openai/gpt-4.1-mini",
          isDefault: true,
          thinkingOptions: undefined,
          defaultThinkingOptionId: undefined,
        },
      ],
      modes: [],
    });
  });
});

describe("ACPAgentClient catalog discovery without a model resolver", () => {
  test("never switches models during catalog discovery even with multiple models and a thinking picker", async () => {
    // Model discovery extensions live on providers that opt in. The base
    // client ships no catalog model resolver, so a slow or nonconforming ACP can't
    // stall its catalog probe on extra setSessionConfigOption round trips.
    const setSessionConfigOption = vi.fn();

    class TestACPAgentClient extends ACPAgentClient {
      protected override async spawnProcess(): Promise<SpawnedACPProcess> {
        return {
          child: { kill: vi.fn(), exitCode: 0, signalCode: null, once: vi.fn() },
          connection: {
            newSession: vi.fn().mockResolvedValue({
              sessionId: "session-1",
              configOptions: [
                {
                  id: "model",
                  name: "Model",
                  category: "model",
                  type: "select",
                  currentValue: "model-a",
                  options: [
                    { value: "model-a", name: "Model A" },
                    { value: "model-b", name: "Model B" },
                  ],
                },
                {
                  id: "thinking",
                  name: "Thinking",
                  category: "thought_level",
                  type: "select",
                  currentValue: "off",
                  options: [
                    { value: "off", name: "Off" },
                    { value: "on", name: "On" },
                  ],
                },
              ],
            }),
            setSessionConfigOption,
          },
          initialize: { agentCapabilities: {} },
        } as unknown as SpawnedACPProcess;
      }

      protected override async closeProbe(): Promise<void> {}
    }

    const client = new TestACPAgentClient({
      provider: "acp",
      logger: createTestLogger(),
      defaultCommand: ["acp-agent"],
      defaultModes: [],
    });

    const catalog = await client.fetchCatalog({
      scope: "workspace",
      cwd: "/tmp/acp-generic-catalog",
      force: false,
    });

    expect(setSessionConfigOption).not.toHaveBeenCalled();
    expect(catalog.models.map((model) => model.id)).toEqual(["model-a", "model-b"]);
  });
});

describe("ACPAgentClient config features", () => {
  test("enables Auto Accept for unattended ACP creation", () => {
    const client = new ACPAgentClient({
      provider: "generic-acp",
      logger: createTestLogger(),
      defaultCommand: ["generic-acp", "acp"],
    });

    expect(
      client.resolveCreateConfig({
        provider: "generic-acp",
        requestedMode: undefined,
        featureValues: { provider_feature: "kept" },
        parent: null,
        unattended: true,
        availableModes: [],
      }),
    ).toEqual({
      modeId: undefined,
      featureValues: { provider_feature: "kept", auto_accept: true },
    });
  });

  test("preserves an explicit Auto Accept override for unattended ACP creation", () => {
    const client = new ACPAgentClient({
      provider: "generic-acp",
      logger: createTestLogger(),
      defaultCommand: ["generic-acp", "acp"],
    });

    expect(
      client.resolveCreateConfig({
        provider: "generic-acp",
        requestedMode: undefined,
        featureValues: { auto_accept: false },
        parent: null,
        unattended: true,
        availableModes: [],
      }),
    ).toEqual({ modeId: undefined, featureValues: { auto_accept: false } });
  });

  test("maps an unattended cross-provider parent to ACP Auto Accept", () => {
    const client = new ACPAgentClient({
      provider: "generic-acp",
      logger: createTestLogger(),
      defaultCommand: ["generic-acp", "acp"],
    });

    expect(
      client.resolveCreateConfig({
        provider: "generic-acp",
        requestedMode: undefined,
        featureValues: undefined,
        parent: {
          provider: "claude",
          modeId: "bypassPermissions",
          isUnattended: true,
        },
        unattended: false,
        availableModes: [{ id: "agent", label: "Agent" }],
      }),
    ).toEqual({ modeId: undefined, featureValues: { auto_accept: true } });
  });

  test("treats Auto Accept as an unattended ACP configuration", () => {
    const client = new ACPAgentClient({
      provider: "generic-acp",
      logger: createTestLogger(),
      defaultCommand: ["generic-acp", "acp"],
    });

    expect(
      client.isCreateConfigUnattended({
        modeId: null,
        config: {
          provider: "generic-acp",
          cwd: "/tmp/acp-features",
          featureValues: { auto_accept: true },
        },
        availableModes: [],
      }),
    ).toBe(true);
  });

  test("exposes Auto Accept for every ACP provider without starting a probe", async () => {
    const client = new ACPAgentClient({
      provider: "generic-acp",
      logger: createTestLogger(),
      defaultCommand: ["generic-acp", "acp"],
    });

    await expect(
      client.listFeatures({
        provider: "generic-acp",
        cwd: "/tmp/acp-features",
        featureValues: { auto_accept: true },
      }),
    ).resolves.toEqual([
      expect.objectContaining({
        type: "toggle",
        id: "auto_accept",
        label: "Auto Accept",
        value: true,
      }),
    ]);
  });

  test("derives features from configured ACP select options", async () => {
    class TestACPAgentClient extends ACPAgentClient {
      protected override async spawnProcess(): Promise<SpawnedACPProcess> {
        return {
          child: { kill: vi.fn(), exitCode: 0, signalCode: null, once: vi.fn() },
          connection: {
            newSession: vi.fn().mockResolvedValue({
              sessionId: "session-1",
              configOptions: [copilotAgentConfigOption("Probe Agent")],
            }),
          },
          initialize: { agentCapabilities: {} },
        } as SpawnedACPProcess;
      }

      protected override async closeProbe(): Promise<void> {}
    }

    const client = new TestACPAgentClient({
      provider: "copilot",
      logger: createTestLogger(),
      defaultCommand: ["copilot", "--acp"],
      configFeatureOptions: [COPILOT_AGENT_FEATURE_OPTION],
    });

    await expect(
      client.listFeatures({
        provider: "copilot",
        cwd: "/tmp/acp-features",
      }),
    ).resolves.toEqual([
      expect.objectContaining({
        type: "toggle",
        id: "auto_accept",
        value: false,
      }),
      expect.objectContaining({
        type: "select",
        id: "agent",
        value: "Probe Agent",
        options: [
          expect.objectContaining({ id: "", label: "Default", isDefault: false }),
          expect.objectContaining({ id: "Probe Agent", label: "Probe Agent", isDefault: true }),
        ],
      }),
    ]);
  });
});

describe("ACPAgentClient sessionResponseTransformer", () => {
  class TestACPAgentClient extends ACPAgentClient {
    protected override async spawnProcess(): Promise<SpawnedACPProcess> {
      const response: SessionStateResponse = {
        sessionId: "session-1",
        modes: {
          availableModes: [{ id: "raw", name: "Raw", description: "Before transform" }],
          currentModeId: "raw",
        },
        models: null,
        configOptions: [],
      };

      return {
        child: { kill: vi.fn(), exitCode: 0, signalCode: null, once: vi.fn() },
        connection: {
          newSession: vi.fn().mockResolvedValue(response),
        },
        initialize: { agentCapabilities: {} },
      } as SpawnedACPProcess;
    }

    protected override async closeProbe(): Promise<void> {}
  }

  test("applies sessionResponseTransformer before deriving catalog modes", async () => {
    const client = new TestACPAgentClient({
      provider: "claude-acp",
      logger: createTestLogger(),
      defaultCommand: ["claude", "--acp"],
      defaultModes: [],
      sessionResponseTransformer: (response) => ({
        ...response,
        modes: {
          availableModes: [{ id: "review", name: "Review", description: "After transform" }],
          currentModeId: "review",
        },
      }),
    });

    await expect(
      client.fetchCatalog({ scope: "workspace", cwd: "/tmp/acp-modes", force: false }),
    ).resolves.toEqual({
      models: [],
      modes: [
        {
          id: "review",
          label: "Review",
          description: "After transform",
        },
      ],
    });
  });
});

describe("ACPAgentClient fetchCatalog", () => {
  test("passes the requested cwd to the catalog probe", async () => {
    const newSession = vi.fn().mockResolvedValue({ modes: null, models: null, configOptions: [] });

    class TestACPAgentClient extends ACPAgentClient {
      protected override async spawnProcess(): Promise<SpawnedACPProcess> {
        return {
          child: { kill: vi.fn(), exitCode: 0, signalCode: null, once: vi.fn() },
          connection: { newSession },
          initialize: { agentCapabilities: {} },
        } as SpawnedACPProcess;
      }

      protected override async closeProbe(): Promise<void> {}
    }

    const client = new TestACPAgentClient({
      provider: "pi",
      logger: createTestLogger(),
      defaultCommand: ["test-acp"],
      defaultModes: [],
    });

    await client.fetchCatalog({ scope: "workspace", cwd: "/tmp/acp-catalog-cwd", force: false });

    expect(newSession).toHaveBeenCalledWith({
      cwd: "/tmp/acp-catalog-cwd",
      mcpServers: [],
    });
  });

  test("returns an empty modes array when no ACP modes are reported and fallback modes are empty", async () => {
    class TestACPAgentClient extends ACPAgentClient {
      protected override async spawnProcess(): Promise<SpawnedACPProcess> {
        return {
          child: { kill: vi.fn(), exitCode: 0, signalCode: null, once: vi.fn() },
          connection: {
            newSession: vi.fn().mockResolvedValue({
              modes: null,
              configOptions: [
                {
                  id: "thought_level",
                  name: "Thinking",
                  category: "thought_level",
                  type: "select",
                  currentValue: "medium",
                  options: [
                    { value: "low", name: "Low" },
                    { value: "medium", name: "Medium" },
                    { value: "high", name: "High" },
                  ],
                },
              ],
            }),
          },
          initialize: { agentCapabilities: {} },
        } as SpawnedACPProcess;
      }

      protected override async closeProbe(): Promise<void> {}
    }

    const client = new TestACPAgentClient({
      provider: "pi",
      logger: createTestLogger(),
      defaultCommand: ["test-acp"],
      defaultModes: [],
    });

    await expect(
      client.fetchCatalog({ scope: "workspace", cwd: "/tmp/acp-modes", force: false }),
    ).resolves.toEqual({
      models: [],
      modes: [],
    });
  });
});

describe("ACPAgentClient listImportableSessions", () => {
  function makeClient(args: {
    listSessions: ReturnType<typeof vi.fn>;
    supportsList?: boolean;
    terminateProcess?: (child: unknown, options: unknown) => Promise<void>;
    child?: unknown;
  }) {
    const kill = vi.fn();
    const defaultChild = {
      kill,
      exitCode: 0,
      signalCode: null,
      once: vi.fn(),
      stdin: { destroy: vi.fn() },
      stdout: { destroy: vi.fn() },
      stderr: { destroy: vi.fn() },
    };
    class TestACPAgentClient extends ACPAgentClient {
      protected override async spawnProcess(): Promise<SpawnedACPProcess> {
        return {
          child: (args.child ?? defaultChild) as never,
          connection: { listSessions: args.listSessions },
          initialize: {
            agentCapabilities:
              args.supportsList === false ? {} : { sessionCapabilities: { list: {} } },
          },
        } as unknown as SpawnedACPProcess;
      }

      protected override async closeProbe(): Promise<void> {}
    }

    return new TestACPAgentClient({
      provider: "kimi",
      logger: createTestLogger(),
      defaultCommand: ["kimi", "acp"],
      defaultModes: [],
      ...(args.terminateProcess ? { terminateProcess: args.terminateProcess as never } : {}),
    });
  }

  test("forwards the requested cwd to session/list so the agent filters by directory", async () => {
    const listSessions = vi.fn().mockResolvedValue({
      sessions: [
        {
          sessionId: "session-1",
          cwd: "/Users/moonshot",
          title: "细致查看一下本仓库内容",
          updatedAt: "2026-06-13T00:00:00.000Z",
        },
      ],
      nextCursor: null,
    });

    const client = makeClient({ listSessions });
    const result = await client.listImportableSessions({ cwd: "/Users/moonshot", limit: 20 });

    expect(listSessions).toHaveBeenCalledWith({ cwd: "/Users/moonshot" });
    expect(result).toEqual([
      {
        providerHandleId: "session-1",
        cwd: "/Users/moonshot",
        title: "细致查看一下本仓库内容",
        firstPromptPreview: null,
        lastPromptPreview: null,
        lastActivityAt: new Date("2026-06-13T00:00:00.000Z"),
      },
    ]);
  });

  test("omits cwd from session/list when none is requested", async () => {
    const listSessions = vi.fn().mockResolvedValue({ sessions: [], nextCursor: null });
    const client = makeClient({ listSessions });

    await client.listImportableSessions({ limit: 20 });

    expect(listSessions).toHaveBeenCalledWith({});
  });

  test("stops at the requested session limit after a source-scoped page", async () => {
    const listSessions = vi.fn().mockResolvedValue({
      sessions: [
        { sessionId: "s1", cwd: "/Users/moonshot", title: null, updatedAt: null },
        { sessionId: "s2", cwd: "/Users/moonshot", title: null, updatedAt: null },
      ],
      nextCursor: "later",
    });
    const client = makeClient({ listSessions });

    await expect(
      client.listImportableSessions({ cwd: "/Users/moonshot", limit: 1 }),
    ).resolves.toHaveLength(1);
    expect(listSessions).toHaveBeenCalledTimes(1);
  });

  test("forwards cwd alongside the pagination cursor across pages", async () => {
    const listSessions = vi
      .fn()
      .mockResolvedValueOnce({
        sessions: [{ sessionId: "s1", cwd: "/Users/moonshot", title: null, updatedAt: null }],
        nextCursor: "cursor-2",
      })
      .mockResolvedValueOnce({
        sessions: [{ sessionId: "s2", cwd: "/Users/moonshot", title: null, updatedAt: null }],
        nextCursor: null,
      });

    const client = makeClient({ listSessions });
    await client.listImportableSessions({ cwd: "/Users/moonshot" });

    expect(listSessions).toHaveBeenNthCalledWith(1, { cwd: "/Users/moonshot" });
    expect(listSessions).toHaveBeenNthCalledWith(2, {
      cursor: "cursor-2",
      cwd: "/Users/moonshot",
    });
  });

  test("terminates the probe process when the listing is aborted", async () => {
    // The provider answers the first page and then stops answering. Without an
    // abort signal the probe keeps running for as long as the provider needs
    // to finish the requests it already received.
    let rejectSecondPage!: (error: Error) => void;
    const secondPage = new Promise<never>((_resolve, reject) => {
      rejectSecondPage = reject;
    });
    const listSessions = vi
      .fn()
      .mockResolvedValueOnce({
        sessions: [{ sessionId: "s1", cwd: "/Users/moonshot", title: null, updatedAt: null }],
        nextCursor: "cursor-2",
      })
      .mockReturnValue(secondPage);
    const terminateProcess = vi.fn().mockImplementation(async () => {
      rejectSecondPage(new Error("probe terminated"));
    });
    const client = makeClient({ listSessions, terminateProcess });
    const controller = new AbortController();

    const listing = client.listImportableSessions({
      cwd: "/Users/moonshot",
      signal: controller.signal,
    });
    // let the first page resolve so the listing is waiting on the second one
    await new Promise((resolve) => setTimeout(resolve, 0));
    controller.abort();

    await expect(listing).rejects.toThrow();
    expect(terminateProcess).toHaveBeenCalledTimes(1);
  });

  test("bounds session/list so a provider that never answers cannot hang the listing", async () => {
    vi.useFakeTimers();
    try {
      const listSessions = vi.fn().mockImplementation(() => new Promise<never>(() => {}));
      const client = makeClient({ listSessions });

      const assertion = expect(
        client.listImportableSessions({ cwd: "/Users/moonshot" }),
      ).rejects.toThrow(/session\/list timed out/);
      await vi.advanceTimersByTimeAsync(20_000);
      await assertion;
    } finally {
      vi.useRealTimers();
    }
  });
});

describe("ACPAgentClient listImportableSessions while the probe handshake stalls", () => {
  // The probe is only useful once its handshake finished, but the child exists
  // earlier than that. A provider that never answers `initialize` used to leave
  // a process nobody could reach: the abort handler only saw the probe after
  // the handshake, and nothing bounded the handshake itself.
  function makeClient(args: {
    initialize: () => Promise<{ agentCapabilities: Record<string, unknown> }>;
    terminateProcess?: ProcessTerminator;
  }) {
    const children: ChildProcessWithoutNullStreams[] = [];
    class TestACPAgentClient extends ACPAgentClient {
      // Only the transport is faked, so the real `spawnProcess` keeps its
      // handshake timeout and its cleanup path.
      protected override async spawnTransport() {
        const child = createProbeChildStub();
        children.push(child);
        return {
          child,
          connection: { initialize: args.initialize } as unknown as ClientSideConnection,
          stderrChunks: [],
          spawnReady: Promise.resolve(),
          spawnError: new Promise<never>(() => {}),
        };
      }
    }

    const client = new TestACPAgentClient({
      provider: "kimi",
      logger: createTestLogger(),
      defaultCommand: ["kimi", "acp"],
      defaultModes: [],
      ...(args.terminateProcess ? { terminateProcess: args.terminateProcess } : {}),
    });
    return { client, children };
  }

  test("fails the listing and kills the probe when the handshake never finishes", async () => {
    vi.useFakeTimers();
    try {
      const terminateProcess = vi.fn(async () => {});
      const { client, children } = makeClient({
        initialize: () => new Promise<never>(() => {}),
        terminateProcess,
      });

      const listing = client.listImportableSessions({ cwd: "/Users/moonshot" });
      const assertion = expect(listing).rejects.toThrow(/ACP initialize timed out/);

      // The provider is still in `initialize`: the listing waits, but not
      // forever.
      await vi.advanceTimersByTimeAsync(14_000);
      expect(terminateProcess).not.toHaveBeenCalled();

      await vi.advanceTimersByTimeAsync(1_000);
      await assertion;

      // The stalled probe is gone instead of lingering as an orphan.
      expect(children).toHaveLength(1);
      expect(terminateProcess).toHaveBeenCalledTimes(1);
      expect(terminateProcess.mock.calls[0]?.[0]).toBe(children[0]);
    } finally {
      vi.useRealTimers();
    }
  });

  test("kills the probe child when the listing is aborted during the handshake", async () => {
    vi.useFakeTimers();
    try {
      const terminateProcess = vi.fn(async () => {});
      let finishHandshake: (() => void) | undefined;
      const { client, children } = makeClient({
        initialize: () =>
          new Promise<{ agentCapabilities: Record<string, unknown> }>((resolve) => {
            finishHandshake = () => resolve({ agentCapabilities: {} });
          }),
        terminateProcess,
      });

      const controller = new AbortController();
      const listing = client.listImportableSessions({
        cwd: "/Users/moonshot",
        signal: controller.signal,
      });
      const assertion = expect(listing).rejects.toThrow(/aborted/);

      // The child exists while the handshake is still running - this is the
      // window in which an abort used to reach nothing.
      await vi.waitFor(() => expect(children).toHaveLength(1));
      controller.abort();
      await vi.advanceTimersByTimeAsync(0);

      expect(terminateProcess).toHaveBeenCalled();
      expect(terminateProcess.mock.calls[0]?.[0]).toBe(children[0]);

      // The caller already gave up, so the late handshake must not hand out a
      // probe that nobody waits for any more.
      finishHandshake?.();
      await assertion;
    } finally {
      vi.useRealTimers();
    }
  });
});

describe("ACPAgentClient listImportableSessions with a busy provider", () => {
  // Providers answer ACP requests one at a time. A session/load that never
  // finishes therefore blocks every later request on the same connection -
  // including the next session/list page, which used to time out and fail the
  // whole listing.
  function makeClient(args: {
    listSessions: ReturnType<typeof vi.fn>;
    loadSession: ReturnType<typeof vi.fn>;
    terminateProcess: (child: unknown, options: unknown) => Promise<void>;
    onSpawn?: (index: number, client: ACPClient | undefined) => void;
    now?: () => number;
  }) {
    const children: unknown[] = [];
    class TestACPAgentClient extends ACPAgentClient {
      protected override async spawnProcess(
        _launchEnv?: Record<string, string>,
        options?: { client?: ACPClient },
      ): Promise<SpawnedACPProcess> {
        const child = {
          kill: vi.fn(),
          exitCode: 0,
          signalCode: null,
          once: vi.fn(),
          stdin: { destroy: vi.fn() },
          stdout: { destroy: vi.fn() },
          stderr: { destroy: vi.fn() },
        };
        children.push(child);
        args.onSpawn?.(children.length - 1, options?.client);
        return {
          child: child as never,
          connection: { listSessions: args.listSessions, loadSession: args.loadSession },
          initialize: {
            agentCapabilities: { sessionCapabilities: { list: {} }, loadSession: true },
          },
        } as unknown as SpawnedACPProcess;
      }
    }

    const client = new TestACPAgentClient({
      provider: "kimi",
      logger: createTestLogger(),
      defaultCommand: ["kimi", "acp"],
      defaultModes: [],
      terminateProcess: args.terminateProcess as never,
      ...(args.now ? { now: args.now } : {}),
    });
    return { client, children };
  }

  function userMessage(text: string): SessionUpdate {
    return {
      sessionUpdate: "user_message_chunk",
      messageId: "user-1",
      content: { type: "text", text },
    } as SessionUpdate;
  }

  test("reads the next list page from a fresh probe after the provider got stuck", async () => {
    vi.useFakeTimers();
    try {
      let probeClient: ACPClient | undefined;
      let probeIndex = 0;
      const listProbes: number[] = [];
      const loadProbes: number[] = [];
      let page = 0;
      const listSessions = vi.fn(async () => {
        listProbes.push(probeIndex);
        page += 1;
        return page === 1
          ? {
              sessions: [
                {
                  sessionId: "stuck",
                  cwd: "/Users/moonshot",
                  title: "huge",
                  updatedAt: "2026-06-13T00:00:00.000Z",
                },
              ],
              nextCursor: "cursor-2",
            }
          : {
              sessions: [
                {
                  sessionId: "next",
                  cwd: "/Users/moonshot",
                  title: "next",
                  updatedAt: "2026-06-14T00:00:00.000Z",
                },
              ],
              nextCursor: null,
            };
      });
      let loads = 0;
      const loadSession = vi.fn(async (params: { sessionId: string }) => {
        loadProbes.push(probeIndex);
        loads += 1;
        // The first history never finishes: the provider is busy with it.
        if (loads === 1) return await new Promise<never>(() => {});
        await probeClient?.sessionUpdate({
          sessionId: params.sessionId,
          update: userMessage("hello"),
        });
      });

      const { client } = makeClient({
        listSessions,
        loadSession,
        terminateProcess: async () => {},
        onSpawn: (index, spawned) => {
          probeIndex = index;
          probeClient = spawned;
        },
      });

      const listing = client.listImportableSessions({ cwd: "/Users/moonshot" });
      await vi.advanceTimersByTimeAsync(5_000);
      const result = await listing;

      expect(result.map((session) => session.providerHandleId)).toEqual(["stuck", "next"]);
      // The stuck provider answers no request, so the next page has to come
      // from the replacement probe.
      expect(listProbes).toEqual([0, 1]);
      expect(loadProbes).toEqual([0, 1]);
    } finally {
      vi.useRealTimers();
    }
  });

  test("replaces a probe whose provider is stuck on a session history", async () => {
    vi.useFakeTimers();
    try {
      const terminateProcess = vi.fn(async () => {});
      let probeClient: ACPClient | undefined;
      const listSessions = vi.fn(async () => ({
        sessions: [
          {
            sessionId: "stuck",
            cwd: "/Users/moonshot",
            title: "huge",
            updatedAt: "2026-06-13T00:00:00.000Z",
          },
          {
            sessionId: "fast",
            cwd: "/Users/moonshot",
            title: "small",
            updatedAt: "2026-06-14T00:00:00.000Z",
          },
        ],
        nextCursor: null,
      }));
      let loads = 0;
      const loadSession = vi.fn(async (params: { sessionId: string }) => {
        loads += 1;
        // The first history never finishes: the provider is busy with it.
        if (loads === 1) return await new Promise<never>(() => {});
        await probeClient?.sessionUpdate({
          sessionId: params.sessionId,
          update: userMessage("hello"),
        });
      });

      const { client, children } = makeClient({
        listSessions,
        loadSession,
        terminateProcess,
        onSpawn: (_index, spawned) => {
          probeClient = spawned;
        },
      });

      const listing = client.listImportableSessions({ cwd: "/Users/moonshot" });
      await vi.advanceTimersByTimeAsync(5_000);
      const result = await listing;

      // The stuck session stays visible, the rest of the listing still works.
      expect(result.map((session) => session.providerHandleId)).toEqual(["stuck", "fast"]);
      expect(result[0].firstPromptPreview).toBeNull();
      expect(result[1].firstPromptPreview).toBe("hello");
      // The stuck provider was killed and replaced by a fresh probe.
      expect(children).toHaveLength(2);
      expect(terminateProcess).toHaveBeenCalledTimes(2);
      expect(terminateProcess.mock.calls[0]?.[0]).toBe(children[0]);
      expect(terminateProcess.mock.calls[1]?.[0]).toBe(children[1]);
    } finally {
      vi.useRealTimers();
    }
  });

  test("replaces the probe when the provider never answers a history load", async () => {
    vi.useFakeTimers();
    try {
      const terminateProcess = vi.fn(async () => {});
      let probeIndex = 0;
      const listProbes: number[] = [];
      const listSessions = vi.fn(async () => {
        listProbes.push(probeIndex);
        return {
          sessions: [
            {
              sessionId: "mute",
              cwd: "/Users/moonshot",
              title: "mute",
              updatedAt: "2026-06-13T00:00:00.000Z",
            },
          ],
          nextCursor: null,
        };
      });
      // The provider never answers, so the request stays pending on its side
      // and the connection never becomes free again.
      const loadSession = vi.fn(() => new Promise<never>(() => {}));

      const { client, children } = makeClient({
        listSessions,
        loadSession,
        terminateProcess,
        onSpawn: (index) => {
          probeIndex = index;
        },
      });

      const listing = client.listImportableSessions({ cwd: "/Users/moonshot" });
      // Inside the slow threshold the probe is still considered usable.
      await vi.advanceTimersByTimeAsync(4_000);
      expect(children).toHaveLength(1);

      // Once the threshold passes the probe is replaced, not kept.
      await vi.advanceTimersByTimeAsync(1_000);
      expect(children).toHaveLength(2);
      // The follow-up list page comes from the replacement probe instead of
      // queueing behind the load the provider is still working on.
      expect(listProbes).toEqual([0, 1]);

      const result = await listing;
      expect(result.map((session) => session.providerHandleId)).toEqual(["mute"]);
      expect(result[0].firstPromptPreview).toBeNull();
      expect(terminateProcess.mock.calls[0]?.[0]).toBe(children[0]);
    } finally {
      vi.useRealTimers();
    }
  });

  test("stops asking for previews when the listing budget is nearly spent", async () => {
    vi.useFakeTimers();
    try {
      const terminateProcess = vi.fn(async () => {});
      // The history budget is measured with the client clock. Moving it
      // forward leaves less than the slow-load threshold for the load, so a
      // load could only be abandoned by the guard - and the probe that would
      // be replaced afterwards would cost a fresh process for a result nobody
      // waits for.
      let clock = 0;
      const listSessions = vi.fn(async () => {
        clock = 41_000;
        return {
          sessions: [
            {
              sessionId: "slow",
              cwd: "/Users/moonshot",
              title: "slow",
              updatedAt: "2026-06-13T00:00:00.000Z",
            },
          ],
          nextCursor: null,
        };
      });
      const loadSession = vi.fn(() => new Promise<never>(() => {}));

      const { client, children } = makeClient({
        listSessions,
        loadSession,
        terminateProcess,
        now: () => clock,
      });

      const listing = client.listImportableSessions({ cwd: "/Users/moonshot" });

      // The session stays visible, without a preview, and no replacement
      // probe is started for a load that the budget cannot pay for.
      await expect(listing).resolves.toHaveLength(1);
      expect(loadSession).not.toHaveBeenCalled();
      expect(children).toHaveLength(1);
      // Only the cleanup of the one probe that was started - no replacement
      // probe was spawned for a load the budget cannot pay for.
      expect(terminateProcess).toHaveBeenCalledTimes(1);
      expect(terminateProcess.mock.calls[0]?.[0]).toBe(children[0]);
    } finally {
      vi.useRealTimers();
    }
  });

  test("kills the live probe when the listing is aborted after a recycle", async () => {
    vi.useFakeTimers();
    try {
      // The provider connection breaks when its process is killed, so the
      // pending request rejects the same way it does in a real run.
      let rejectPending: ((error: Error) => void) | undefined;
      const terminateProcess = vi.fn(async () => {
        rejectPending?.(new Error("probe terminated"));
      });
      let page = 0;
      const listSessions = vi.fn(async () => {
        page += 1;
        return page === 1
          ? {
              sessions: [
                {
                  sessionId: "s1",
                  cwd: "/Users/moonshot",
                  title: "one",
                  updatedAt: "2026-06-13T00:00:00.000Z",
                },
              ],
              nextCursor: "cursor-2",
            }
          : {
              sessions: [
                {
                  sessionId: "s2",
                  cwd: "/Users/moonshot",
                  title: "two",
                  updatedAt: "2026-06-14T00:00:00.000Z",
                },
              ],
              nextCursor: null,
            };
      });
      // Every history stays unanswered: the provider is stuck on each of them.
      const loadSession = vi.fn(
        async () =>
          await new Promise<never>((_resolve, reject) => {
            rejectPending = reject;
          }),
      );

      const { client, children } = makeClient({
        listSessions,
        loadSession,
        terminateProcess,
      });

      const controller = new AbortController();
      const listing = client.listImportableSessions({
        cwd: "/Users/moonshot",
        signal: controller.signal,
      });
      // The first history times out as slow and the probe is replaced.
      await vi.advanceTimersByTimeAsync(5_000);
      expect(children).toHaveLength(2);
      const killsBeforeAbort = terminateProcess.mock.calls.length;
      controller.abort();

      await expect(listing).rejects.toThrow(/aborted/);
      // The probe that was alive at the moment of the abort is the one that
      // gets killed, not the replaced one.
      const killsAfterAbort = terminateProcess.mock.calls.slice(killsBeforeAbort);
      expect(killsAfterAbort[0]?.[0]).toBe(children[1]);
    } finally {
      vi.useRealTimers();
    }
  });

  test("stops asking a stuck probe for previews once the recycle budget is spent", async () => {
    vi.useFakeTimers();
    try {
      const terminateProcess = vi.fn(async () => {});
      const stuckSessions = Array.from({ length: 5 }, (_value, index) => ({
        sessionId: `s${index + 1}`,
        cwd: "/Users/moonshot",
        title: `session ${index + 1}`,
        updatedAt: "2026-06-13T00:00:00.000Z",
      }));
      const listSessions = vi.fn(async () => ({
        sessions: stuckSessions,
        nextCursor: null,
      }));
      const loadSession = vi.fn(async () => await new Promise<never>(() => {}));

      const { client, children } = makeClient({
        listSessions,
        loadSession,
        terminateProcess,
      });

      const listing = client.listImportableSessions({ cwd: "/Users/moonshot" });
      await vi.advanceTimersByTimeAsync(20_000);
      const result = await listing;

      // The probe is replaced a bounded number of times and then left alone.
      expect(children).toHaveLength(4);
      expect(loadSession).toHaveBeenCalledTimes(4);
      // Every session stays visible, the ones without a preview included.
      expect(result.map((session) => session.providerHandleId)).toEqual([
        "s1",
        "s2",
        "s3",
        "s4",
        "s5",
      ]);
    } finally {
      vi.useRealTimers();
    }
  });

  test("skips the previews of the later pages once the probe is known to be stuck", async () => {
    vi.useFakeTimers();
    try {
      const terminateProcess = vi.fn(async () => {});
      const firstPage = Array.from({ length: 5 }, (_value, index) => ({
        sessionId: `s${index + 1}`,
        cwd: "/Users/moonshot",
        title: `session ${index + 1}`,
        updatedAt: "2026-06-13T00:00:00.000Z",
      }));
      const secondPage = Array.from({ length: 3 }, (_value, index) => ({
        sessionId: `s${index + 6}`,
        cwd: "/Users/moonshot",
        title: `session ${index + 6}`,
        updatedAt: "2026-06-14T00:00:00.000Z",
      }));
      const listSessions = vi.fn(async (params?: { cursor?: string }) =>
        params?.cursor
          ? { sessions: secondPage, nextCursor: null }
          : { sessions: firstPage, nextCursor: "cursor-2" },
      );
      const loadSession = vi.fn(async () => await new Promise<never>(() => {}));

      const { client } = makeClient({
        listSessions,
        loadSession,
        terminateProcess,
      });

      const listing = client.listImportableSessions({ cwd: "/Users/moonshot" });
      await vi.advanceTimersByTimeAsync(20_000);
      const result = await listing;

      // The probe is replaced a bounded number of times and then left alone,
      // so the sessions of the later page are listed without previews.
      expect(loadSession).toHaveBeenCalledTimes(4);
      expect(result.map((session) => session.providerHandleId)).toEqual([
        "s1",
        "s2",
        "s3",
        "s4",
        "s5",
        "s6",
        "s7",
        "s8",
      ]);
      expect(result.slice(5).every((session) => session.firstPromptPreview === null)).toBe(true);
    } finally {
      vi.useRealTimers();
    }
  });

  test("aborts immediately when the signal is already aborted", async () => {
    const terminateProcess = vi.fn(async () => {});
    const listSessions = vi.fn(async () => ({
      sessions: [
        {
          sessionId: "s1",
          cwd: "/Users/moonshot",
          title: "one",
          updatedAt: "2026-06-13T00:00:00.000Z",
        },
      ],
      nextCursor: null,
    }));
    const loadSession = vi.fn(async () => await new Promise<never>(() => {}));

    const { client, children } = makeClient({
      listSessions,
      loadSession,
      terminateProcess,
    });

    const controller = new AbortController();
    controller.abort();

    await expect(
      client.listImportableSessions({
        cwd: "/Users/moonshot",
        signal: controller.signal,
      }),
    ).rejects.toThrow(/aborted/);
    expect(listSessions).not.toHaveBeenCalled();
    // The caller gave up before the probe existed, so no process is started
    // at all - there is nothing left to terminate.
    expect(children).toHaveLength(0);
    expect(terminateProcess).not.toHaveBeenCalled();
  });
});

describe("ACPAgentClient listImportableSessions without previews", () => {
  // ACP has no cheap transcript read, so a preview costs a session/load - and
  // minimax-code answers that by restoring the whole session: 322 MB of
  // snapshots become 2.7 GB of RAM and a pinned CPU core. The import list still
  // has to show those sessions, so the preview is the only thing it drops.
  function makeClient(args: {
    listSessions: ReturnType<typeof vi.fn>;
    loadSession: ReturnType<typeof vi.fn>;
    onSpawn?: (client: ACPClient | undefined) => void;
  }) {
    class TestACPAgentClient extends ACPAgentClient {
      protected override async spawnProcess(
        _launchEnv?: Record<string, string>,
        options?: { client?: ACPClient },
      ): Promise<SpawnedACPProcess> {
        args.onSpawn?.(options?.client);
        return {
          child: {
            kill: vi.fn(),
            exitCode: 0,
            signalCode: null,
            once: vi.fn(),
            stdin: { destroy: vi.fn() },
            stdout: { destroy: vi.fn() },
            stderr: { destroy: vi.fn() },
          } as never,
          connection: { listSessions: args.listSessions, loadSession: args.loadSession },
          initialize: {
            agentCapabilities: { sessionCapabilities: { list: {} }, loadSession: true },
          },
        } as unknown as SpawnedACPProcess;
      }

      protected override async closeProbe(): Promise<void> {}
    }

    return new TestACPAgentClient({
      provider: "minimax-code",
      logger: createTestLogger(),
      defaultCommand: ["minimax-code", "acp"],
      defaultModes: [],
      terminateProcess: (async () => {}) as never,
    });
  }

  test("lists descriptors without a single history load when previews are excluded", async () => {
    const listSessions = vi.fn(async () => ({
      sessions: [
        {
          sessionId: "huge",
          cwd: "/Users/moonshot",
          title: "Release notes for 2.0",
          updatedAt: "2026-06-13T00:00:00.000Z",
        },
      ],
      nextCursor: null,
    }));
    const loadSession = vi.fn(async () => {
      throw new Error("session/load must not run when previews are excluded");
    });

    const client = makeClient({ listSessions, loadSession });

    const result = await client.listImportableSessions({
      cwd: "/Users/moonshot",
      query: "release notes",
      includePreviews: false,
    });

    // The provider is never asked to restore a session for the listing.
    expect(loadSession).not.toHaveBeenCalled();
    // The descriptor alone still describes the session, title included, so the
    // manager can match a query against it without any preview text.
    expect(result).toEqual([
      {
        providerHandleId: "huge",
        cwd: "/Users/moonshot",
        title: "Release notes for 2.0",
        firstPromptPreview: null,
        lastPromptPreview: null,
        lastActivityAt: new Date("2026-06-13T00:00:00.000Z"),
      },
    ]);
  });

  test("loads prompt previews when the caller does not exclude them", async () => {
    let probeClient: ACPClient | undefined;
    const listSessions = vi.fn(async () => ({
      sessions: [
        {
          sessionId: "small",
          cwd: "/Users/moonshot",
          title: "small",
          updatedAt: "2026-06-13T00:00:00.000Z",
        },
      ],
      nextCursor: null,
    }));
    const loadSession = vi.fn(async (params: { sessionId: string }) => {
      await probeClient?.sessionUpdate({
        sessionId: params.sessionId,
        update: {
          sessionUpdate: "user_message_chunk",
          messageId: "user-1",
          content: { type: "text", text: "hello" },
        } as SessionUpdate,
      });
    });

    const client = makeClient({
      listSessions,
      loadSession,
      onSpawn: (spawned) => {
        probeClient = spawned;
      },
    });

    const result = await client.listImportableSessions({ cwd: "/Users/moonshot", limit: 20 });

    expect(loadSession).toHaveBeenCalledTimes(1);
    expect(result[0].firstPromptPreview).toBe("hello");
  });
});

describe("ACP providers advertise session listing", () => {
  // The daemon's agent-manager only queries providers whose
  // capabilities.supportsSessionListing is true. Without it, ACP providers
  // (Kimi and other custom ACP agents, Copilot) are skipped and import shows
  // nothing even though listImportableSessions is implemented.
  test("generic ACP clients (e.g. Kimi) report supportsSessionListing", () => {
    const client = new GenericACPAgentClient({
      logger: createTestLogger(),
      command: ["kimi", "acp"],
    });
    expect(client.capabilities.supportsSessionListing).toBe(true);
  });

  test("Copilot ACP client reports supportsSessionListing", () => {
    const client = new CopilotACPAgentClient({ logger: createTestLogger() });
    expect(client.capabilities.supportsSessionListing).toBe(true);
  });

  test("generic ACP clients report that import previews are not cheap", () => {
    // The import list only skips previews for providers that say so, so this
    // flag is what keeps a big minimax-code session from being restored into
    // memory on every keystroke in the import search.
    const client = new GenericACPAgentClient({
      logger: createTestLogger(),
      command: ["minimax-code", "acp"],
    });
    expect(client.capabilities.importSessionPreviewsAreCheap).toBe(false);
  });

  test("Copilot ACP client reports that import previews are not cheap", () => {
    // Copilot replaces the ACP capabilities wholesale, so without the flag it
    // would keep loading every session to read its first prompt.
    const client = new CopilotACPAgentClient({ logger: createTestLogger() });
    expect(client.capabilities.importSessionPreviewsAreCheap).toBe(false);
  });

  test("asks the provider for its extension notifications", () => {
    // A provider that runs its own sub-agents sends their tree only when the
    // client asked for extension notifications in the handshake. Without the
    // request the background work of a session stays invisible.
    const client = new GenericACPAgentClient({
      logger: createTestLogger(),
      command: ["minimax-code", "acp"],
    });
    const meta = asInternals<{ clientCapabilityMeta: Record<string, unknown> }>(
      client,
    ).clientCapabilityMeta;
    expect(meta).toMatchObject({
      "minimax-code/extensions": { version: 1, notifications: true },
    });
  });
  test("does not offer the MiniMax Code extensions to another ACP provider", () => {
    // The extension request is a vendor handshake. Another ACP provider has
    // no business receiving it, and a client that carries it announces
    // capabilities the provider does not know.
    const client = new GenericACPAgentClient({
      logger: createTestLogger(),
      command: [process.execPath, "fake-acp-agent.mjs"],
      providerId: "configured-acp",
    });
    const meta = asInternals<{ clientCapabilityMeta?: Record<string, unknown> }>(
      client,
    ).clientCapabilityMeta;
    expect(meta).toBeUndefined();
  });
});

describe("transformPiModels", () => {
  test("keeps slash-free labels unchanged", () => {
    expect(
      transformPiModels([
        {
          provider: "pi",
          id: "gpt-4.1-mini",
          label: "GPT 4.1 Mini",
          description: "Fast",
        },
      ]),
    ).toEqual([
      {
        provider: "pi",
        id: "gpt-4.1-mini",
        label: "GPT 4.1 Mini",
        description: "Fast",
      },
    ]);
  });

  test("uses the last path segment as label and preserves existing descriptions", () => {
    expect(
      transformPiModels([
        {
          provider: "pi",
          id: "openrouter/openai/gpt-4.1-mini",
          label: "openrouter/openai/gpt-4.1-mini",
          description: undefined,
        },
        {
          provider: "pi",
          id: "anthropic/claude-sonnet-4",
          label: "anthropic/claude-sonnet-4",
          description: "Balanced",
        },
      ]),
    ).toEqual([
      {
        provider: "pi",
        id: "openrouter/openai/gpt-4.1-mini",
        label: "gpt-4.1-mini",
        description: "openrouter/openai/gpt-4.1-mini",
      },
      {
        provider: "pi",
        id: "anthropic/claude-sonnet-4",
        label: "claude-sonnet-4",
        description: "Balanced",
      },
    ]);
  });
});

describe("ACPAgentSession slash commands", () => {
  test("returns immediately for ACP sessions that do not wait for async command discovery", async () => {
    const session = new ACPAgentSession(
      {
        provider: "claude-acp",
        cwd: "/tmp/paseo-acp-test",
      },
      {
        provider: "claude-acp",
        logger: createTestLogger(),
        defaultCommand: ["claude", "--acp"],
        defaultModes: [],
        capabilities: {
          supportsStreaming: true,
          supportsSessionPersistence: true,
          supportsDynamicModes: true,
          supportsMcpServers: true,
          supportsReasoningStream: true,
          supportsToolInvocations: true,
        },
        waitForInitialCommands: false,
      },
    );

    await expect(session.listCommands()).resolves.toEqual([]);
  });

  test("waits for async available_commands_update when enabled", async () => {
    const session = new ACPAgentSession(
      {
        provider: "claude-acp",
        cwd: "/tmp/paseo-acp-test",
      },
      {
        provider: "claude-acp",
        logger: createTestLogger(),
        defaultCommand: ["claude", "--acp"],
        defaultModes: [],
        capabilities: {
          supportsStreaming: true,
          supportsSessionPersistence: true,
          supportsDynamicModes: true,
          supportsMcpServers: true,
          supportsReasoningStream: true,
          supportsToolInvocations: true,
        },
        waitForInitialCommands: true,
        initialCommandsWaitTimeoutMs: 1500,
      },
    );

    const listCommandsPromise = session.listCommands();

    asInternals<ACPSessionInternals>(session).translateSessionUpdate({
      sessionUpdate: "available_commands_update",
      availableCommands: [
        {
          name: "research_codebase",
          description: "Search the workspace for relevant files",
        },
        {
          name: "create_plan",
          description: "Draft a plan for the requested work",
        },
      ],
    });

    expect(await listCommandsPromise).toEqual([
      {
        name: "research_codebase",
        description: "Search the workspace for relevant files",
        argumentHint: "",
        kind: "command",
      },
      {
        name: "create_plan",
        description: "Draft a plan for the requested work",
        argumentHint: "",
        kind: "command",
      },
    ]);

    expect(await session.listCommands()).toEqual([
      {
        name: "research_codebase",
        description: "Search the workspace for relevant files",
        argumentHint: "",
        kind: "command",
      },
      {
        name: "create_plan",
        description: "Draft a plan for the requested work",
        argumentHint: "",
        kind: "command",
      },
    ]);
  });
});

describe("ACPAgentSession", () => {
  test("drops MCP servers from ACP requests when the provider does not support MCP", () => {
    const session = new ACPAgentSession(
      {
        provider: "no-mcp-acp",
        cwd: "/tmp/paseo-acp-test",
        mcpServers: {
          paseo: {
            type: "http",
            url: "http://127.0.0.1:6767/mcp/agents?callerAgentId=agent-1",
          },
        },
      },
      {
        provider: "no-mcp-acp",
        logger: createTestLogger(),
        defaultCommand: ["no-mcp-acp", "serve"],
        defaultModes: [],
        capabilities: {
          supportsStreaming: true,
          supportsSessionPersistence: true,
          supportsDynamicModes: true,
          supportsMcpServers: false,
          supportsReasoningStream: true,
          supportsToolInvocations: true,
        },
      },
    );

    expect(asInternals<ACPSessionInternals>(session).acpMcpServers()).toEqual([]);
  });

  test("summarizes JSON-RPC error details without stringifying objects", () => {
    const summary = summarizeACPRequestError(
      new RequestError(-32603, "Internal error", {
        details: "Droid process exited unexpectedly (exit code 1)",
      }),
    );

    expect(summary).toMatchObject({
      message: "Internal error: Droid process exited unexpectedly (exit code 1)",
      code: "-32603",
    });
    expect(summary.message).not.toContain("[object Object]");
    expect(summary.diagnostic).toContain("Droid process exited unexpectedly");
  });

  test("accepts ACP extension notifications without failing the JSON-RPC connection", async () => {
    const logger = createTestLogger();
    const trace = vi.spyOn(logger, "trace");
    const session = createSessionWithConfig({ provider: "kiro" }, logger);

    await expect(
      session.extNotification("_kiro.dev/session/initialized", {
        sessionId: "session-1",
      }),
    ).resolves.toBeUndefined();
    expect(trace).toHaveBeenCalledWith(
      expect.objectContaining({
        provider: "kiro",
        method: "_kiro.dev/session/initialized",
        sessionId: "session-1",
      }),
      "provider.acp.extension_notification",
    );
  });

  test("maps the Kiro _kiro.dev/commands/available notification into slash commands and skills", async () => {
    const session = createKiroSession({
      waitForInitialCommands: true,
      initialCommandsWaitTimeoutMs: 1500,
    });
    asInternals<ACPSessionInternals>(session).sessionId = "session-1";

    const listCommandsPromise = session.listCommands();

    await session.extNotification("_kiro.dev/commands/available", {
      sessionId: "session-1",
      commands: [
        {
          name: "/agent",
          description: "Select or list available agents",
          meta: { inputType: "selection", hint: "swap <name>" },
        },
      ],
      prompts: [
        {
          name: "agent-sync-doctor",
          description: "Hand off Claude or Codex state across Macs",
          arguments: [],
          serverName: "skill:config",
        },
      ],
      // Tools are not slash commands and must be ignored.
      tools: [{ name: "code", description: "Code intelligence", source: "built-in" }],
    });

    expect(await listCommandsPromise).toEqual([
      {
        name: "agent",
        description: "Select or list available agents",
        argumentHint: "swap <name>",
        kind: "command",
      },
      {
        name: "agent-sync-doctor",
        description: "Hand off Claude or Codex state across Macs",
        argumentHint: "",
        kind: "skill",
      },
    ]);
  });

  test("ignores Kiro _kiro.dev/commands/available for a different session", async () => {
    const session = createKiroSession();
    asInternals<ACPSessionInternals>(session).sessionId = "session-1";

    await session.extNotification("_kiro.dev/commands/available", {
      sessionId: "other-session",
      commands: [{ name: "/agent", description: "Select or list available agents" }],
      prompts: [],
    });

    expect(await session.listCommands()).toEqual([]);
  });

  test("settles listCommands() immediately on an empty Kiro commands batch", async () => {
    // A long timeout means a resolution can only come from settleCommandsReady()
    // firing — not from the wait timer — so this test would hang if the empty
    // batch failed to unblock listCommands() (the P1 regression).
    const session = createKiroSession({
      waitForInitialCommands: true,
      initialCommandsWaitTimeoutMs: 60_000,
    });
    asInternals<ACPSessionInternals>(session).sessionId = "session-1";

    const listCommandsPromise = session.listCommands();

    await session.extNotification("_kiro.dev/commands/available", {
      sessionId: "session-1",
      commands: [],
      prompts: [],
    });

    expect(await listCommandsPromise).toEqual([]);
  });

  test("emits assistant and reasoning chunks as deltas while user chunks stay accumulated", async () => {
    const session = createSession();
    const events: Array<{
      type: string;
      item?: { type: string; text?: string; messageId?: string };
    }> = [];
    asInternals<ACPSessionInternals>(session).sessionId = "session-1";

    session.subscribe((event) => {
      events.push(
        event as { type: string; item?: { type: string; text?: string; messageId?: string } },
      );
    });

    await session.sessionUpdate({
      sessionId: "session-1",
      update: {
        sessionUpdate: "agent_message_chunk",
        messageId: "assistant-1",
        content: { type: "text", text: "Hey!" },
      } as SessionUpdate,
    });
    await session.sessionUpdate({
      sessionId: "session-1",
      update: {
        sessionUpdate: "agent_message_chunk",
        messageId: "assistant-1",
        content: { type: "text", text: " How are you?" },
      } as SessionUpdate,
    });
    await session.sessionUpdate({
      sessionId: "session-1",
      update: {
        sessionUpdate: "agent_thought_chunk",
        messageId: "thought-1",
        content: { type: "text", text: "Thinking" },
      } as SessionUpdate,
    });
    await session.sessionUpdate({
      sessionId: "session-1",
      update: {
        sessionUpdate: "agent_thought_chunk",
        messageId: "thought-1",
        content: { type: "text", text: " more" },
      } as SessionUpdate,
    });
    await session.sessionUpdate({
      sessionId: "session-1",
      update: {
        sessionUpdate: "user_message_chunk",
        messageId: "user-1",
        content: { type: "text", text: "hel" },
      } as SessionUpdate,
    });
    await session.sessionUpdate({
      sessionId: "session-1",
      update: {
        sessionUpdate: "user_message_chunk",
        messageId: "user-1",
        content: { type: "text", text: "lo" },
      } as SessionUpdate,
    });
    await session.sessionUpdate({
      sessionId: "session-1",
      update: {
        sessionUpdate: "agent_message_chunk",
        messageId: "assistant-2",
        content: { type: "text", text: "" },
      } as SessionUpdate,
    });

    const timeline = events
      .filter((event) => event.type === "timeline")
      .map((event) => event.item)
      .filter(Boolean);

    expect(timeline).toEqual([
      { type: "assistant_message", text: "Hey!", messageId: "assistant-1" },
      { type: "assistant_message", text: " How are you?", messageId: "assistant-1" },
      { type: "reasoning", text: "Thinking" },
      { type: "reasoning", text: " more" },
      { type: "user_message", text: "hello", messageId: "user-1" },
    ]);
  });

  test("assigns one fallback ID per contiguous assistant message", async () => {
    const session = createSession();
    const assistantMessages: Array<{ text: string; messageId?: string }> = [];
    asInternals<ACPSessionInternals>(session).sessionId = "session-1";

    session.subscribe((event) => {
      if (event.type === "timeline" && event.item.type === "assistant_message") {
        assistantMessages.push(event.item);
      }
    });

    for (const text of ["First", " message"]) {
      await session.sessionUpdate({
        sessionId: "session-1",
        update: {
          sessionUpdate: "agent_message_chunk",
          content: { type: "text", text },
        } as SessionUpdate,
      });
    }
    await session.sessionUpdate({
      sessionId: "session-1",
      update: {
        sessionUpdate: "agent_thought_chunk",
        content: { type: "text", text: "Next response" },
      } as SessionUpdate,
    });
    await session.sessionUpdate({
      sessionId: "session-1",
      update: {
        sessionUpdate: "agent_message_chunk",
        content: { type: "text", text: "Second message" },
      } as SessionUpdate,
    });

    expect(assistantMessages).toHaveLength(3);
    expect(assistantMessages[0].messageId).toEqual(expect.any(String));
    expect(assistantMessages[1].messageId).toBe(assistantMessages[0].messageId);
    expect(assistantMessages[2].messageId).toEqual(expect.any(String));
    expect(assistantMessages[2].messageId).not.toBe(assistantMessages[0].messageId);
  });

  test("keeps ACP configuration notifications outside the turn lifecycle", async () => {
    const session = createSession();
    asInternals<ACPSessionInternals>(session).sessionId = "session-1";

    const events: AgentStreamEvent[] = [];
    session.subscribe((event) => {
      events.push(event);
    });

    await session.sessionUpdate({
      sessionId: "session-1",
      update: {
        sessionUpdate: "config_option_update",
        configOptions: [
          selectConfigOption("mode", ["plan", "yolo"], "yolo"),
          selectConfigOption("model", ["kimi-code/kimi-for-coding"]),
          selectConfigOption("thought_level", ["off", "on"], "on"),
        ],
      } as SessionUpdate,
    });

    expect(events.map((event) => event.type)).toEqual([
      "thread_started",
      "mode_changed",
      "model_changed",
      "thinking_option_changed",
    ]);
  });

  test("forwards out-of-prompt ACP content without inventing a turn", async () => {
    const session = createSession();
    asInternals<ACPSessionInternals>(session).sessionId = "session-1";

    const events: AgentStreamEvent[] = [];
    session.subscribe((event) => {
      events.push(event);
    });

    await session.sessionUpdate({
      sessionId: "session-1",
      update: {
        sessionUpdate: "agent_message_chunk",
        messageId: "unscoped-message",
        content: { type: "text", text: "Unscoped ACP update" },
      } as SessionUpdate,
    });

    expect(events).toEqual([
      {
        type: "thread_started",
        provider: "claude-acp",
        sessionId: "session-1",
      },
      {
        type: "timeline",
        provider: "claude-acp",
        item: {
          type: "assistant_message",
          text: "Unscoped ACP update",
          messageId: "unscoped-message",
        },
      },
    ]);
  });

  test("retries the prompt when the provider still owns the previous turn", async () => {
    // A provider that just answered a cancelled prompt frees the session a
    // moment later and refuses the next prompt in that window.
    const session = createSession();
    const turnIdDuringRetry: (string | null)[] = [];
    const prompt = vi
      .fn()
      .mockRejectedValueOnce(
        new Error(
          "Internal error: MiniMax Code Runtime failed: Session already has an active Turn. Use queue send to deliver the message after it.",
        ),
      )
      .mockImplementation(async () => {
        turnIdDuringRetry.push(asInternals<ACPSessionInternals>(session).activeForegroundTurnId);
        return { stopReason: "end_turn" };
      });

    asInternals<ACPSessionInternals>(session).sessionId = "session-1";
    asInternals<ACPSessionInternals>(session).connection = { prompt };

    const { turnId } = await session.startTurn("hello");
    await vi.waitFor(() => expect(prompt).toHaveBeenCalledTimes(2));

    expect(prompt).toHaveBeenCalledTimes(2);
    // The retried prompt carries the same message, so the run stays one run.
    expect(prompt.mock.calls[1]?.[0]).toEqual(prompt.mock.calls[0]?.[0]);
    // The foreground turn stays owned by the same run across the retry.
    expect(turnIdDuringRetry).toEqual([turnId]);
  });

  test("queues the message when the provider is busy and offers a queue", async () => {
    // The provider refuses the prompt because it still owns the previous turn.
    // Its own error points at its queue, so the message goes there instead of
    // being dropped with an error.
    const session = createSession();
    const events: AgentStreamEvent[] = [];
    session.subscribe((event) => {
      events.push(event);
    });
    const prompt = vi
      .fn()
      .mockRejectedValue(
        new Error(
          "Internal error: MiniMax Code Runtime failed: Session already has an active Turn. Use queue send to deliver the message after it.",
        ),
      );
    const extMethod = vi.fn().mockResolvedValue({ status: "queued", itemId: "item-1" });

    asInternals<ACPSessionInternals>(session).sessionId = "session-1";
    asInternals<ACPSessionInternals>(session).connection = { prompt, extMethod };
    asInternals<ACPSessionInternals>(session).agentMeta = {
      "minimax-code/extensions": { methods: ["mcode/session/queue/enqueue"] },
    };

    await session.startTurn("hello");
    await vi.waitFor(() => expect(extMethod).toHaveBeenCalled());

    expect(extMethod).toHaveBeenCalledWith("mcode/session/queue/enqueue", {
      sessionId: "session-1",
      text: "hello",
    });
    // The turn ends as queued, not as a failure: the message is not lost.
    expect(events.some((event) => event.type === "turn_failed")).toBe(false);
    expect(events.some((event) => event.type === "turn_queued")).toBe(true);
    expect(
      events.some((event) => event.type === "timeline" && event.item.type === "notification"),
    ).toBe(true);
  });

  test("queues the message when the provider reports an active prompt", async () => {
    // A provider that tracks its own prompt refuses the second one with its
    // own wording instead of the runtime's busy message.
    const session = createSession();
    const events: AgentStreamEvent[] = [];
    session.subscribe((event) => {
      events.push(event);
    });
    const prompt = vi.fn().mockRejectedValue({
      code: -32602,
      message: "Invalid params: A prompt is already active for this session.",
    });
    const extMethod = vi.fn().mockResolvedValue({ status: "queued", itemId: "item-1" });

    asInternals<ACPSessionInternals>(session).sessionId = "session-1";
    asInternals<ACPSessionInternals>(session).connection = { prompt, extMethod };
    asInternals<ACPSessionInternals>(session).agentMeta = {
      "minimax-code/extensions": { methods: ["mcode/session/queue/enqueue"] },
    };

    await session.startTurn("hello");
    await vi.waitFor(() => expect(extMethod).toHaveBeenCalled());

    expect(extMethod).toHaveBeenCalledWith("mcode/session/queue/enqueue", {
      sessionId: "session-1",
      text: "hello",
    });
    expect(events.some((event) => event.type === "turn_failed")).toBe(false);
    expect(events.some((event) => event.type === "turn_queued")).toBe(true);
  });

  test("does not report a queued turn when the provider answer carries no item id", async () => {
    // The provider's status alone is not delivery: without an item id the message
    // cannot be taken back, and no later stop could find it. Saying "queued" would
    // promise a run that nothing on our side can point at.
    const session = createSession();
    const events: AgentStreamEvent[] = [];
    session.subscribe((event) => events.push(event));
    const prompt = vi
      .fn()
      .mockRejectedValue(
        new Error(
          "Internal error: MiniMax Code Runtime failed: Session already has an active Turn. Use queue send to deliver the message after it.",
        ),
      );
    const extMethod = vi.fn(async (method: string) =>
      method === "mcode/session/queue/enqueue" ? { status: "queued" } : {},
    );

    asInternals<ACPSessionInternals>(session).sessionId = "session-1";
    asInternals<ACPSessionInternals>(session).connection = { prompt, extMethod };
    asInternals<ACPSessionInternals>(session).agentMeta = {
      "minimax-code/extensions": { methods: ["mcode/session/queue/enqueue"] },
    };

    const hasTurnFailed = (): boolean => events.some((event) => event.type === "turn_failed");
    await session.startTurn("hello");
    await vi.waitFor(() => expect(hasTurnFailed()).toBe(true));

    expect(events.some((event) => event.type === "turn_queued")).toBe(false);
    const failed = events.find(
      (event): event is Extract<AgentStreamEvent, { type: "turn_failed" }> =>
        event.type === "turn_failed",
    );
    expect(failed?.code).toBe("queue_unconfirmed");
    expect(failed?.error).toMatch(/may still run it in a later turn/);
    const notifications = events.flatMap((event) =>
      event.type === "timeline" && event.item.type === "notification" ? [event.item] : [],
    );
    expect(
      notifications.some(
        (item) => item.level === "warning" && /did not confirm/.test(item.message),
      ),
    ).toBe(true);
    // One request only: the provider does not deduplicate clientRequestId.
    expect(
      extMethod.mock.calls.filter(([method]) => method === "mcode/session/queue/enqueue"),
    ).toHaveLength(1);
  });

  test("reports the provider queue when the provider advertises one", async () => {
    // A queued message has no answer in the session stream, so the list is the
    // only place its position can be seen - including after a reconnect, when the
    // provider has already sent (and stopped replaying) its queue notifications.
    const session = createSession();
    const extMethod = vi.fn(async (method: string) => {
      if (method !== "mcode/session/queue/list") {
        throw new Error(`unexpected method ${method}`);
      }
      return {
        items: [
          {
            itemId: "item-1",
            sessionId: "session-1",
            status: "queued",
            content: "second thought",
            createdAt: 1_700_000_000_000,
          },
        ],
      };
    });

    asInternals<ACPSessionInternals>(session).sessionId = "session-1";
    asInternals<ACPSessionInternals>(session).connection = { extMethod };
    asInternals<ACPSessionInternals>(session).agentMeta = {
      "minimax-code/extensions": { methods: ["mcode/session/queue/list"] },
    };

    await expect(session.getProviderQueue()).resolves.toEqual([
      {
        itemId: "item-1",
        text: "second thought",
        status: "queued",
        createdAt: new Date(1_700_000_000_000).toISOString(),
      },
    ]);
    expect(extMethod).toHaveBeenCalledWith("mcode/session/queue/list", { sessionId: "session-1" });
  });

  test("does not ask a provider that never advertised a queue list", async () => {
    // Guessing would turn a missing feature into an error the user cannot act on.
    const session = createSession();
    const extMethod = vi.fn();
    asInternals<ACPSessionInternals>(session).sessionId = "session-1";
    asInternals<ACPSessionInternals>(session).connection = { extMethod };
    asInternals<ACPSessionInternals>(session).agentMeta = {
      "minimax-code/extensions": { methods: ["mcode/session/queue/enqueue"] },
    };

    await expect(session.getProviderQueue()).resolves.toEqual([]);
    expect(extMethod).not.toHaveBeenCalled();
  });

  test("reports a message leaving the provider queue as a result it never gave", async () => {
    // Measured against MiniMax Code 0.7.3: a queued message produces no
    // session/update, so its answer never reaches the timeline. The only honest
    // report is that the position left the queue and nothing came back.
    const session = createSession();
    const events: AgentStreamEvent[] = [];
    session.subscribe((event) => events.push(event));
    asInternals<ACPSessionInternals>(session).sessionId = "session-1";

    const notifications = (): string[] =>
      events.flatMap((event) =>
        event.type === "timeline" && event.item.type === "notification" ? [event.item.message] : [],
      );

    await session.extNotification("mcode/session/queue_update", {
      sessionId: "session-1",
      items: [{ itemId: "item-1", status: "queued", content: "second thought", createdAt: 1 }],
    });
    // Seeing a position for the first time is not a departure.
    expect(notifications()).toEqual([]);

    await session.extNotification("mcode/session/queue_update", {
      sessionId: "session-1",
      items: [],
    });

    expect(notifications().some((message) => /left the provider.s queue/.test(message))).toBe(true);
  });

  test("ignores a queue notification that belongs to another session", async () => {
    // The provider may publish the queue of another session on the same client.
    const session = createSession();
    const events: AgentStreamEvent[] = [];
    session.subscribe((event) => events.push(event));
    asInternals<ACPSessionInternals>(session).sessionId = "session-1";

    await session.extNotification("mcode/session/queue_update", {
      sessionId: "session-2",
      items: [{ itemId: "item-9", status: "queued", content: "elsewhere", createdAt: 1 }],
    });
    await session.extNotification("mcode/session/queue_update", {
      sessionId: "session-2",
      items: [],
    });

    expect(events.some((event) => event.type === "timeline")).toBe(false);
  });

  test("reports the queue acceptance as unknown when the provider never answers", async () => {
    // An enqueue that fails on the way back may still have landed. Repeating it could
    // duplicate the message and calling it queued would invent a run, so the turn says
    // exactly what is known.
    const session = createSession();
    const events: AgentStreamEvent[] = [];
    session.subscribe((event) => events.push(event));
    const prompt = vi
      .fn()
      .mockRejectedValue(
        new Error(
          "Internal error: MiniMax Code Runtime failed: Session already has an active Turn. Use queue send to deliver the message after it.",
        ),
      );
    const extMethod = vi.fn(async () => {
      throw new Error("ACP extension request timed out");
    });

    asInternals<ACPSessionInternals>(session).sessionId = "session-1";
    asInternals<ACPSessionInternals>(session).connection = { prompt, extMethod };
    asInternals<ACPSessionInternals>(session).agentMeta = {
      "minimax-code/extensions": { methods: ["mcode/session/queue/enqueue"] },
    };

    const hasTurnFailed = (): boolean => events.some((event) => event.type === "turn_failed");
    await session.startTurn("hello");
    await vi.waitFor(() => expect(hasTurnFailed()).toBe(true));

    expect(events.some((event) => event.type === "turn_queued")).toBe(false);
    const failed = events.find(
      (event): event is Extract<AgentStreamEvent, { type: "turn_failed" }> =>
        event.type === "turn_failed",
    );
    expect(failed?.code).toBe("queue_unconfirmed");
    expect(extMethod).toHaveBeenCalledTimes(1);
  });

  test("withdraws a message that lands in the queue after the stop", async () => {
    // The stop arrives while queue/enqueue is still in flight, so there is no item id to
    // delete yet. The answer has to take the item back itself: otherwise a message the
    // user stopped waits in the provider's queue with nobody able to find it.
    const session = createSession();
    const events: AgentStreamEvent[] = [];
    session.subscribe((event) => events.push(event));
    const prompt = vi
      .fn()
      .mockRejectedValue(
        new Error(
          "Internal error: MiniMax Code Runtime failed: Session already has an active Turn. Use queue send to deliver the message after it.",
        ),
      );
    const enqueueStarted = Promise.withResolvers<void>();
    const enqueueAnswer = Promise.withResolvers<unknown>();
    const extMethod = vi.fn(async (method: string) => {
      if (method === "mcode/session/queue/enqueue") {
        enqueueStarted.resolve();
        return enqueueAnswer.promise;
      }
      return { item: { itemId: "item-1", status: "cancelled" } };
    });
    const cancel = vi.fn().mockResolvedValue(undefined);

    asInternals<ACPSessionInternals>(session).sessionId = "session-1";
    asInternals<ACPSessionInternals>(session).connection = { prompt, extMethod, cancel };
    asInternals<ACPSessionInternals>(session).agentMeta = {
      "minimax-code/extensions": {
        methods: ["mcode/session/queue/enqueue", "mcode/session/queue/delete"],
      },
    };

    await session.startTurn("hello");
    await enqueueStarted.promise;

    const deleteRequests = (): number =>
      extMethod.mock.calls.filter(([method]) => method === "mcode/session/queue/delete").length;
    // The stop finds nothing queued yet: the item only exists once the provider answers.
    const stopping = session.interrupt();
    await vi.waitFor(() => expect(deleteRequests()).toBe(0));
    enqueueAnswer.resolve({ status: "queued", itemId: "item-1" });
    await stopping;

    expect(extMethod).toHaveBeenCalledWith("mcode/session/queue/delete", {
      sessionId: "session-1",
      itemId: "item-1",
    });
    // The turn is not reported as queued: its message no longer exists.
    expect(events.some((event) => event.type === "turn_queued")).toBe(false);
    expect(events.some((event) => event.type === "turn_canceled")).toBe(true);
    const notifications = events.flatMap((event) =>
      event.type === "timeline" && event.item.type === "notification" ? [event.item] : [],
    );
    expect(notifications.some((item) => /withdrawn/.test(item.message))).toBe(true);
    expect(notifications.some((item) => item.level === "warning")).toBe(false);
  });

  test("keeps a delegation snapshot that arrived before the manager subscribed", async () => {
    // A session reads its delegation tree while it initializes, and the agent
    // manager subscribes only afterwards. A child that already finished never
    // publishes again, so a dropped first snapshot hides the delegated work of
    // a reopened session for good.
    const session = createSession();
    asInternals<ACPSessionInternals>(session).sessionId = "session-1";

    await session.extNotification("mcode/session/delegation_update", {
      sessionId: "session-1",
      snapshot: {
        schemaVersion: 1,
        rootSessionId: "session-1",
        members: [
          {
            sessionId: "child-1",
            parentSessionId: "session-1",
            agentName: "explore",
            task: "List files",
            status: "completed",
            createdAtMs: 1_700_000_000_000,
            updatedAtMs: 1_700_000_001_000,
          },
          {
            sessionId: "child-2",
            parentSessionId: "session-1",
            agentName: "worker",
            task: "Run the tests",
            status: "completed",
            createdAtMs: 1_700_000_000_000,
            updatedAtMs: 1_700_000_002_000,
          },
        ],
      },
    });

    const events: AgentStreamEvent[] = [];
    session.subscribe((event) => events.push(event));

    const publishedIds = events.map((event) =>
      event.type === "provider_subagent" ? event.event.id : null,
    );
    expect(publishedIds.filter((id) => id !== null)).toEqual(["child-1", "child-2"]);

    // The rows belong to the timeline once: a later subscriber does not get
    // them a second time.
    const laterEvents: AgentStreamEvent[] = [];
    session.subscribe((event) => laterEvents.push(event));
    expect(laterEvents.some((event) => event.type === "provider_subagent")).toBe(false);
  });

  test("publishes the provider's delegation tree as sub-agent rows", async () => {
    // A provider that runs its own sub-agents keeps them out of the session
    // list. The delegation tree is the only place they exist, and without it
    // the background work of a session stays invisible.
    const session = createSession();
    const events: AgentStreamEvent[] = [];
    session.subscribe((event) => {
      events.push(event);
    });
    asInternals<ACPSessionInternals>(session).sessionId = "session-1";

    await session.extNotification("mcode/session/delegation_update", {
      sessionId: "session-1",
      snapshot: {
        schemaVersion: 1,
        rootSessionId: "session-1",
        members: [
          {
            sessionId: "child-1",
            parentSessionId: "session-1",
            agentName: "explore",
            task: "List files in the project",
            status: "running",
            createdAtMs: 1_700_000_000_000,
            updatedAtMs: 1_700_000_001_000,
          },
        ],
      },
    });

    expect(events).toEqual([
      {
        type: "provider_subagent",
        provider: "claude-acp",
        event: {
          type: "upsert",
          id: "child-1",
          title: "List files in the project",
          description: "explore",
          status: "running",
          // The parent is the session itself, not another sub-agent.
          parentSubagentId: null,
          timestamp: new Date(1_700_000_001_000).toISOString(),
        },
      },
    ]);
  });

  test("keeps a sub-agent that has not started yet out of the finished state", async () => {
    // Some providers project a child that has not started yet as finished, with
    // the same timestamp for its creation and its last update. Reporting that
    // as finished claims the work is done before it began.
    const session = createSession();
    const events: AgentStreamEvent[] = [];
    session.subscribe((event) => {
      events.push(event);
    });
    asInternals<ACPSessionInternals>(session).sessionId = "session-1";

    const member = (status: string, updatedAtMs: number) => ({
      schemaVersion: 1,
      rootSessionId: "session-1",
      members: [
        {
          sessionId: "child-1",
          parentSessionId: "session-1",
          agentName: "explore",
          task: "List files in the project",
          status,
          createdAtMs: 1_700_000_000_000,
          updatedAtMs,
        },
      ],
    });

    await session.extNotification("mcode/session/delegation_update", {
      sessionId: "session-1",
      snapshot: member("completed", 1_700_000_000_000),
    });
    await session.extNotification("mcode/session/delegation_update", {
      sessionId: "session-1",
      snapshot: member("running", 1_700_000_000_500),
    });
    await session.extNotification("mcode/session/delegation_update", {
      sessionId: "session-1",
      snapshot: member("completed", 1_700_000_009_000),
    });

    expect(
      events.map((event) => (event.type === "provider_subagent" ? event.event.status : undefined)),
    ).toEqual(["running", "running", "completed"]);
  });

  test("does not republish an unchanged delegation tree", async () => {
    const session = createSession();
    const events: AgentStreamEvent[] = [];
    session.subscribe((event) => {
      events.push(event);
    });
    asInternals<ACPSessionInternals>(session).sessionId = "session-1";
    const snapshot = {
      schemaVersion: 1,
      rootSessionId: "session-1",
      members: [
        {
          sessionId: "child-1",
          parentSessionId: "session-1",
          agentName: "explore",
          task: "List files",
          status: "completed",
          updatedAtMs: 1_700_000_001_000,
        },
      ],
    };

    await session.extNotification("mcode/session/delegation_update", {
      sessionId: "session-1",
      snapshot,
    });
    await session.extNotification("mcode/session/delegation_update", {
      sessionId: "session-1",
      snapshot,
    });

    expect(events).toHaveLength(1);
  });

  test("maps a stopped delegation to a cancelled sub-agent", async () => {
    const session = createSession();
    const events: AgentStreamEvent[] = [];
    session.subscribe((event) => {
      events.push(event);
    });
    asInternals<ACPSessionInternals>(session).sessionId = "session-1";

    await session.extNotification("mcode/session/delegation_update", {
      sessionId: "session-1",
      snapshot: {
        schemaVersion: 1,
        rootSessionId: "session-1",
        members: [
          {
            sessionId: "child-1",
            parentSessionId: "session-1",
            agentName: "worker",
            task: "Background sleep",
            status: "stopped",
            updatedAtMs: 1_700_000_001_000,
          },
        ],
      },
    });

    expect(events[0]).toMatchObject({
      type: "provider_subagent",
      event: { type: "upsert", id: "child-1", status: "canceled" },
    });
  });

  test("takes the queued message back when the user stops after queueing it", async () => {
    // The message lives in the provider's queue, not in the turn that queued
    // it, so a Stop that only cancels the running turn still lets it run.
    // The provider removes a queued item on request, so the stop has to ask.
    const session = createSession();
    const events: AgentStreamEvent[] = [];
    session.subscribe((event) => events.push(event));
    const prompt = vi
      .fn()
      .mockRejectedValue(
        new Error(
          "Internal error: MiniMax Code Runtime failed: Session already has an active Turn. Use queue send to deliver the message after it.",
        ),
      );
    const extMethod = vi.fn(async (method: string) => {
      if (method === "mcode/session/queue/enqueue") {
        return { status: "queued", itemId: "item-1" };
      }
      return { item: { itemId: "item-1", status: "cancelled" } };
    });

    asInternals<ACPSessionInternals>(session).sessionId = "session-1";
    asInternals<ACPSessionInternals>(session).connection = { prompt, extMethod };
    asInternals<ACPSessionInternals>(session).agentMeta = {
      "minimax-code/extensions": {
        methods: ["mcode/session/queue/enqueue", "mcode/session/queue/delete"],
      },
    };

    const turnIsQueued = (): boolean => events.some((event) => event.type === "turn_queued");

    await session.startTurn("hello");
    await vi.waitFor(() => expect(turnIsQueued()).toBe(true));

    await session.interrupt();

    expect(extMethod).toHaveBeenCalledWith("mcode/session/queue/delete", {
      sessionId: "session-1",
      itemId: "item-1",
    });
    const notifications = events.flatMap((event) =>
      event.type === "timeline" && event.item.type === "notification" ? [event.item] : [],
    );
    // The message was withdrawn, so nothing warns that it will still run.
    expect(notifications.some((item) => item.level === "warning")).toBe(false);
    expect(notifications.some((item) => /withdrawn/.test(item.message))).toBe(true);
  });

  test("says the queued message stays when the provider already claimed it", async () => {
    // Once the provider takes the item for a turn, the removal is refused. A
    // Stop cannot undo that, and telling the user otherwise is a lie.
    const session = createSession();
    const events: AgentStreamEvent[] = [];
    session.subscribe((event) => events.push(event));
    const prompt = vi
      .fn()
      .mockRejectedValue(
        new Error(
          "Internal error: MiniMax Code Runtime failed: Session already has an active Turn. Use queue send to deliver the message after it.",
        ),
      );
    const extMethod = vi.fn(async (method: string) => {
      if (method === "mcode/session/queue/enqueue") {
        return { status: "queued", itemId: "item-1" };
      }
      throw new Error("Queue item is not editable: session-1/item-1");
    });

    asInternals<ACPSessionInternals>(session).sessionId = "session-1";
    asInternals<ACPSessionInternals>(session).connection = { prompt, extMethod };
    asInternals<ACPSessionInternals>(session).agentMeta = {
      "minimax-code/extensions": {
        methods: ["mcode/session/queue/enqueue", "mcode/session/queue/delete"],
      },
    };

    const turnIsQueued = (): boolean => events.some((event) => event.type === "turn_queued");

    await session.startTurn("hello");
    await vi.waitFor(() => expect(turnIsQueued()).toBe(true));

    await session.interrupt();

    const notifications = events.flatMap((event) =>
      event.type === "timeline" && event.item.type === "notification" ? [event.item] : [],
    );
    const warning = notifications.find((item) => item.level === "warning");
    expect(warning?.message).toMatch(/cannot be withdrawn/);
    // The stop is not reported as a withdrawal that did not happen.
    expect(notifications.some((item) => /will not run/.test(item.message))).toBe(false);
  });

  test("fails the turn when the provider is busy and offers no queue", async () => {
    const session = createSession();
    const events: AgentStreamEvent[] = [];
    session.subscribe((event) => {
      events.push(event);
    });
    const prompt = vi
      .fn()
      .mockRejectedValue(
        new Error(
          "Internal error: MiniMax Code Runtime failed: Session already has an active Turn. Use queue send to deliver the message after it.",
        ),
      );
    const extMethod = vi.fn().mockResolvedValue({});

    asInternals<ACPSessionInternals>(session).sessionId = "session-1";
    asInternals<ACPSessionInternals>(session).connection = { prompt, extMethod };
    asInternals<ACPSessionInternals>(session).agentMeta = {
      "minimax-code/extensions": { methods: ["mcode/session/steer"] },
    };

    const hasTurnFailed = (): boolean => events.some((event) => event.type === "turn_failed");
    await session.startTurn("hello");
    await vi.waitFor(() => expect(hasTurnFailed()).toBe(true));

    // A provider without a queue keeps the old behaviour: the user sees the
    // failure instead of a promise the provider will not keep.
    expect(extMethod).not.toHaveBeenCalled();
    expect(events.some((event) => event.type === "turn_queued")).toBe(false);
  });

  test("closes the turn when the user stops it while the busy retry waits", async () => {
    // The retry pause is a window where the turn has no provider answer yet.
    // A Stop in that window has to end the turn itself: a turn left open keeps
    // the session busy for good, refusing every later prompt.
    const session = createSession();
    const events: AgentStreamEvent[] = [];
    session.subscribe((event) => events.push(event));
    const prompt = vi
      .fn()
      .mockRejectedValueOnce(
        new Error(
          "Internal error: MiniMax Code Runtime failed: Session already has an active Turn. Use queue send to deliver the message after it.",
        ),
      )
      .mockResolvedValue({ stopReason: "end_turn" });
    const cancel = vi.fn().mockResolvedValue(undefined);

    asInternals<ACPSessionInternals>(session).sessionId = "session-1";
    asInternals<ACPSessionInternals>(session).connection = { prompt, cancel };

    const { turnId } = await session.startTurn("hello");
    await vi.waitFor(() => expect(prompt).toHaveBeenCalledTimes(1));

    await session.interrupt().catch(() => undefined);

    const canceledTurnIds = events.flatMap((event) =>
      event.type === "turn_canceled" ? [event.turnId] : [],
    );
    expect(canceledTurnIds).toEqual([turnId]);

    // The stopped session takes the next prompt instead of reporting a
    // foreground turn that no longer exists.
    await expect(session.startTurn("again")).resolves.toEqual({ turnId: expect.any(String) });
    expect(asInternals<ACPSessionInternals>(session).activeForegroundTurnId).toBeNull();
  });

  test("does not deliver the prompt again after the turn was interrupted", async () => {
    const session = createSession();
    const prompt = vi
      .fn()
      .mockRejectedValue(
        new Error(
          "Internal error: MiniMax Code Runtime failed: Session already has an active Turn. Use queue send to deliver the message after it.",
        ),
      );
    const cancel = vi.fn().mockResolvedValue(undefined);

    asInternals<ACPSessionInternals>(session).sessionId = "session-1";
    asInternals<ACPSessionInternals>(session).connection = { prompt, cancel };

    await session.startTurn("hello");
    await vi.waitFor(() => expect(prompt).toHaveBeenCalledTimes(1));

    // The user stops the turn while the busy retry waits for its pause.
    const interruption = session.interrupt();
    await interruption.catch(() => undefined);

    // Give the retry its full pause, then let the interruption settle.
    await new Promise((resolve) => setTimeout(resolve, 600));

    expect(prompt).toHaveBeenCalledTimes(1);
    expect(cancel).toHaveBeenCalledWith({ sessionId: "session-1" });
  });

  test("does not retry a prompt that failed for another reason", async () => {
    const session = createSession();
    const prompt = vi.fn().mockRejectedValue(new Error("Internal error: something else"));

    asInternals<ACPSessionInternals>(session).sessionId = "session-1";
    asInternals<ACPSessionInternals>(session).connection = { prompt };

    await session.startTurn("hello");
    await vi.waitFor(() => expect(prompt).toHaveBeenCalled());

    expect(prompt).toHaveBeenCalledTimes(1);
  });

  test("steers the running turn through the provider extension", async () => {
    const session = createSession();
    const extMethod = vi.fn().mockResolvedValue({ turnId: "native-turn-1", mode: "steered" });
    const prompt = vi.fn(() => new Promise<PromptResponse>(() => {}));

    asInternals<ACPSessionInternals>(session).sessionId = "session-1";
    asInternals<ACPSessionInternals>(session).connection = { prompt, extMethod };
    asInternals<ACPSessionInternals>(session).agentMeta = {
      "minimax-code/extensions": {
        version: 1,
        methods: ["mcode/session/steer"],
      },
    };

    const { turnId } = await session.startTurn("work");
    const result = await session.steerActiveTurn!("answer with one word", {
      expectedTurnId: turnId,
      clientMessageId: "steer-1",
    });

    expect(result).toEqual({ status: "accepted" });
    expect(extMethod).toHaveBeenCalledWith("mcode/session/steer", {
      sessionId: "session-1",
      text: "answer with one word",
      clientRequestId: "steer-1",
    });
  });

  test("reports steering as unavailable when the provider never advertised the extension", async () => {
    const session = createSession();
    const extMethod = vi.fn();
    const prompt = vi.fn(() => new Promise<PromptResponse>(() => {}));

    asInternals<ACPSessionInternals>(session).sessionId = "session-1";
    asInternals<ACPSessionInternals>(session).connection = { prompt, extMethod };
    asInternals<ACPSessionInternals>(session).agentMeta = {
      "minimax-code/extensions": { version: 1, methods: ["mcode/session/queue/enqueue"] },
    };

    const { turnId } = await session.startTurn("work");
    const result = await session.steerActiveTurn!("follow up", { expectedTurnId: turnId });

    expect(result).toEqual({ status: "unavailable" });
    expect(extMethod).not.toHaveBeenCalled();
  });

  test("reports steering as unavailable when the provider does not implement the method", async () => {
    const session = createSession();
    const extMethod = vi
      .fn()
      .mockRejectedValue(Object.assign(new Error("Method not found"), { code: -32601 }));
    const prompt = vi.fn(() => new Promise<PromptResponse>(() => {}));

    asInternals<ACPSessionInternals>(session).sessionId = "session-1";
    asInternals<ACPSessionInternals>(session).connection = { prompt, extMethod };
    asInternals<ACPSessionInternals>(session).agentMeta = {
      "minimax-code/extensions": { version: 1, methods: ["mcode/session/steer"] },
    };

    const { turnId } = await session.startTurn("work");
    const result = await session.steerActiveTurn!("follow up", { expectedTurnId: turnId });

    expect(result).toEqual({ status: "unavailable" });
  });

  test("surfaces an ambiguous steer failure instead of retrying it", async () => {
    const session = createSession();
    const extMethod = vi.fn().mockRejectedValue(new Error("steer transport failed"));
    const prompt = vi.fn(() => new Promise<PromptResponse>(() => {}));

    asInternals<ACPSessionInternals>(session).sessionId = "session-1";
    asInternals<ACPSessionInternals>(session).connection = { prompt, extMethod };
    asInternals<ACPSessionInternals>(session).agentMeta = {
      "minimax-code/extensions": { version: 1, methods: ["mcode/session/steer"] },
    };

    const { turnId } = await session.startTurn("work");

    await expect(session.steerActiveTurn!("follow up", { expectedTurnId: turnId })).rejects.toThrow(
      "steer transport failed",
    );
    expect(extMethod).toHaveBeenCalledOnce();
  });

  test("does not steer a prompt with attachments, because the extension carries text only", async () => {
    const session = createSession();
    const extMethod = vi.fn();
    const prompt = vi.fn(() => new Promise<PromptResponse>(() => {}));

    asInternals<ACPSessionInternals>(session).sessionId = "session-1";
    asInternals<ACPSessionInternals>(session).connection = { prompt, extMethod };
    asInternals<ACPSessionInternals>(session).agentMeta = {
      "minimax-code/extensions": { version: 1, methods: ["mcode/session/steer"] },
    };

    const { turnId } = await session.startTurn("work");
    const result = await session.steerActiveTurn!(
      [
        { type: "text", text: "look at this" },
        { type: "image", data: "AAAA", mimeType: "image/png" },
      ],
      { expectedTurnId: turnId },
    );

    expect(result).toEqual({ status: "unavailable" });
    expect(extMethod).not.toHaveBeenCalled();
  });

  test("rejects a steer that the provider turned into a new turn", async () => {
    const session = createSession();
    const extMethod = vi.fn().mockResolvedValue({ turnId: "native-turn-2", mode: "activated" });
    const prompt = vi.fn(() => new Promise<PromptResponse>(() => {}));

    asInternals<ACPSessionInternals>(session).sessionId = "session-1";
    asInternals<ACPSessionInternals>(session).connection = { prompt, extMethod };
    asInternals<ACPSessionInternals>(session).agentMeta = {
      "minimax-code/extensions": { version: 1, methods: ["mcode/session/steer"] },
    };

    const { turnId } = await session.startTurn("work");

    await expect(session.steerActiveTurn!("follow up", { expectedTurnId: turnId })).rejects.toThrow(
      "activated a new turn",
    );
  });

  test("rejects a steer with an unknown disposition", async () => {
    const session = createSession();
    const extMethod = vi.fn().mockResolvedValue({ turnId: "native-turn-1", mode: "queued" });
    const prompt = vi.fn(() => new Promise<PromptResponse>(() => {}));

    asInternals<ACPSessionInternals>(session).sessionId = "session-1";
    asInternals<ACPSessionInternals>(session).connection = { prompt, extMethod };
    asInternals<ACPSessionInternals>(session).agentMeta = {
      "minimax-code/extensions": { version: 1, methods: ["mcode/session/steer"] },
    };

    const { turnId } = await session.startTurn("work");

    await expect(session.steerActiveTurn!("follow up", { expectedTurnId: turnId })).rejects.toThrow(
      "unknown steer disposition",
    );
  });

  test("does not steer a turn that is no longer the active one", async () => {
    const session = createSession();
    const extMethod = vi.fn();
    const prompt = vi.fn(() => new Promise<PromptResponse>(() => {}));

    asInternals<ACPSessionInternals>(session).sessionId = "session-1";
    asInternals<ACPSessionInternals>(session).connection = { prompt, extMethod };
    asInternals<ACPSessionInternals>(session).agentMeta = {
      "minimax-code/extensions": { version: 1, methods: ["mcode/session/steer"] },
    };

    await session.startTurn("work");
    const result = await session.steerActiveTurn!("follow up", { expectedTurnId: "other-turn" });

    expect(result).toEqual({ status: "unavailable" });
    expect(extMethod).not.toHaveBeenCalled();
  });

  test("startTurn returns before the ACP prompt settles and completes later via subscribers", async () => {
    const session = createSession();
    const events: Array<{ type: string; turnId?: string }> = [];
    let resolvePrompt!: (value: PromptResponse) => void;
    const prompt = vi.fn(
      () =>
        new Promise((resolve) => {
          resolvePrompt = resolve;
        }),
    );

    asInternals<ACPSessionInternals>(session).sessionId = "session-1";
    asInternals<ACPSessionInternals>(session).connection = { prompt };

    session.subscribe((event) => {
      events.push(event as { type: string; turnId?: string });
    });

    const { turnId } = await session.startTurn("hello");

    expect(prompt).toHaveBeenCalledOnce();
    expect(events.find((event) => event.type === "turn_started")).toMatchObject({
      type: "turn_started",
      turnId,
    });
    expect(asInternals<ACPSessionInternals>(session).activeForegroundTurnId).toBe(turnId);

    resolvePrompt({ stopReason: "end_turn", usage: { outputTokens: 3 } });
    await Promise.resolve();
    await Promise.resolve();

    expect(events.find((event) => event.type === "turn_completed")).toMatchObject({
      type: "turn_completed",
      turnId,
    });
    expect(asInternals<ACPSessionInternals>(session).activeForegroundTurnId).toBeNull();
  });

  test("startTurn emits the submitted user message even when ACP does not echo it", async () => {
    const session = createSession();
    const events: AgentStreamEvent[] = [];
    let resolvePrompt!: (value: PromptResponse) => void;
    const prompt = vi.fn(
      () =>
        new Promise<PromptResponse>((resolve) => {
          resolvePrompt = resolve;
        }),
    );

    asInternals<ACPSessionInternals>(session).sessionId = "session-1";
    asInternals<ACPSessionInternals>(session).connection = { prompt };

    session.subscribe((event) => {
      events.push(event);
    });

    const { turnId } = await session.startTurn("hello", {
      clientMessageId: "msg-client-1",
    });

    expect(prompt).toHaveBeenCalledWith({
      sessionId: "session-1",
      messageId: "msg-client-1",
      prompt: [{ type: "text", text: "hello" }],
    });
    expect(
      events
        .filter(
          (event) =>
            event.type === "turn_started" ||
            (event.type === "timeline" && event.item.type === "user_message"),
        )
        .map((event) => event.type),
    ).toEqual(["turn_started", "timeline"]);
    expect(
      events.filter((event) => event.type === "timeline" && event.item.type === "user_message"),
    ).toEqual([
      {
        type: "timeline",
        provider: "claude-acp",
        turnId,
        item: {
          type: "user_message",
          text: "hello",
          messageId: "msg-client-1",
          clientMessageId: "msg-client-1",
        },
      },
    ]);

    resolvePrompt({ stopReason: "end_turn" });
  });

  test("startTurn dedupes ACP user echo chunks for the submitted message", async () => {
    const session = createSession();
    const events: AgentStreamEvent[] = [];
    const prompt = vi.fn(() => new Promise<PromptResponse>(() => {}));

    asInternals<ACPSessionInternals>(session).sessionId = "session-1";
    asInternals<ACPSessionInternals>(session).connection = { prompt };

    session.subscribe((event) => {
      events.push(event);
    });

    await session.startTurn("hello", { clientMessageId: "msg-client-1" });
    await session.sessionUpdate({
      sessionId: "session-1",
      update: {
        sessionUpdate: "user_message_chunk",
        messageId: "msg-client-1",
        content: { type: "text", text: "hello" },
      } as SessionUpdate,
    });

    expect(
      events.filter((event) => event.type === "timeline" && event.item.type === "user_message"),
    ).toHaveLength(1);
  });

  test("startTurn dedupes ACP user echo chunks without message ids for the submitted message", async () => {
    const session = createSession();
    const events: AgentStreamEvent[] = [];
    const prompt = vi.fn(() => new Promise<PromptResponse>(() => {}));

    asInternals<ACPSessionInternals>(session).sessionId = "session-1";
    asInternals<ACPSessionInternals>(session).connection = { prompt };

    session.subscribe((event) => {
      events.push(event);
    });

    await session.startTurn("hello", { clientMessageId: "msg-client-1" });
    await session.sessionUpdate({
      sessionId: "session-1",
      update: {
        sessionUpdate: "user_message_chunk",
        content: { type: "text", text: "hello" },
      } as SessionUpdate,
    });

    expect(
      events.filter((event) => event.type === "timeline" && event.item.type === "user_message"),
    ).toEqual([
      {
        type: "timeline",
        provider: "claude-acp",
        item: {
          type: "user_message",
          text: "hello",
          messageId: "msg-client-1",
          clientMessageId: "msg-client-1",
        },
        turnId: expect.any(String),
      },
    ]);
  });

  test("startTurn dedupes ACP user echo chunks without message ids across turns", async () => {
    const session = createSession();
    const events: AgentStreamEvent[] = [];
    let resolvePrompt!: (value: PromptResponse) => void;
    const prompt = vi.fn(
      () =>
        new Promise<PromptResponse>((resolve) => {
          resolvePrompt = resolve;
        }),
    );

    asInternals<ACPSessionInternals>(session).sessionId = "session-1";
    asInternals<ACPSessionInternals>(session).connection = { prompt };

    session.subscribe((event) => {
      events.push(event);
    });

    await session.startTurn("first", { clientMessageId: "msg-client-1" });
    await session.sessionUpdate({
      sessionId: "session-1",
      update: {
        sessionUpdate: "user_message_chunk",
        content: { type: "text", text: "first" },
      } as SessionUpdate,
    });
    resolvePrompt({ stopReason: "end_turn" });
    await Promise.resolve();
    await Promise.resolve();

    await session.startTurn("second", { clientMessageId: "msg-client-2" });
    await session.sessionUpdate({
      sessionId: "session-1",
      update: {
        sessionUpdate: "user_message_chunk",
        content: { type: "text", text: "second" },
      } as SessionUpdate,
    });

    expect(
      events.filter((event) => event.type === "timeline" && event.item.type === "user_message"),
    ).toEqual([
      {
        type: "timeline",
        provider: "claude-acp",
        item: {
          type: "user_message",
          text: "first",
          messageId: "msg-client-1",
          clientMessageId: "msg-client-1",
        },
        turnId: expect.any(String),
      },
      {
        type: "timeline",
        provider: "claude-acp",
        item: {
          type: "user_message",
          text: "second",
          messageId: "msg-client-2",
          clientMessageId: "msg-client-2",
        },
        turnId: expect.any(String),
      },
    ]);
  });

  test("startTurn dedupes ACP user echo chunks with provider-owned ids for the submitted message", async () => {
    const session = createSession();
    const events: AgentStreamEvent[] = [];
    const prompt = vi.fn(() => new Promise<PromptResponse>(() => {}));

    asInternals<ACPSessionInternals>(session).sessionId = "session-1";
    asInternals<ACPSessionInternals>(session).connection = { prompt };

    session.subscribe((event) => {
      events.push(event);
    });

    await session.startTurn("hello", { clientMessageId: "msg-client-1" });
    await session.sessionUpdate({
      sessionId: "session-1",
      update: {
        sessionUpdate: "user_message_chunk",
        messageId: "msg-provider-1",
        content: { type: "text", text: "hello" },
      } as SessionUpdate,
    });

    expect(
      events.filter((event) => event.type === "timeline" && event.item.type === "user_message"),
    ).toEqual([
      {
        type: "timeline",
        provider: "claude-acp",
        item: {
          type: "user_message",
          text: "hello",
          messageId: "msg-client-1",
          clientMessageId: "msg-client-1",
        },
        turnId: expect.any(String),
      },
    ]);
  });

  test("startTurn dedupes a provider-owned user echo streamed as text and image chunks", async () => {
    const session = createSession();
    const events: AgentStreamEvent[] = [];
    const prompt = vi.fn(() => new Promise<PromptResponse>(() => {}));

    asInternals<ACPSessionInternals>(session).sessionId = "session-1";
    asInternals<ACPSessionInternals>(session).connection = { prompt };

    session.subscribe((event) => {
      events.push(event);
    });

    await session.startTurn(
      [
        { type: "text", text: "hey" },
        { type: "image", data: "AA==", mimeType: "image/png" },
      ],
      { clientMessageId: "msg-client-1" },
    );
    await session.sessionUpdate({
      sessionId: "session-1",
      update: {
        sessionUpdate: "user_message_chunk",
        messageId: "msg-provider-1",
        content: { type: "text", text: "hey" },
      } as SessionUpdate,
    });
    await session.sessionUpdate({
      sessionId: "session-1",
      update: {
        sessionUpdate: "user_message_chunk",
        messageId: "msg-provider-1",
        content: { type: "image", data: "AA==", mimeType: "image/png" },
      } as SessionUpdate,
    });

    expect(
      events.filter((event) => event.type === "timeline" && event.item.type === "user_message"),
    ).toEqual([
      {
        type: "timeline",
        provider: session.provider,
        item: {
          type: "user_message",
          text: "hey",
          messageId: "msg-client-1",
          clientMessageId: "msg-client-1",
        },
        turnId: expect.any(String),
      },
    ]);
  });

  test("startTurn keeps an image-only provider echo when no canonical user message was emitted", async () => {
    const session = createSession();
    const events: AgentStreamEvent[] = [];
    const prompt = vi.fn(() => new Promise<PromptResponse>(() => {}));

    asInternals<ACPSessionInternals>(session).sessionId = "session-1";
    asInternals<ACPSessionInternals>(session).connection = { prompt };

    session.subscribe((event) => {
      events.push(event);
    });

    await session.startTurn([{ type: "image", data: "AA==", mimeType: "image/png" }], {
      clientMessageId: "msg-client-1",
    });
    await session.sessionUpdate({
      sessionId: "session-1",
      update: {
        sessionUpdate: "user_message_chunk",
        content: { type: "image", data: "AA==", mimeType: "image/png" },
      } as SessionUpdate,
    });
    await session.sessionUpdate({
      sessionId: "session-1",
      update: {
        sessionUpdate: "agent_message_chunk",
        messageId: "assistant-1",
        content: { type: "text", text: "I see it" },
      } as SessionUpdate,
    });

    expect(
      events.filter((event) => event.type === "timeline" && event.item.type === "user_message"),
    ).toEqual([
      {
        type: "timeline",
        provider: session.provider,
        item: { type: "user_message", text: "[image]" },
        turnId: expect.any(String),
      },
    ]);
  });

  test("startTurn converts background prompt rejections into turn_failed events", async () => {
    const session = createSession();
    const events: Array<{ type: string; turnId?: string; error?: string }> = [];
    let rejectPrompt!: (error: Error) => void;
    const prompt = vi.fn(
      () =>
        new Promise((_, reject) => {
          rejectPrompt = reject;
        }),
    );

    asInternals<ACPSessionInternals>(session).sessionId = "session-1";
    asInternals<ACPSessionInternals>(session).connection = { prompt };

    session.subscribe((event) => {
      events.push(event as { type: string; turnId?: string; error?: string });
    });

    const { turnId } = await session.startTurn("hello");

    rejectPrompt(new Error("prompt failed"));
    await Promise.resolve();
    await Promise.resolve();

    const turnFailedEvent = events.find((event) => event.type === "turn_failed");
    expect(turnFailedEvent).toMatchObject({
      type: "turn_failed",
      turnId,
      error: "prompt failed",
    });
    expect(asInternals<ACPSessionInternals>(session).activeForegroundTurnId).toBeNull();
  });

  test("flushes an image-only provider echo before a rejected turn finishes", async () => {
    const session = createSession();
    const events: AgentStreamEvent[] = [];
    let rejectPrompt!: (error: Error) => void;
    const prompt = vi.fn(
      () =>
        new Promise<PromptResponse>((_, reject) => {
          rejectPrompt = reject;
        }),
    );

    asInternals<ACPSessionInternals>(session).sessionId = "session-1";
    asInternals<ACPSessionInternals>(session).connection = { prompt };
    session.subscribe((event) => events.push(event));

    const { turnId } = await session.startTurn([
      { type: "image", data: "AA==", mimeType: "image/png" },
    ]);
    await session.sessionUpdate({
      sessionId: "session-1",
      update: {
        sessionUpdate: "user_message_chunk",
        content: { type: "image", data: "AA==", mimeType: "image/png" },
      } as SessionUpdate,
    });

    rejectPrompt(new Error("prompt failed"));
    await Promise.resolve();
    await Promise.resolve();

    expect(
      events.filter((event) => event.type === "timeline" || event.type === "turn_failed"),
    ).toEqual([
      {
        type: "timeline",
        provider: session.provider,
        item: { type: "user_message", text: "[image]" },
        turnId,
      },
      expect.objectContaining({ type: "turn_failed", turnId, error: "prompt failed" }),
    ]);
  });

  test("startTurn preserves JSON-RPC error details from a real ACP prompt response", async () => {
    const session = createSession();
    const clientToAgent = new TransformStream();
    const agentToClient = new TransformStream();
    const upstreamMessage =
      "Authentication failed: Please authenticate to continue. Run `/login` to log in.";
    const upstreamData = {
      cause: "auth_required",
      errorMessage: "Please authenticate to continue. Run `/login` to log in.",
    };
    const agent: Agent = {
      async initialize() {
        return {
          protocolVersion: PROTOCOL_VERSION,
          agentCapabilities: {},
          authMethods: [{ id: "windsurf-api-key", name: "API Key" }],
        };
      },
      async newSession() {
        return { sessionId: "session-1" };
      },
      async prompt() {
        throw new RequestError(-32000, upstreamMessage, upstreamData);
      },
      async authenticate() {},
      async cancel() {},
    };
    const agentConnection = new AgentSideConnection(
      () => agent,
      ndJsonStream(agentToClient.writable, clientToAgent.readable),
    );
    const connection = new ClientSideConnection(
      () => ({
        async requestPermission() {
          return { outcome: { outcome: "cancelled" } };
        },
        async sessionUpdate() {},
      }),
      ndJsonStream(clientToAgent.writable, agentToClient.readable),
    );
    await connection.initialize({
      protocolVersion: PROTOCOL_VERSION,
      clientCapabilities: {},
      clientInfo: { name: "Paseo test", version: "dev" },
    });
    expect(agentConnection.signal.aborted).toBe(false);
    const sessionResponse = await connection.newSession({
      cwd: "/tmp/paseo-acp-test",
      mcpServers: [],
    });
    const turnFailed = new Promise<Extract<AgentStreamEvent, { type: "turn_failed" }>>(
      (resolve) => {
        session.subscribe((event) => {
          if (event.type === "turn_failed") {
            resolve(event);
          }
        });
      },
    );

    asInternals<ACPSessionInternals>(session).sessionId = sessionResponse.sessionId;
    asInternals<ACPSessionInternals>(session).connection = connection;

    await session.startTurn("hello");

    await expect(turnFailed).resolves.toMatchObject({
      error: expect.stringContaining(upstreamMessage),
      code: "-32000",
      diagnostic: expect.stringContaining("auth_required"),
    });
    await expect(turnFailed).resolves.toMatchObject({
      error: expect.not.stringContaining("[object Object]"),
    });
  });
});

interface ACPCloseInternals {
  child: ChildProcess | null;
  connection: unknown;
  sessionId: string | null;
  activeForegroundTurnId: string | null;
  agentCapabilities: { sessionCapabilities?: { close?: unknown } } | null;
}

async function startTerminal(
  session: ACPAgentSession,
  child: ChildProcess,
  command = "sleep",
): Promise<string> {
  vi.spyOn(spawnUtils, "spawnProcess").mockReturnValue(child as ChildProcessWithoutNullStreams);
  const terminal = await session.createTerminal({
    sessionId: "session-1",
    command,
    args: ["60"],
  });
  vi.restoreAllMocks();
  return terminal.terminalId;
}

describe("ACPAgentSession close() tree-kill", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  test("close() flushes a buffered provider-owned user message before unsubscribing", async () => {
    const session = createSession();
    const events: AgentStreamEvent[] = [];
    session.subscribe((event) => events.push(event));
    asInternals<ACPCloseInternals>(session).sessionId = "session-1";

    await session.sessionUpdate({
      sessionId: "session-1",
      update: {
        sessionUpdate: "user_message_chunk",
        content: { type: "image", data: "AA==", mimeType: "image/png" },
      } as SessionUpdate,
    });
    expect(events).toEqual([]);

    await session.close();

    expect(events).toEqual([
      {
        type: "timeline",
        provider: session.provider,
        item: { type: "user_message", text: "[image]" },
      },
    ]);
  });

  test("close() terminates the main child process via the process tree", async () => {
    const terminator = new FakeTerminator();
    const session = createSession({ terminateProcess: terminator.terminate });
    const internals = asInternals<ACPCloseInternals>(session);

    const child = createTerminalChildStub();
    // The ACP host process is set by the live connect handshake, which has no
    // in-test seam; everything else is driven through the public API.
    internals.child = child;
    internals.connection = null;
    internals.sessionId = null;

    await session.close();

    expect(terminator.terminated).toContain(child);
    expect(child.kill).not.toHaveBeenCalled();
  });

  test("close() terminates the provider when cancel and closeSession never settle", async () => {
    vi.useFakeTimers();
    try {
      const terminator = new FakeTerminator();
      const session = createSession({ terminateProcess: terminator.terminate });
      const child = createTerminalChildStub();
      const cancel = vi.fn(() => new Promise<void>(() => undefined));
      const unstableCloseSession = vi.fn(() => new Promise<void>(() => undefined));
      const internals = asInternals<ACPCloseInternals>(session);
      internals.child = child;
      internals.sessionId = "session-1";
      internals.activeForegroundTurnId = "turn-1";
      internals.agentCapabilities = { sessionCapabilities: { close: {} } };
      internals.connection = { cancel, unstable_closeSession: unstableCloseSession };

      let settled = false;
      const closing = (async () => {
        await session.close();
        settled = true;
      })();
      await vi.advanceTimersByTimeAsync(5_000);

      expect(settled).toBe(true);
      expect(cancel).toHaveBeenCalledWith({ sessionId: "session-1" });
      expect(unstableCloseSession).toHaveBeenCalledWith({ sessionId: "session-1" });
      expect(terminator.terminated).toContain(child);
      await closing;
    } finally {
      vi.useRealTimers();
    }
  });

  test("close() terminates running terminal child processes", async () => {
    const terminator = new FakeTerminator();
    const session = createSession({ terminateProcess: terminator.terminate });

    const terminalChild = createTerminalChildStub();
    await startTerminal(session, terminalChild);

    await session.close();

    expect(terminator.terminated).toContain(terminalChild);
    expect(terminalChild.kill).not.toHaveBeenCalled();
  });

  test("close() terminates terminal child processes in parallel", async () => {
    const terminator = new FakeTerminator("deferred");
    const session = createSession({ terminateProcess: terminator.terminate });

    const firstChild = createTerminalChildStub();
    const secondChild = createTerminalChildStub();
    await startTerminal(session, firstChild);
    await startTerminal(session, secondChild);

    const close = session.close();
    await Promise.resolve();

    expect(terminator.terminated).toEqual([firstChild, secondChild]);

    terminator.releaseAll();
    await close;
  });

  test("killTerminal terminates the terminal process tree without a direct SIGTERM", async () => {
    const terminator = new FakeTerminator();
    const session = createSession({ terminateProcess: terminator.terminate });

    const child = createTerminalChildStub();
    const terminalId = await startTerminal(session, child);

    await session.killTerminal({ sessionId: "session-1", terminalId });

    expect(terminator.terminated).toContain(child);
    expect(child.kill).not.toHaveBeenCalled();
  });

  test("releaseTerminal terminates and removes a running terminal", async () => {
    const terminator = new FakeTerminator();
    const session = createSession({ terminateProcess: terminator.terminate });

    const child = createTerminalChildStub();
    const terminalId = await startTerminal(session, child);

    await session.releaseTerminal({ sessionId: "session-1", terminalId });

    expect(terminator.terminated).toContain(child);
    expect(child.kill).not.toHaveBeenCalled();
    await expect(session.terminalOutput({ sessionId: "session-1", terminalId })).rejects.toThrow(
      `Unknown terminal '${terminalId}'`,
    );
  });
});

describe("ACPAgentSession initialization cleanup", () => {
  test("rejects a resume whose working directory was deleted instead of crashing", async () => {
    const root = await mkdtemp(path.join(tmpdir(), "paseo-acp-deleted-cwd-"));
    const deletedCwd = path.join(root, "worktree");
    const terminator = new FakeTerminator();
    const session = new ACPAgentSession(
      { provider: "test-acp", cwd: deletedCwd },
      {
        provider: "test-acp",
        logger: createTestLogger(),
        defaultCommand: [process.execPath, "-e", "process.stdin.resume()"],
        defaultModes: [],
        capabilities: {
          supportsStreaming: true,
          supportsSessionPersistence: true,
        },
        handle: { provider: "test-acp", sessionId: "archived-session" },
        terminateProcess: terminator.terminate,
      },
    );

    try {
      await expect(session.initializeResumedSession()).rejects.toThrow("ENOENT");
      expect(terminator.terminated).toHaveLength(1);
    } finally {
      await rm(root, { recursive: true, force: true });
    }
  });

  test("terminates the ACP process when session/new fails", async () => {
    const terminator = new FakeTerminator();
    const child = createProbeChildStub();

    class FailingNewSession extends ACPAgentSession {
      protected override async spawnProcess(): Promise<SpawnedACPProcess> {
        return {
          child,
          connection: {
            newSession: vi.fn().mockRejectedValue(new Error("session/new failed")),
          } as unknown as ClientSideConnection,
          initialize: { agentCapabilities: {} },
        };
      }
    }

    const session = new FailingNewSession(
      { provider: "copilot", cwd: "/tmp/paseo-acp-test" },
      {
        provider: "copilot",
        logger: createTestLogger(),
        defaultCommand: ["copilot", "--acp"],
        defaultModes: [],
        capabilities: {
          supportsStreaming: true,
          supportsSessionPersistence: true,
        },
        terminateProcess: terminator.terminate,
      },
    );

    await expect(session.initializeNewSession()).rejects.toThrow("session/new failed");

    expect(terminator.terminated).toContain(child);
  });

  test("terminates the ACP process when session/load fails", async () => {
    const terminator = new FakeTerminator();
    const child = createProbeChildStub();

    class FailingLoadSession extends ACPAgentSession {
      protected override async spawnProcess(): Promise<SpawnedACPProcess> {
        return {
          child,
          connection: {
            loadSession: vi.fn().mockRejectedValue(new Error("session/load failed")),
          } as unknown as ClientSideConnection,
          initialize: { agentCapabilities: { loadSession: true } },
        };
      }
    }

    const session = new FailingLoadSession(
      { provider: "cursor", cwd: "/tmp/paseo-acp-test" },
      {
        provider: "cursor",
        logger: createTestLogger(),
        defaultCommand: ["cursor-agent", "acp"],
        defaultModes: [],
        capabilities: {
          supportsStreaming: true,
          supportsSessionPersistence: true,
        },
        handle: { provider: "cursor", sessionId: "session-1" },
        terminateProcess: terminator.terminate,
      },
    );

    await expect(session.initializeResumedSession()).rejects.toThrow("session/load failed");

    expect(terminator.terminated).toContain(child);
  });
});

describe("ACPAgentClient probe cleanup", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  test("terminates the probe process tree and closes its stdio", async () => {
    const terminator = new FakeTerminator();
    const child = createProbeChildStub();

    class TestACPAgentClient extends ACPAgentClient {
      protected override async spawnProcess(): Promise<SpawnedACPProcess> {
        return {
          child,
          connection: {
            newSession: vi.fn().mockResolvedValue({
              modes: null,
              models: null,
              configOptions: [],
            }),
          },
          initialize: { agentCapabilities: {} },
        } as SpawnedACPProcess;
      }
    }

    const client = new TestACPAgentClient({
      provider: "claude-acp",
      logger: createTestLogger(),
      defaultCommand: ["claude", "--acp"],
      defaultModes: [],
      terminateProcess: terminator.terminate,
    });

    await client.fetchCatalog({ scope: "workspace", cwd: "/tmp/acp-models", force: false });

    expect(terminator.terminated).toContain(child);
    expect(child.stdin.destroyed).toBe(true);
    expect(child.stdout.destroyed).toBe(true);
    expect(child.stderr.destroyed).toBe(true);
  });

  test("closes the native catalog probe session before terminating its process", async () => {
    const terminator = new FakeTerminator();
    const child = createProbeChildStub();
    const closeSession = vi.fn().mockImplementation(async () => {
      expect(terminator.terminated).toEqual([]);
      return {};
    });

    class TestACPAgentClient extends ACPAgentClient {
      protected override async spawnProcess(): Promise<SpawnedACPProcess> {
        return {
          child,
          connection: {
            newSession: vi.fn().mockResolvedValue({
              sessionId: "catalog-probe-session",
              modes: null,
              models: null,
              configOptions: [],
            }),
            unstable_closeSession: closeSession,
          },
          initialize: { agentCapabilities: { sessionCapabilities: { close: {} } } },
        } as unknown as SpawnedACPProcess;
      }
    }

    const client = new TestACPAgentClient({
      provider: "claude-acp",
      logger: createTestLogger(),
      defaultCommand: ["claude", "--acp"],
      defaultModes: [],
      terminateProcess: terminator.terminate,
    });

    await client.fetchCatalog({ scope: "workspace", cwd: "/tmp/acp-models", force: false });

    expect(closeSession).toHaveBeenCalledWith({ sessionId: "catalog-probe-session" });
    expect(terminator.terminated).toContain(child);
  });

  test("closes a catalog probe session that finishes being created after refresh aborts", async () => {
    const terminator = new FakeTerminator();
    const child = createProbeChildStub();
    const abort = new AbortController();
    let resolveNewSession!: (value: {
      sessionId: string;
      modes: null;
      models: null;
      configOptions: [];
    }) => void;
    const newSession = vi.fn(
      () =>
        new Promise<{
          sessionId: string;
          modes: null;
          models: null;
          configOptions: [];
        }>((resolve) => {
          resolveNewSession = resolve;
        }),
    );
    const closeSession = vi.fn().mockResolvedValue({});

    class TestACPAgentClient extends ACPAgentClient {
      protected override async spawnProcess(): Promise<SpawnedACPProcess> {
        return {
          child,
          connection: { newSession, unstable_closeSession: closeSession },
          initialize: { agentCapabilities: { sessionCapabilities: { close: {} } } },
        } as unknown as SpawnedACPProcess;
      }
    }

    const client = new TestACPAgentClient({
      provider: "claude-acp",
      logger: createTestLogger(),
      defaultCommand: ["claude", "--acp"],
      defaultModes: [],
      terminateProcess: terminator.terminate,
    });
    const context: ProviderRefreshContext = {
      signal: abort.signal,
      runActivity: async (_name, operation) => await operation(),
    };
    const catalog = client.fetchCatalog(
      { scope: "workspace", cwd: "/tmp/acp-models", force: false },
      context,
    );
    await vi.waitFor(() => expect(newSession).toHaveBeenCalledOnce());

    abort.abort(new Error("refresh cancelled"));
    resolveNewSession({
      sessionId: "late-catalog-probe-session",
      modes: null,
      models: null,
      configOptions: [],
    });

    await expect(catalog).rejects.toThrow("refresh cancelled");
    expect(closeSession).toHaveBeenCalledWith({ sessionId: "late-catalog-probe-session" });
    expect(terminator.terminated).toContain(child);
  });

  test("closes the native feature probe session before terminating its process", async () => {
    const terminator = new FakeTerminator();
    const child = createProbeChildStub();
    const closeSession = vi.fn().mockResolvedValue({});

    class TestACPAgentClient extends ACPAgentClient {
      protected override async spawnProcess(): Promise<SpawnedACPProcess> {
        return {
          child,
          connection: {
            newSession: vi.fn().mockResolvedValue({
              sessionId: "feature-probe-session",
              modes: null,
              models: null,
              configOptions: [copilotAgentConfigOption("Probe Agent")],
            }),
            unstable_closeSession: closeSession,
          },
          initialize: { agentCapabilities: { sessionCapabilities: { close: {} } } },
        } as unknown as SpawnedACPProcess;
      }
    }

    const client = new TestACPAgentClient({
      provider: "copilot",
      logger: createTestLogger(),
      defaultCommand: ["copilot", "--acp"],
      defaultModes: [],
      configFeatureOptions: [COPILOT_AGENT_FEATURE_OPTION],
      terminateProcess: terminator.terminate,
    });

    await client.listFeatures({ provider: "copilot", cwd: "/tmp/acp-features" });

    expect(closeSession).toHaveBeenCalledWith({ sessionId: "feature-probe-session" });
    expect(terminator.terminated).toContain(child);
  });

  test("still terminates the probe when native session close fails", async () => {
    const terminator = new FakeTerminator();
    const child = createProbeChildStub();

    class TestACPAgentClient extends ACPAgentClient {
      protected override async spawnProcess(): Promise<SpawnedACPProcess> {
        return {
          child,
          connection: {
            newSession: vi.fn().mockResolvedValue({
              sessionId: "catalog-probe-session",
              modes: null,
              models: null,
              configOptions: [],
            }),
            unstable_closeSession: vi.fn().mockRejectedValue(new Error("close failed")),
          },
          initialize: { agentCapabilities: { sessionCapabilities: { close: {} } } },
        } as unknown as SpawnedACPProcess;
      }
    }

    const client = new TestACPAgentClient({
      provider: "claude-acp",
      logger: createTestLogger(),
      defaultCommand: ["claude", "--acp"],
      defaultModes: [],
      terminateProcess: terminator.terminate,
    });

    await client.fetchCatalog({ scope: "workspace", cwd: "/tmp/acp-models", force: false });

    expect(terminator.terminated).toContain(child);
  });

  test("keeps a session visible and cleans up after its history load times out", async () => {
    vi.useFakeTimers();
    try {
      const terminator = new FakeTerminator();
      const child = createProbeChildStub();
      const loadSession = vi.fn(async () => await new Promise(() => undefined));

      class TestACPAgentClient extends ACPAgentClient {
        protected override async spawnProcess(): Promise<SpawnedACPProcess> {
          return {
            child,
            connection: {
              listSessions: vi.fn().mockResolvedValue({
                sessions: [
                  {
                    sessionId: "slow-session",
                    cwd: "/tmp/acp-import",
                    title: "Slow but real",
                    updatedAt: "2026-09-04T20:00:00.000Z",
                  },
                ],
                nextCursor: null,
              }),
              loadSession,
              unstable_closeSession: vi.fn().mockResolvedValue({}),
            },
            initialize: {
              agentCapabilities: {
                loadSession: true,
                sessionCapabilities: { list: {}, close: {} },
              },
            },
          } as unknown as SpawnedACPProcess;
        }
      }

      const client = new TestACPAgentClient({
        provider: "claude-acp",
        logger: createTestLogger(),
        defaultCommand: ["claude", "--acp"],
        defaultModes: [],
        terminateProcess: terminator.terminate,
      });
      const listing = client.listImportableSessions({ limit: 1 });
      await vi.waitFor(() => expect(loadSession).toHaveBeenCalledOnce());

      await vi.advanceTimersByTimeAsync(30_000);

      await expect(listing).resolves.toEqual([
        {
          providerHandleId: "slow-session",
          cwd: "/tmp/acp-import",
          title: "Slow but real",
          firstPromptPreview: null,
          lastPromptPreview: null,
          lastActivityAt: new Date("2026-09-04T20:00:00.000Z"),
        },
      ]);
      expect(terminator.terminated).toContain(child);
    } finally {
      vi.useRealTimers();
    }
  });
});

describe("ACP session/load invariant — cwd and mcpServers always passed", () => {
  /**
   * Shared factory: creates an ACPAgentSession subclass whose spawnProcess
   * returns stubbed ACP internals so tests can inspect connection method calls
   * without spawning real processes. Each call produces fresh vi.fn() stubs.
   */
  function makeTestSession(args: {
    capabilities?: AgentCapabilityFlags;
    handle: AgentPersistenceHandle;
    loadSession?: ReturnType<typeof vi.fn>;
    unstableResumeSession?: ReturnType<typeof vi.fn>;
  }) {
    const loadSession =
      args.loadSession ??
      vi.fn().mockResolvedValue({
        sessionId: "session-1",
        modes: null,
        models: null,
        configOptions: [],
      });
    const unstableResumeSession =
      args.unstableResumeSession ??
      vi.fn().mockResolvedValue({
        sessionId: "session-1",
        modes: null,
        models: null,
        configOptions: [],
      });

    class TestSession extends ACPAgentSession {
      protected override async spawnProcess(): Promise<SpawnedACPProcess> {
        return {
          child: createProbeChildStub(),
          connection: {
            prompt: vi.fn(),
            loadSession,
            unstable_resumeSession: unstableResumeSession,
          } as unknown as ClientSideConnection,
          initialize: { agentCapabilities: args.capabilities ?? {} },
        } as SpawnedACPProcess;
      }
    }

    // Pass handle through the typed constructor option (no private-field casts).
    const session = new TestSession(
      { provider: "claude-acp", cwd: "/tmp/paseo-acp-test" },
      {
        provider: "claude-acp",
        logger: createTestLogger(),
        defaultCommand: ["claude", "--acp"],
        defaultModes: [],
        capabilities: {
          supportsStreaming: true,
          supportsSessionPersistence: true,
          supportsDynamicModes: true,
          supportsMcpServers: true,
          supportsReasoningStream: true,
          supportsToolInvocations: true,
          ...args.capabilities,
        },
        handle: args.handle,
      },
    );

    return { session, loadSession, unstableResumeSession };
  }

  test("loadSession is always called with sessionId, cwd, and mcpServers even when mcpServers is empty", async () => {
    const { session, loadSession } = makeTestSession({
      capabilities: { loadSession: true, supportsMcpServers: true },
      handle: { sessionId: "session-1", provider: "claude-acp" },
    });

    await session.initializeResumedSession();

    expect(loadSession).toHaveBeenCalledWith({
      sessionId: "session-1",
      cwd: "/tmp/paseo-acp-test",
      mcpServers: [],
    });
  });

  test("preserves assistant message IDs from loadSession replay", async () => {
    let session!: ACPAgentSession;
    const loadSession = async () => {
      await session.sessionUpdate({
        sessionId: "session-1",
        update: {
          sessionUpdate: "agent_message_chunk",
          messageId: "assistant-replay-1",
          content: { type: "text", text: "Welcome back" },
        } as SessionUpdate,
      });
      return {
        sessionId: "session-1",
        modes: null,
        models: null,
        configOptions: [],
      };
    };
    ({ session } = makeTestSession({
      capabilities: { loadSession: true },
      handle: { sessionId: "session-1", provider: "claude-acp" },
      loadSession,
    }));

    await session.initializeResumedSession();

    const history: AgentStreamEvent[] = [];
    for await (const event of session.streamHistory()) {
      history.push(event);
    }
    expect(history).toEqual([
      {
        type: "timeline",
        provider: "claude-acp",
        item: {
          type: "assistant_message",
          text: "Welcome back",
          messageId: "assistant-replay-1",
        },
      },
    ]);
  });

  test("coalesces an ID-less text and image user message during loadSession replay", async () => {
    let session!: ACPAgentSession;
    const loadSession = async () => {
      await session.sessionUpdate({
        sessionId: "session-1",
        update: {
          sessionUpdate: "user_message_chunk",
          content: { type: "text", text: "hey" },
        } as SessionUpdate,
      });
      await session.sessionUpdate({
        sessionId: "session-1",
        update: {
          sessionUpdate: "user_message_chunk",
          content: { type: "image", data: "AA==", mimeType: "image/png" },
        } as SessionUpdate,
      });
      await session.sessionUpdate({
        sessionId: "session-1",
        update: {
          sessionUpdate: "agent_message_chunk",
          messageId: "assistant-replay-1",
          content: { type: "text", text: "Hello" },
        } as SessionUpdate,
      });
      return {
        sessionId: "session-1",
        modes: null,
        models: null,
        configOptions: [],
      };
    };
    ({ session } = makeTestSession({
      capabilities: { loadSession: true },
      handle: { sessionId: "session-1", provider: "test-acp" },
      loadSession,
    }));

    await session.initializeResumedSession();

    const history: AgentStreamEvent[] = [];
    for await (const event of session.streamHistory()) {
      history.push(event);
    }
    expect(history).toEqual([
      {
        type: "timeline",
        provider: session.provider,
        item: { type: "user_message", text: "hey[image]" },
      },
      {
        type: "timeline",
        provider: session.provider,
        item: {
          type: "assistant_message",
          text: "Hello",
          messageId: "assistant-replay-1",
        },
      },
    ]);
  });

  test("assigns stable fallback IDs to ID-less assistant messages during loadSession replay", async () => {
    let session!: ACPAgentSession;
    const loadSession = async () => {
      for (const text of ["Loaded", " response"]) {
        await session.sessionUpdate({
          sessionId: "session-1",
          update: {
            sessionUpdate: "agent_message_chunk",
            content: { type: "text", text },
          } as SessionUpdate,
        });
      }
      await session.sessionUpdate({
        sessionId: "session-1",
        update: {
          sessionUpdate: "user_message_chunk",
          content: { type: "text", text: "Follow up" },
        } as SessionUpdate,
      });
      await session.sessionUpdate({
        sessionId: "session-1",
        update: {
          sessionUpdate: "agent_message_chunk",
          content: { type: "text", text: "Loaded second response" },
        } as SessionUpdate,
      });
      return {
        sessionId: "session-1",
        modes: null,
        models: null,
        configOptions: [],
      };
    };
    ({ session } = makeTestSession({
      capabilities: { loadSession: true },
      handle: { sessionId: "session-1", provider: "claude-acp" },
      loadSession,
    }));

    await session.initializeResumedSession();

    const assistantMessages: Array<{ text: string; messageId?: string }> = [];
    for await (const event of session.streamHistory()) {
      if (event.type === "timeline" && event.item.type === "assistant_message") {
        assistantMessages.push(event.item);
      }
    }
    expect(assistantMessages).toHaveLength(3);
    expect(assistantMessages[0].messageId).toEqual(expect.any(String));
    expect(assistantMessages[1].messageId).toBe(assistantMessages[0].messageId);
    expect(assistantMessages[2].messageId).toEqual(expect.any(String));
    expect(assistantMessages[2].messageId).not.toBe(assistantMessages[0].messageId);
  });

  test("loadSession is always called with mcpServers even when supportsMcpServers is false", async () => {
    const { session, loadSession } = makeTestSession({
      capabilities: { loadSession: true, supportsMcpServers: false },
      handle: { sessionId: "session-1", provider: "claude-acp" },
    });

    await session.initializeResumedSession();

    // Even with supportsMcpServers=false, mcpServers: [] must still be passed
    expect(loadSession).toHaveBeenCalledWith({
      sessionId: "session-1",
      cwd: "/tmp/paseo-acp-test",
      mcpServers: [],
    });
  });

  test("unstable_resumeSession is always called with sessionId, cwd, and mcpServers", async () => {
    const { session, unstableResumeSession } = makeTestSession({
      capabilities: { sessionCapabilities: { resume: {} } },
      handle: { sessionId: "session-1", provider: "claude-acp" },
    });

    await session.initializeResumedSession();

    expect(unstableResumeSession).toHaveBeenCalledWith({
      sessionId: "session-1",
      cwd: "/tmp/paseo-acp-test",
      mcpServers: [],
    });
  });
});

const SILENT_CLOSE_ACP_AGENT = `
import { readFileSync, writeFileSync } from "node:fs";
import { Readable, Writable } from "node:stream";
const { AgentSideConnection, PROTOCOL_VERSION, ndJsonStream } = await import(process.env.ACP_SDK_URL);
writeFileSync(process.env.ACP_PID_FILE, String(process.pid));
new AgentSideConnection(
  () => ({
    async initialize() {
      return {
        protocolVersion: PROTOCOL_VERSION,
        agentCapabilities: { sessionCapabilities: { close: {} } },
      };
    },
    async newSession() {
      return { sessionId: "silent-close-session" };
    },
    async authenticate() {},
    async cancel() {},
    unstable_closeSession() {
      return new Promise(() => {});
    },
  }),
  ndJsonStream(Writable.toWeb(process.stdout), Readable.toWeb(process.stdin)),
);
`;

function isProcessAlive(pid: number): boolean {
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
}

async function settlesWithin(promise: Promise<unknown>, timeoutMs: number): Promise<boolean> {
  let timer: NodeJS.Timeout | undefined;
  const timedOut = new Promise<false>((resolve) => {
    timer = setTimeout(() => resolve(false), timeoutMs);
  });
  try {
    return await Promise.race([promise.then(() => true), timedOut]);
  } finally {
    clearTimeout(timer);
  }
}

interface SilentCloseProvider {
  session: ACPAgentSession;
  readPid(): Promise<number>;
  dispose(): Promise<void>;
}

async function startSilentCloseProvider(): Promise<SilentCloseProvider> {
  const dir = await mkdtemp(path.join(tmpdir(), "paseo-acp-silent-close-"));
  const agentScript = path.join(dir, "agent.mjs");
  const pidFile = path.join(dir, "agent.pid");
  await writeFile(agentScript, SILENT_CLOSE_ACP_AGENT);
  const session = new ACPAgentSession(
    { provider: "silent-close-acp", cwd: dir },
    {
      provider: "silent-close-acp",
      logger: createTestLogger(),
      defaultCommand: [process.execPath, agentScript],
      defaultModes: [],
      capabilities: {
        supportsStreaming: true,
        supportsSessionPersistence: true,
        supportsDynamicModes: true,
        supportsMcpServers: true,
        supportsReasoningStream: true,
        supportsToolInvocations: true,
      },
      launchEnv: {
        ACP_SDK_URL: pathToFileURL(
          createRequire(import.meta.url).resolve("@agentclientprotocol/sdk"),
        ).href,
        ACP_PID_FILE: pidFile,
      },
    },
  );
  const readPid = async () => Number(await readFile(pidFile, "utf8"));
  return {
    session,
    readPid,
    async dispose() {
      const pid = await readPid().catch(() => null);
      if (pid !== null && isProcessAlive(pid)) {
        process.kill(pid, "SIGKILL");
      }
      await rm(dir, { recursive: true, force: true });
    },
  };
}

describe("ACPAgentSession close() with an unresponsive provider", () => {
  let provider: SilentCloseProvider | null = null;

  afterEach(async () => {
    await provider?.dispose();
    provider = null;
  });

  test("terminates a provider that never answers session/close", async () => {
    provider = await startSilentCloseProvider();
    await provider.session.initializeNewSession();
    const pid = await provider.readPid();
    expect(isProcessAlive(pid)).toBe(true);

    expect(await settlesWithin(provider.session.close(), 8_000)).toBe(true);
    expect(isProcessAlive(pid)).toBe(false);
  }, 15_000);
});

// An ACP agent that advertises two models (or none, with ACP_ADVERTISE_MODELS=none) but,
// like Cline, runs any model id it is given.
const PERMISSIVE_MODEL_ACP_AGENT = `
import { Readable, Writable } from "node:stream";
const { AgentSideConnection, PROTOCOL_VERSION, ndJsonStream } = await import(process.env.ACP_SDK_URL);
let currentModelId = "advertised-default";
const modelState = () =>
  process.env.ACP_ADVERTISE_MODELS === "none"
    ? undefined
    : {
        currentModelId,
        availableModels: [
          { modelId: "advertised-default", name: "Advertised default" },
          { modelId: "advertised-other", name: "Advertised other" },
        ],
      };
new AgentSideConnection(
  () => ({
    async initialize() {
      return { protocolVersion: PROTOCOL_VERSION, agentCapabilities: { loadSession: true } };
    },
    async newSession() {
      return { sessionId: "permissive-session", models: modelState() };
    },
    async loadSession() {
      return { models: modelState() };
    },
    async unstable_setSessionModel({ modelId }) {
      currentModelId = modelId;
      return {};
    },
    async authenticate() {},
    async cancel() {},
    async prompt() {
      return { stopReason: "end_turn" };
    },
  }),
  ndJsonStream(Writable.toWeb(process.stdout), Readable.toWeb(process.stdin)),
);
`;

describe("ACP idle runtime cleanup", () => {
  test("pending permissions veto cleanup until answered", async () => {
    const session = createSession();
    try {
      expect(await session.canStopRuntime()).toBe(true);
      const permission = session.requestPermission({
        sessionId: "session-1",
        toolCall: { toolCallId: "tool-1", title: "Run command", status: "pending" },
        options: [{ optionId: "allow", name: "Allow", kind: "allow_once" }],
      });
      expect(await session.canStopRuntime()).toBe(false);
      await session.respondToPermission(session.getPendingPermissions()[0]!.id, {
        behavior: "allow",
      });
      await permission;
      expect(await session.canStopRuntime()).toBe(true);
    } finally {
      await session.close();
    }
  });

  test("a failed terminal spawn does not pin the runtime or discard its diagnostic", async () => {
    const session = createSession();
    try {
      const terminal = await session.createTerminal({
        sessionId: "session-1",
        command: path.join(tmpdir(), "paseo-no-such-command", "missing"),
      });
      await expect(
        session.waitForTerminalExit({ sessionId: "session-1", ...terminal }),
      ).rejects.toThrow("ENOENT");
      expect(await session.canStopRuntime()).toBe(true);
      expect(
        (await session.terminalOutput({ sessionId: "session-1", ...terminal })).output,
      ).toContain("ENOENT");
      await session.releaseTerminal({ sessionId: "session-1", ...terminal });
      expect(await session.canStopRuntime()).toBe(true);
    } finally {
      await session.close();
    }
  });

  test("retains a quiet terminal after a finished turn, then waits a fresh idle hour after exit", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "paseo-acp-idle-"));
    const agentScript = path.join(dir, "agent.mjs");
    await writeFile(agentScript, PERMISSIVE_MODEL_ACP_AGENT);
    const session = new ACPAgentSession(
      { provider: "test-acp", cwd: dir },
      {
        provider: "test-acp",
        logger: createTestLogger(),
        defaultCommand: [process.execPath, agentScript],
        defaultModes: [],
        capabilities: { supportsStreaming: true, supportsSessionPersistence: true },
        launchEnv: {
          ACP_SDK_URL: pathToFileURL(
            createRequire(import.meta.url).resolve("@agentclientprotocol/sdk"),
          ).href,
        },
      },
    );
    const hour = 60 * 60_000;
    let now = 0;
    let closed = false;
    const candidate = {
      identity: session,
      agentId: "test-agent",
      parentAgentId: undefined,
      provider: session.provider,
      sessionId: "permissive-session",
      lastActivityAt: 0,
      isEligible: () => !closed,
      canStopRuntime: (session as AgentSession).canStopRuntime?.bind(session),
      async closeIfEligible(revalidate: () => boolean) {
        if (!revalidate()) return false;
        await session.close();
        closed = true;
        return true;
      },
    };
    const cleanup = createIdleRuntimeCleanup({
      enabled: true,
      now: () => now,
      runtimes: () => (closed ? [] : [candidate]),
      schedule: () => () => {},
      logger: createTestLogger(),
    });
    try {
      await session.initializeNewSession();
      const completed = new Promise<void>((resolve) =>
        session.subscribe((event) => {
          if (event.type === "turn_completed") resolve();
        }),
      );
      let activeTurnSafety: Promise<boolean> | undefined;
      session.subscribe((event) => {
        if (event.type === "turn_started") activeTurnSafety = session.canStopRuntime();
      });
      await session.startTurn("Finish this foreground turn");
      await expect(activeTurnSafety).resolves.toBe(false);
      await completed;
      expect(await session.canStopRuntime()).toBe(true);
      await session.sessionUpdate({
        sessionId: candidate.sessionId,
        update: {
          sessionUpdate: "user_message_chunk",
          content: { type: "text", text: "buffered" },
        },
      });
      expect(await session.canStopRuntime()).toBe(false);
      await session.sessionUpdate({
        sessionId: candidate.sessionId,
        update: {
          sessionUpdate: "agent_message_chunk",
          content: { type: "text", text: "flush" },
        },
      });
      expect(await session.canStopRuntime()).toBe(true);
      const terminal = await session.createTerminal({
        sessionId: candidate.sessionId,
        command: process.execPath,
        args: ["-e", "setInterval(() => {}, 1000)"],
      });
      now = 2 * hour;
      expect((await cleanup.sweep()).collected).toEqual([]);
      expect(closed).toBe(false);
      expect(await session.terminalOutput({ sessionId: candidate.sessionId, ...terminal })).toEqual(
        {
          output: "",
          truncated: false,
          exitStatus: undefined,
        },
      );
      await session.killTerminal({ sessionId: candidate.sessionId, ...terminal });
      await session.waitForTerminalExit({ sessionId: candidate.sessionId, ...terminal });
      expect((await cleanup.sweep()).collected).toEqual([]);
      now += hour - 1;
      expect((await cleanup.sweep()).collected).toEqual([]);
      now += 1;
      expect((await cleanup.sweep()).collected).toEqual([
        {
          agentId: candidate.agentId,
          provider: candidate.provider,
          sessionId: candidate.sessionId,
        },
      ]);
      expect(closed).toBe(true);
    } finally {
      await cleanup.close();
      await session.close();
      await rm(dir, { recursive: true, force: true });
    }
  });
});

describe("custom ACP provider with configured models", () => {
  let dir: string;
  let agentScript: string;
  let session: AgentSession | null = null;

  beforeEach(async () => {
    dir = await mkdtemp(path.join(tmpdir(), "paseo-acp-configured-models-"));
    agentScript = path.join(dir, "agent.mjs");
    await writeFile(agentScript, PERMISSIVE_MODEL_ACP_AGENT);
  });

  afterEach(async () => {
    await session?.close();
    session = null;
    await rm(dir, { recursive: true, force: true });
  });

  function buildRegistry(
    override: {
      models?: Array<{ id: string; label: string }>;
      additionalModels?: Array<{ id: string; label: string }>;
    },
    agentEnv: Record<string, string> = {},
  ) {
    return buildProviderRegistry(createTestLogger(), {
      providerOverrides: {
        permissive: {
          extends: "acp",
          label: "Permissive",
          command: [process.execPath, agentScript],
          env: {
            ACP_SDK_URL: pathToFileURL(
              createRequire(import.meta.url).resolve("@agentclientprotocol/sdk"),
            ).href,
            ...agentEnv,
          },
          ...override,
        },
      },
    });
  }

  async function createSessionOnModel(
    registry: ReturnType<typeof buildRegistry>,
    model: string,
  ): Promise<AgentSession> {
    const client = registry.permissive.createClient(createTestLogger());
    session = await client.createSession({ provider: "permissive", cwd: dir, model });
    return session;
  }

  test("runs on a model added through additionalModels that the agent does not advertise", async () => {
    const registry = buildRegistry({
      additionalModels: [{ id: "stealth/pixel-canary", label: "Pixel Canary" }],
    });

    const created = await createSessionOnModel(registry, "stealth/pixel-canary");

    expect((await created.getRuntimeInfo()).model).toBe("stealth/pixel-canary");
  });

  test("runs on a model from a replacement models list that the agent does not advertise", async () => {
    const registry = buildRegistry({
      models: [{ id: "stealth/pixel-canary", label: "Pixel Canary" }],
    });

    const created = await createSessionOnModel(registry, "stealth/pixel-canary");

    expect((await created.getRuntimeInfo()).model).toBe("stealth/pixel-canary");
  });

  test("runs on a model from additionalModels when the agent advertises no models", async () => {
    const registry = buildRegistry(
      { additionalModels: [{ id: "stealth/pixel-canary", label: "Pixel Canary" }] },
      { ACP_ADVERTISE_MODELS: "none" },
    );

    const created = await createSessionOnModel(registry, "stealth/pixel-canary");

    expect((await created.getRuntimeInfo()).model).toBe("stealth/pixel-canary");
  });

  test("resumes on a model added through additionalModels that the agent does not advertise", async () => {
    const registry = buildRegistry({
      additionalModels: [{ id: "stealth/pixel-canary", label: "Pixel Canary" }],
    });
    const client = registry.permissive.createClient(createTestLogger());

    session = await client.resumeSession(
      { provider: "permissive", sessionId: "permissive-session", metadata: { cwd: dir } },
      { model: "stealth/pixel-canary" },
    );

    expect((await session.getRuntimeInfo()).model).toBe("stealth/pixel-canary");
  });

  test("keeps the agent's model when the requested model is neither advertised nor configured", async () => {
    const registry = buildRegistry({
      additionalModels: [{ id: "stealth/pixel-canary", label: "Pixel Canary" }],
    });

    const created = await createSessionOnModel(registry, "stale/removed-model");

    expect((await created.getRuntimeInfo()).model).toBe("advertised-default");
  });
});

describe("ACP interruption contract — interruption waits for the provider to release the session", () => {
  test("interrupt() does not resolve while the provider still owns the foreground turn", async () => {
    const session = createSession();
    let resolvePrompt!: (value: PromptResponse) => void;
    const prompt = vi.fn(
      () =>
        new Promise<PromptResponse>((resolve) => {
          resolvePrompt = resolve;
        }),
    );
    const cancel = vi.fn().mockResolvedValue(undefined);

    asInternals<ACPSessionInternals>(session).sessionId = "session-1";
    asInternals<ACPSessionInternals>(session).connection = { prompt, cancel };

    await session.startTurn("hello");

    let interruptResolved = false;
    const interrupt = session.interrupt().then(() => {
      interruptResolved = true;
      return;
    });

    // Let the cancellation notification and every pending microtask settle, so
    // an implementation that resolves the interruption too early is observable.
    await new Promise((resolve) => setTimeout(resolve, 20));

    expect(cancel).toHaveBeenCalledWith({ sessionId: "session-1" });

    // The provider still owns the turn, so the interruption must not resolve
    // yet: a caller that replaces the turn would otherwise send its prompt
    // into a busy session.
    expect(interruptResolved).toBe(false);

    resolvePrompt({ stopReason: "cancelled", usage: { outputTokens: 0 } });
    await interrupt;

    expect(interruptResolved).toBe(true);
  });

  test("interrupt() rejects when the provider keeps the turn past the release timeout", async () => {
    vi.useFakeTimers();
    try {
      const session = createSession();
      const prompt = vi.fn(() => new Promise<PromptResponse>(() => {}));
      const cancel = vi.fn().mockResolvedValue(undefined);

      asInternals<ACPSessionInternals>(session).sessionId = "session-1";
      asInternals<ACPSessionInternals>(session).connection = { prompt, cancel };

      await session.startTurn("hello");

      const assertion = expect(session.interrupt()).rejects.toThrow(
        /still owns the foreground turn/,
      );
      await vi.advanceTimersByTimeAsync(2_000);
      await assertion;
    } finally {
      vi.useRealTimers();
    }
  });

  test("interrupt() waits for the cancelled turn and not for a turn that starts later", async () => {
    const session = createSession();
    let resolvePrompt!: (value: PromptResponse) => void;
    const prompt = vi.fn(
      () =>
        new Promise<PromptResponse>((resolve) => {
          resolvePrompt = resolve;
        }),
    );
    // The provider answers the cancelled turn and starts another one while
    // the cancellation notification is still in flight. The interruption must
    // follow the turn it cancelled, not the newer one.
    const cancel = vi.fn(async () => {
      resolvePrompt({ stopReason: "cancelled", usage: { outputTokens: 0 } });
      await new Promise((resolve) => setTimeout(resolve, 0));
      await session.startTurn("second");
    });

    asInternals<ACPSessionInternals>(session).sessionId = "session-1";
    asInternals<ACPSessionInternals>(session).connection = { prompt, cancel };

    await session.startTurn("hello");

    const interrupt = session.interrupt();
    const outcome = await Promise.race([
      interrupt.then(
        () => "resolved",
        (error: Error) => `rejected: ${error.message}`,
      ),
      new Promise<string>((resolve) => setTimeout(() => resolve("timeout"), 500)),
    ]);

    expect(outcome).toBe("resolved");
  });

  test("interrupt() rejects when the provider never confirms the end of the turn", async () => {
    const session = createSession();
    let rejectPrompt!: (error: Error) => void;
    const prompt = vi.fn(
      () =>
        new Promise<PromptResponse>((_resolve, reject) => {
          rejectPrompt = reject;
        }),
    );
    const cancel = vi.fn().mockResolvedValue(undefined);

    asInternals<ACPSessionInternals>(session).sessionId = "session-1";
    asInternals<ACPSessionInternals>(session).connection = { prompt, cancel };

    await session.startTurn("hello");

    const interrupt = session.interrupt();
    await new Promise((resolve) => setTimeout(resolve, 10));

    // A rejected prompt never confirms that the provider released the session,
    // so the interruption has to fail instead of reporting a free session.
    rejectPrompt(new Error("provider rejected the prompt"));

    await expect(interrupt).rejects.toThrow(/did not confirm the end of the foreground turn/);
  });

  test("interrupt() resolves when the provider process exits before it answers", async () => {
    const session = createSession();
    const prompt = vi.fn(() => new Promise<PromptResponse>(() => {}));
    const cancel = vi.fn().mockResolvedValue(undefined);

    asInternals<ACPSessionInternals>(session).sessionId = "session-1";
    asInternals<ACPSessionInternals>(session).connection = { prompt, cancel };

    await session.startTurn("hello");

    const interrupt = session.interrupt();

    // A confirmed process exit ends the turn: the provider can no longer own
    // the session, so waiting for its answer would hang forever.
    asInternals<ACPSessionInternals>(session).handleProviderExit(1, null, []);

    await expect(interrupt).resolves.toBeUndefined();
  });

  test("interrupt() resolves immediately when no foreground turn is running", async () => {
    const session = createSession();
    const prompt = vi.fn(() => new Promise<PromptResponse>(() => {}));
    const cancel = vi.fn().mockResolvedValue(undefined);

    asInternals<ACPSessionInternals>(session).sessionId = "session-1";
    asInternals<ACPSessionInternals>(session).connection = { prompt, cancel };

    await expect(session.interrupt()).resolves.toBeUndefined();
    expect(cancel).not.toHaveBeenCalled();
  });
});

describe("ACP import cancellation", () => {
  const IMPORT_SESSION_ID = "big-history-session";

  function createImportPersistenceHandle(): AgentPersistenceHandle {
    return {
      provider: "claude-acp",
      sessionId: IMPORT_SESSION_ID,
      metadata: { provider: "claude-acp", cwd: "/tmp/paseo-acp-test" },
    };
  }

  /**
   * Stands in for the provider process. Importing is a session/load, so the
   * load is what hangs while the user waits; the abort has to end the child.
   */
  function makeImportSession(args: {
    spawn: () => Promise<SpawnedACPProcess>;
    terminateProcess: ProcessTerminator;
  }) {
    class TestSession extends ACPAgentSession {
      protected override async spawnProcess(): Promise<SpawnedACPProcess> {
        return await args.spawn();
      }
    }

    return new TestSession(
      { provider: "claude-acp", cwd: "/tmp/paseo-acp-test" },
      {
        provider: "claude-acp",
        logger: createTestLogger(),
        defaultCommand: ["claude", "--acp"],
        defaultModes: [],
        capabilities: {
          supportsStreaming: true,
          supportsSessionPersistence: true,
          supportsDynamicModes: true,
          supportsMcpServers: true,
          supportsReasoningStream: true,
          supportsToolInvocations: true,
        },
        handle: createImportPersistenceHandle(),
        terminateProcess: args.terminateProcess,
      },
    );
  }

  function makeSpawned(
    child: ChildProcessWithoutNullStreams,
    loadSession: ReturnType<typeof vi.fn>,
  ) {
    return {
      child,
      connection: {
        prompt: vi.fn(),
        loadSession,
        unstable_resumeSession: vi.fn(),
      } as unknown as ClientSideConnection,
      initialize: { agentCapabilities: { loadSession: true } },
    } as SpawnedACPProcess;
  }

  test("cancelling an import kills the process that is loading the history", async () => {
    const terminate = new FakeTerminator();
    const child = createProbeChildStub();
    let loadStarted: () => void = () => {};
    const loadStartedPromise = new Promise<void>((resolve) => {
      loadStarted = resolve;
    });
    const loadSession = vi.fn(() => {
      loadStarted();
      // The provider rebuilds the whole conversation and never answers.
      return new Promise<never>(() => {});
    });
    const session = makeImportSession({
      spawn: async () => makeSpawned(child, loadSession),
      terminateProcess: terminate.terminate,
    });

    const resuming = session.initializeResumedSession();
    await loadStartedPromise;
    expect(countInFlightImportProcesses("claude-acp", IMPORT_SESSION_ID)).toBe(1);

    await expect(cancelInFlightImportProcess("claude-acp", IMPORT_SESSION_ID)).resolves.toBe(1);

    // The resume ends as cancelled instead of waiting for a history nobody will
    // look at, and the process that was loading it is gone.
    await expect(resuming).rejects.toThrow(/session import was cancelled/);
    expect(terminate.terminated[0]).toBe(child);
    expect(countInFlightImportProcesses("claude-acp", IMPORT_SESSION_ID)).toBe(0);
  });

  test("an import cancelled during the handshake stops the process that starts afterwards", async () => {
    const terminate = new FakeTerminator();
    const child = createProbeChildStub();
    let releaseSpawn: (spawned: SpawnedACPProcess) => void = () => {};
    const session = makeImportSession({
      spawn: () =>
        new Promise<SpawnedACPProcess>((resolve) => {
          releaseSpawn = resolve;
        }),
      terminateProcess: terminate.terminate,
    });

    const resuming = session.initializeResumedSession();
    await expect(cancelInFlightImportProcess("claude-acp", IMPORT_SESSION_ID)).resolves.toBe(1);
    await expect(resuming).rejects.toThrow(/session import was cancelled/);

    releaseSpawn(makeSpawned(child, vi.fn()));

    await vi.waitFor(() => {
      expect(terminate.terminated).toContain(child);
    });
  });
});

describe("provider delegation stop", () => {
  const STOP_METHOD = "mcode/session/delegation/stop";
  const GET_METHOD = "mcode/session/delegation/get";
  const EXTENSIONS_KEY = "minimax-code/extensions";

  function stubDelegationConnection(args: { stopResponse?: unknown; snapshot?: unknown }): {
    connection: ACPSessionInternals["connection"];
    extMethod: ReturnType<typeof vi.fn>;
  } {
    const extMethod = vi.fn(async (method: string) => {
      if (method === STOP_METHOD) {
        return args.stopResponse ?? { receipt: {} };
      }
      if (method === GET_METHOD) {
        return args.snapshot ?? { snapshot: { schemaVersion: 1, members: [] } };
      }
      return {};
    });
    return {
      extMethod,
      connection: { prompt: vi.fn(), extMethod } as unknown as ACPSessionInternals["connection"],
    };
  }

  test("calls the stop on the root session and reads what actually stopped from the tree", async () => {
    // A child is not an addressable turn: only the root id stops the tree. The
    // receipt comes back with an empty stoppedSessionIds and calls the root
    // stopped either way, so the answer is `acknowledged` and the rows carry the
    // statuses the user reads.
    const session = createSession();
    const internals = asInternals<ACPSessionInternals>(session);
    internals.sessionId = "root-1";
    internals.agentMeta = { [EXTENSIONS_KEY]: { methods: [GET_METHOD, STOP_METHOD] } };
    const stub = stubDelegationConnection({
      stopResponse: {
        receipt: {
          schemaVersion: 1,
          rootSessionId: "root-1",
          rootStopped: true,
          stoppedSessionIds: [],
          activeSessionIds: [],
          failedSessionIds: [],
        },
      },
      snapshot: {
        snapshot: {
          schemaVersion: 1,
          rootSessionId: "root-1",
          members: [
            {
              sessionId: "child-1",
              parentSessionId: "root-1",
              agentName: "explore",
              task: "List files",
              status: "stopped",
              createdAtMs: 1_700_000_000_000,
              updatedAtMs: 1_700_000_002_000,
            },
          ],
        },
      },
    });
    internals.connection = stub.connection;

    const events: AgentStreamEvent[] = [];
    session.subscribe((event) => events.push(event));

    await expect(session.stopProviderDelegations()).resolves.toEqual({ acknowledged: true });

    expect(stub.extMethod).toHaveBeenCalledWith(STOP_METHOD, { sessionId: "root-1" });
    // The tree is read again afterwards, so the rows report the real status.
    expect(stub.extMethod).toHaveBeenCalledWith(GET_METHOD, { sessionId: "root-1" });
    const rows = events.flatMap((event) =>
      event.type === "provider_subagent" && event.event.type === "upsert"
        ? [{ id: event.event.id, status: event.event.status }]
        : [],
    );
    expect(rows).toEqual([{ id: "child-1", status: "canceled" }]);
  });

  test("refuses to stop when the provider never advertised the stop method", async () => {
    const session = createSession();
    const internals = asInternals<ACPSessionInternals>(session);
    internals.sessionId = "root-1";
    internals.agentMeta = { [EXTENSIONS_KEY]: { methods: [GET_METHOD] } };
    const stub = stubDelegationConnection({});
    internals.connection = stub.connection;

    await expect(session.stopProviderDelegations()).rejects.toThrow(
      "does not support stopping delegated tasks",
    );
    expect(stub.extMethod).not.toHaveBeenCalled();
  });

  function makeInitSession(methods: string[] | null) {
    class TestSession extends ACPAgentSession {
      protected override async spawnProcess(): Promise<SpawnedACPProcess> {
        return {
          child: createProbeChildStub(),
          connection: {
            prompt: vi.fn(),
            newSession: vi.fn().mockResolvedValue({
              sessionId: "root-1",
              modes: null,
              models: null,
              configOptions: [],
            }),
            extMethod: vi.fn().mockResolvedValue({ snapshot: { members: [] } }),
          } as unknown as ClientSideConnection,
          initialize: {
            agentCapabilities: {},
            ...(methods ? { _meta: { [EXTENSIONS_KEY]: { methods } } } : {}),
          },
        } as unknown as SpawnedACPProcess;
      }
    }

    return new TestSession(
      { provider: "minimax-code", cwd: "/tmp/paseo-acp-test" },
      {
        provider: "minimax-code",
        logger: createTestLogger(),
        defaultCommand: ["mcode", "--acp"],
        defaultModes: [],
        capabilities: {
          supportsStreaming: true,
          supportsSessionPersistence: true,
          supportsDynamicModes: true,
          supportsMcpServers: true,
          supportsReasoningStream: true,
          supportsToolInvocations: true,
        },
      },
    );
  }

  test("advertises the stop to the app only after the provider names it", async () => {
    const advertised = makeInitSession([GET_METHOD, STOP_METHOD]);
    await advertised.initializeNewSession();
    expect(advertised.capabilities.supportsProviderDelegationStop).toBe(true);

    const silent = makeInitSession([GET_METHOD]);
    await silent.initializeNewSession();
    expect(silent.capabilities.supportsProviderDelegationStop).toBe(false);
  });
});
