import { describe, expect, it } from "vitest";
import type { ConfirmDialogInput } from "@/utils/confirm-dialog";
import {
  requestStopProviderDelegations,
  resolveStopProviderDelegationsDialog,
  resolveStopProviderDelegationsOutcome,
  type ProviderDelegationStopStatus,
  type StopProviderDelegationsDeps,
} from "./provider-delegation-stop";

const translations: Record<string, string> = {
  "subagents.stopProviderDelegationsDialogTitle": "Stop {{provider}}?",
  "subagents.stopProviderDelegationsDialogMessage":
    "This stops the parent turn and every task {{provider}} delegated ({{count}} running).",
  "subagents.stopProviderDelegationsConfirm": "Stop everything",
  "common.actions.cancel": "Cancel",
};

function t(key: string, options?: Record<string, unknown>): string {
  const template = translations[key] ?? key;
  return template.replace(/\{\{(\w+)\}\}/g, (_match, name: string) =>
    String(options?.[name] ?? `{{${name}}}`),
  );
}

interface RecordedCall {
  serverId: string;
  agentId: string;
}

interface FakeEnv {
  deps: StopProviderDelegationsDeps;
  recordedConfirmInputs: ConfirmDialogInput[];
  recordedStops: RecordedCall[];
  recordedErrors: unknown[];
  activeTaskCount: { value: number };
}

function createFakeEnv(options: {
  confirmResult?: boolean;
  acknowledged?: boolean;
  activeTaskCountAfterStop?: number;
  stopError?: Error;
}): FakeEnv {
  const recordedConfirmInputs: ConfirmDialogInput[] = [];
  const recordedStops: RecordedCall[] = [];
  const recordedErrors: unknown[] = [];
  const activeTaskCount = { value: options.activeTaskCountAfterStop ?? 0 };

  return {
    recordedConfirmInputs,
    recordedStops,
    recordedErrors,
    activeTaskCount,
    deps: {
      confirm: async (dialog) => {
        recordedConfirmInputs.push(dialog);
        return options.confirmResult ?? true;
      },
      t,
      stopProviderDelegations: async (input) => {
        recordedStops.push(input);
        if (options.stopError) {
          throw options.stopError;
        }
        return { acknowledged: options.acknowledged ?? true };
      },
      countActiveTasks: () => activeTaskCount.value,
      reportError: (error) => {
        recordedErrors.push(error);
      },
    },
  };
}

describe("resolveStopProviderDelegationsDialog", () => {
  it("names the parent and the delegated tasks the stop covers", () => {
    expect(
      resolveStopProviderDelegationsDialog(
        { providerLabel: "Minimax Code", activeTaskCount: 3 },
        t,
      ),
    ).toEqual({
      title: "Stop Minimax Code?",
      message: "This stops the parent turn and every task Minimax Code delegated (3 running).",
      confirmLabel: "Stop everything",
      cancelLabel: "Cancel",
      destructive: true,
    });
  });
});

describe("resolveStopProviderDelegationsOutcome", () => {
  it("reports a stop only when no task is running afterwards", () => {
    expect(
      resolveStopProviderDelegationsOutcome({ acknowledged: true, activeTaskCount: 0 }),
    ).toEqual({ kind: "stopped", activeTaskCount: 0 });
  });

  it("reports the tasks the tree still shows as running", () => {
    // The receipt calls the root stopped; a task that is still running makes that
    // receipt wrong, and the tree is what the user reads.
    expect(
      resolveStopProviderDelegationsOutcome({ acknowledged: true, activeTaskCount: 2 }),
    ).toEqual({ kind: "partial", activeTaskCount: 2 });
  });

  it("reports an unknown outcome when the provider did not acknowledge the stop", () => {
    expect(
      resolveStopProviderDelegationsOutcome({ acknowledged: false, activeTaskCount: 0 }),
    ).toEqual({ kind: "unknown", activeTaskCount: 0 });
  });
});

describe("requestStopProviderDelegations", () => {
  it("stops nothing when the confirmation is declined", async () => {
    const env = createFakeEnv({ confirmResult: false });

    const status = await requestStopProviderDelegations(
      {
        serverId: "server-1",
        agentId: "agent-1",
        providerLabel: "Minimax Code",
        activeTaskCount: 2,
      },
      env.deps,
    );

    expect(status).toEqual({ kind: "idle" });
    expect(env.recordedStops).toEqual([]);
    expect(env.recordedConfirmInputs).toHaveLength(1);
  });

  it("asks the parent agent to stop and reports what the tree still runs", async () => {
    const env = createFakeEnv({ acknowledged: true, activeTaskCountAfterStop: 1 });

    const status = await requestStopProviderDelegations(
      {
        serverId: "server-1",
        agentId: "agent-1",
        providerLabel: "Minimax Code",
        activeTaskCount: 3,
      },
      env.deps,
    );

    expect(env.recordedStops).toEqual([{ serverId: "server-1", agentId: "agent-1" }]);
    expect(status).toEqual({ kind: "partial", activeTaskCount: 1 });
  });

  it("keeps an unacknowledged stop unknown instead of calling it stopped", async () => {
    const env = createFakeEnv({ acknowledged: false, activeTaskCountAfterStop: 0 });

    const status = await requestStopProviderDelegations(
      {
        serverId: "server-1",
        agentId: "agent-1",
        providerLabel: "Minimax Code",
        activeTaskCount: 1,
      },
      env.deps,
    );

    expect(status).toEqual({ kind: "unknown", activeTaskCount: 0 });
  });

  it("reports a refused stop and leaves the tasks alone", async () => {
    const env = createFakeEnv({
      stopError: new Error("does not support stopping delegated tasks"),
    });

    const status: ProviderDelegationStopStatus = await requestStopProviderDelegations(
      {
        serverId: "server-1",
        agentId: "agent-1",
        providerLabel: "Minimax Code",
        activeTaskCount: 1,
      },
      env.deps,
    );

    expect(env.recordedErrors).toHaveLength(1);
    expect(status).toEqual({
      kind: "failed",
      message: "does not support stopping delegated tasks",
    });
  });
});
