import type { ConfirmDialogInput } from "@/utils/confirm-dialog";
import { toErrorMessage } from "@/utils/error-messages";

/**
 * What a provider's delegation stop left behind.
 *
 * The provider's receipt is not evidence: it reports the root as stopped even
 * when the tree was idle, and it can leave a child it really stopped out of its
 * list. The delegation rows are the evidence, so the outcome follows the tasks
 * that are still running afterwards.
 */
export type ProviderDelegationStopStatus =
  | { kind: "idle" }
  | { kind: "requesting" }
  | { kind: "stopped"; activeTaskCount: 0 }
  | { kind: "partial"; activeTaskCount: number }
  | { kind: "unknown"; activeTaskCount: number }
  | { kind: "failed"; message: string };

export const IDLE_PROVIDER_DELEGATION_STOP_STATUS: ProviderDelegationStopStatus = { kind: "idle" };

/**
 * Reads the stop from the tree instead of the receipt.
 *
 * An unacknowledged request leaves the outcome unknown: the provider may have
 * stopped everything, or nothing, and only a running task proves work is left.
 */
export function resolveStopProviderDelegationsOutcome(input: {
  acknowledged: boolean;
  activeTaskCount: number;
}): ProviderDelegationStopStatus {
  if (!input.acknowledged) {
    return { kind: "unknown", activeTaskCount: input.activeTaskCount };
  }
  if (input.activeTaskCount > 0) {
    return { kind: "partial", activeTaskCount: input.activeTaskCount };
  }
  return { kind: "stopped", activeTaskCount: 0 };
}

export interface ResolveStopProviderDelegationsDialogInput {
  providerLabel: string;
  activeTaskCount: number;
}

/**
 * The confirmation has to name the scope: this is not a per-task stop. A provider
 * sub-agent has no addressable turn, so the only stop there is takes the parent
 * turn and everything the provider delegated under it.
 */
export function resolveStopProviderDelegationsDialog(
  input: ResolveStopProviderDelegationsDialogInput,
  t: (key: string, options?: Record<string, unknown>) => string,
): ConfirmDialogInput {
  return {
    title: t("subagents.stopProviderDelegationsDialogTitle", { provider: input.providerLabel }),
    message: t("subagents.stopProviderDelegationsDialogMessage", {
      provider: input.providerLabel,
      count: input.activeTaskCount,
    }),
    confirmLabel: t("subagents.stopProviderDelegationsConfirm"),
    cancelLabel: t("common.actions.cancel"),
    destructive: true,
  };
}

export interface StopProviderDelegationsDeps {
  confirm: (input: ConfirmDialogInput) => Promise<boolean>;
  t: (key: string, options?: Record<string, unknown>) => string;
  stopProviderDelegations: (input: {
    serverId: string;
    agentId: string;
  }) => Promise<{ acknowledged: boolean }>;
  /** Tasks the delegation tree still reports as running, read after the stop. */
  countActiveTasks: () => number;
  reportError: (error: unknown) => void;
}

export interface RequestStopProviderDelegationsInput {
  serverId: string;
  agentId: string;
  providerLabel: string;
  activeTaskCount: number;
}

export async function requestStopProviderDelegations(
  input: RequestStopProviderDelegationsInput,
  deps: StopProviderDelegationsDeps,
): Promise<ProviderDelegationStopStatus> {
  const confirmed = await deps.confirm(
    resolveStopProviderDelegationsDialog(
      { providerLabel: input.providerLabel, activeTaskCount: input.activeTaskCount },
      deps.t,
    ),
  );
  if (!confirmed) {
    return { kind: "idle" };
  }
  try {
    const { acknowledged } = await deps.stopProviderDelegations({
      serverId: input.serverId,
      agentId: input.agentId,
    });
    return resolveStopProviderDelegationsOutcome({
      acknowledged,
      activeTaskCount: deps.countActiveTasks(),
    });
  } catch (error) {
    deps.reportError(error);
    return { kind: "failed", message: toErrorMessage(error) };
  }
}
