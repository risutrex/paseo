import type { WaitForFinishResult } from "@getpaseo/client/internal/daemon-client";
import type { AgentSnapshotPayload } from "@getpaseo/protocol/messages";

/**
 * The provider took the message into its own queue and runs it after the current
 * turn. Nothing ran for this turn, so it has no result: the previous answer
 * belongs to an earlier turn and must never stand in for it.
 */
export interface QueuedObservationResult {
  status: "queued";
  final: AgentSnapshotPayload | null;
  lastMessage: null;
  error: null;
}

export type FinishObservationResult = (WaitForFinishResult | QueuedObservationResult) & {
  reason?: "was canceled" | "host unreachable" | "host removed";
};

/**
 * The settled baseline a host wait returns. A host reports a queued handoff on the
 * client-facing status it already has ("idle") and flags it here, because that wait
 * result also feeds `wait_for_finish`, which has no queued status of its own.
 */
export type FinishObservationBaseline = WaitForFinishResult & { queued?: boolean };

/** Coalesces duplicate checkpoints and retains one terminal result while delivery is pending. */
export function createObservationBuffer(signal: AbortSignal) {
  const delivered = new Set<string>();
  const permissions = new Map<string, FinishObservationResult>();
  const resolvedBeforeInitial = new Set<string>();
  let initialPending = true;
  let terminal: FinishObservationResult | undefined;
  let failure: unknown;
  let wake: (() => void) | undefined;
  const abort = () => {
    failure = signal.reason;
    wake?.();
  };
  signal.addEventListener("abort", abort, { once: true });
  function push(result: FinishObservationResult): void {
    if (result.status === "timeout") return;
    if (result.status === "permission") {
      for (const request of result.final?.pendingPermissions ?? [])
        if (!delivered.has(request.id)) permissions.set(request.id, result);
    } else terminal = result;
    wake?.();
  }
  return {
    pushInitial(result: FinishObservationBaseline) {
      initialPending = false;
      if (!terminal && result.queued) {
        // The wait already saw the provider take the message. Report the queued
        // handoff as the unknown outcome it is, even when this baseline is the first
        // thing delivery sees.
        push({ status: "queued", final: result.final, lastMessage: null, error: null });
      } else if (result.status === "permission" && result.final) {
        const pendingPermissions = result.final.pendingPermissions.filter(
          (request) => !resolvedBeforeInitial.has(request.id),
        );
        if (pendingPermissions.length)
          push({ ...result, final: { ...result.final, pendingPermissions } });
      } else if (
        !terminal &&
        result.final?.status !== "running" &&
        result.final?.status !== "initializing"
      )
        push(result);
      resolvedBeforeInitial.clear();
    },
    push,
    resolved(id: string) {
      if (initialPending) resolvedBeforeInitial.add(id);
      delivered.delete(id);
      permissions.delete(id);
    },
    fail(error: unknown) {
      failure = error;
      wake?.();
    },
    async next(): Promise<FinishObservationResult> {
      while (true) {
        signal.throwIfAborted();
        const first = permissions.entries().next().value;
        if (first) {
          for (const request of first[1].final?.pendingPermissions ?? []) {
            delivered.add(request.id);
            permissions.delete(request.id);
          }
          return first[1];
        }
        if (terminal) {
          const result = terminal;
          terminal = undefined;
          return result;
        }
        if (failure) throw failure;
        await new Promise<void>((resolve) => {
          wake = resolve;
        });
        wake = undefined;
      }
    },
    dispose() {
      signal.removeEventListener("abort", abort);
    },
  };
}
