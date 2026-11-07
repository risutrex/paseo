/**
 * Provider processes that are rebuilding a session for an import.
 *
 * Importing a provider session is not a passive read. An ACP provider answers
 * `session/load` by replaying the whole history into memory: for a large
 * mcode session that is minutes of work and gigabytes of RAM. Dropping the
 * caller's promise does not stop any of it, so the process that owns the load
 * registers itself here and the import orchestration can end it.
 */

export type ImportProcessCanceller = () => Promise<void> | void;

const inFlightImports = new Map<string, Set<ImportProcessCanceller>>();

function keyFor(provider: string, sessionId: string): string {
  return `${provider}\0${sessionId}`;
}

/**
 * Registers the process loading `sessionId`. The returned function forgets it,
 * so an initialization that finished is never killed by a later import.
 */
export function registerInFlightImportProcess(
  provider: string,
  sessionId: string,
  cancel: ImportProcessCanceller,
): () => void {
  const key = keyFor(provider, sessionId);
  let cancellers = inFlightImports.get(key);
  if (!cancellers) {
    cancellers = new Set();
    inFlightImports.set(key, cancellers);
  }
  cancellers.add(cancel);
  return () => {
    const current = inFlightImports.get(key);
    if (!current) return;
    current.delete(cancel);
    if (current.size === 0) {
      inFlightImports.delete(key);
    }
  };
}

/**
 * Stops everything currently loading this session and reports how many
 * processes were asked to stop. Zero means the import already finished (or
 * never started a provider), so there is nothing left to clean up.
 */
export async function cancelInFlightImportProcess(
  provider: string,
  sessionId: string,
): Promise<number> {
  const key = keyFor(provider, sessionId);
  const cancellers = inFlightImports.get(key);
  if (!cancellers || cancellers.size === 0) {
    return 0;
  }
  inFlightImports.delete(key);
  const pending = [...cancellers];
  await Promise.allSettled(pending.map((cancel) => Promise.resolve(cancel())));
  return pending.length;
}

/** Test seam: how many processes are loading this session right now. */
export function countInFlightImportProcesses(provider: string, sessionId: string): number {
  return inFlightImports.get(keyFor(provider, sessionId))?.size ?? 0;
}
