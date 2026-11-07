import type { z } from "zod";
import type { Logger } from "pino";
import type { ProviderSnapshotManager } from "./provider-snapshot-manager.js";
import type {
  AgentManager,
  ManagedAgent,
  ManagedImportableProviderSession,
} from "./agent-manager.js";
import type { AgentStorage, StoredAgentRecord } from "./agent-storage.js";
import type { AgentPersistenceHandle, AgentProvider } from "./agent-sdk-types.js";
import { ensureAgentLoaded, type AgentLoaderManager } from "./agent-loading.js";
import { cancelInFlightImportProcess } from "./import-session-cancellation.js";
import { unarchiveAgentState } from "./agent-prompt.js";
import { toRecentProviderSessionDescriptorPayload } from "./agent-projections.js";
import type { WorkspaceProvisioningService } from "../session/workspace-provisioning/workspace-provisioning-service.js";
import type { PersistedWorkspaceRecord } from "../workspace-registry.js";
import type {
  FetchRecentProviderSessionsRequestMessage,
  ImportAgentRequestMessageSchema,
  RecentProviderSessionDescriptorPayload,
} from "@getpaseo/protocol/messages";
import { getParentAgentIdFromLabels, PARENT_AGENT_ID_LABEL } from "@getpaseo/protocol/agent-labels";
import { createRealpathAwarePathMatcher } from "../../utils/path.js";

type ImportAgentRequestMessage = z.infer<typeof ImportAgentRequestMessageSchema>;

const METADATA_GENERATION_PROMPT_PREFIX =
  "Generate metadata for a coding agent based on the user prompt.";
const IMPORT_SESSION_SEARCH_SCAN_LIMIT = 500;
export type ImportSessionAgentManager = AgentLoaderManager &
  Pick<
    AgentManager,
    | "archiveSnapshot"
    | "closeAgent"
    | "getTimeline"
    | "importProviderSession"
    | "notifyAgentState"
    | "unarchiveSnapshot"
  >;

const providerSessionImportMutations = new WeakMap<
  ImportSessionAgentManager,
  Map<string, Promise<unknown>>
>();

export interface NormalizedImportAgentRequest {
  provider: AgentProvider;
  providerHandleId: string;
  cwd?: string;
  workspaceId?: string;
  labels?: Record<string, string>;
  requestId: string;
}

/**
 * The client stopped waiting for an import. The provider process loading the
 * session is told to stop, and nothing the import started is kept.
 */
export class ImportCancelledError extends Error {
  constructor(provider: string, providerHandleId: string) {
    super(`Import of ${provider} session ${providerHandleId} was cancelled`);
    this.name = "ImportCancelledError";
  }
}

export class ImportSessionsRequestError extends Error {
  constructor(
    readonly code: string,
    message: string,
  ) {
    super(message);
    this.name = "ImportSessionsRequestError";
  }
}

export interface ListImportableProviderSessionsInput {
  request: FetchRecentProviderSessionsRequestMessage;
  agentManager: Pick<AgentManager, "listAgents" | "listImportableSessions">;
  agentStorage: Pick<AgentStorage, "listAll">;
  providerSnapshotManager: Pick<ProviderSnapshotManager, "getProviderLabel">;
  /**
   * Aborts the listing when the client stopped waiting for it. The provider
   * stops its work, so an abandoned search leaves no probe behind.
   */
  signal?: AbortSignal;
}

export interface ListImportableProviderSessionsResult {
  entries: RecentProviderSessionDescriptorPayload[];
  filteredAlreadyImportedCount: number;
  providerErrors: Array<{ provider: string; message: string }>;
  /** Set when a provider listed sessions without loading their prompt previews. */
  previewsUnavailable?: boolean;
}

export interface ImportProviderSessionInput {
  request: NormalizedImportAgentRequest;
  workspaceProvisioning: Pick<WorkspaceProvisioningService, "runInImportWorkspace">;
  agentManager: ImportSessionAgentManager;
  agentStorage: AgentStorage;
  logger: Logger;
  /**
   * Aborts the import when the client stopped waiting for it. Cancelling is not
   * "stop waiting": the provider process that is rebuilding the session is
   * ended, and an import that raced past completion is rolled back.
   */
  signal?: AbortSignal;
}

export interface ImportProviderSessionResult {
  snapshot: ManagedAgent;
  timelineSize: number;
  createdWorkspace: PersistedWorkspaceRecord | null;
}

interface ImportedProviderSession {
  snapshot: ManagedAgent;
  timelineSize: number;
}

// COMPAT(import-agent-request-v1): accept legacy {provider, sessionId} shape
// alongside the new {providerId, providerHandleId} shape. Old clients
// (< target daemon floor) send the legacy fields. Drop the fallbacks and the
// .optional() in messages.ts when the supported client floor is >= the daemon
// version that ships the new shape (target: 2026-11-08).
export function normalizeImportAgentRequest(
  msg: ImportAgentRequestMessage,
): NormalizedImportAgentRequest | { error: string } {
  const provider = msg.providerId ?? msg.provider;
  const providerHandleId = msg.providerHandleId ?? msg.sessionId;
  if (!provider || !providerHandleId) {
    return { error: "Import requires providerId and providerHandleId" };
  }
  return {
    provider: provider as AgentProvider,
    providerHandleId,
    cwd: msg.cwd,
    workspaceId: msg.workspaceId,
    labels: msg.labels,
    requestId: msg.requestId,
  };
}

export async function listImportableProviderSessions(
  input: ListImportableProviderSessionsInput,
): Promise<ListImportableProviderSessionsResult> {
  const { request, agentManager, agentStorage, providerSnapshotManager } = input;
  const limit = request.limit ?? 20;
  const sinceTimestamp = parseRecentProviderSessionsSince(request.since);
  const providerFilter = request.providers ? new Set(request.providers) : undefined;
  const importedSessions = await collectImportedProviderSessions(
    agentManager,
    agentStorage,
    providerFilter,
  );
  const importedHandles = importedSessions.handles;
  const query = normalizeImportSessionQuery(request.query);
  const listingLimit = query ? IMPORT_SESSION_SEARCH_SCAN_LIMIT : limit + importedSessions.count;

  const listing = await agentManager.listImportableSessions({
    limit: listingLimit,
    ...(query ? { query } : {}),
    ...(query ? { scanLimit: IMPORT_SESSION_SEARCH_SCAN_LIMIT } : {}),
    providerFilter,
    cwd: request.cwd,
    ...(input.signal ? { signal: input.signal } : {}),
  });
  let filteredAlreadyImportedCount = 0;
  const candidates: ManagedImportableProviderSession[] = [];
  const matchesRequestCwd = request.cwd ? createRealpathAwarePathMatcher(request.cwd) : null;
  for (const session of listing.sessions) {
    if (matchesRequestCwd && !matchesRequestCwd(session.cwd)) {
      continue;
    }
    if (sinceTimestamp !== null && session.lastActivityAt.getTime() < sinceTimestamp) {
      continue;
    }
    if (isMetadataGenerationSession(session)) {
      continue;
    }
    if (
      importedHandles.has(toProviderSessionHandleKey(session.provider, session.providerHandleId))
    ) {
      filteredAlreadyImportedCount += 1;
      continue;
    }
    candidates.push(session);
  }

  const entries = candidates
    .sort((a, b) => b.lastActivityAt.getTime() - a.lastActivityAt.getTime())
    .slice(0, limit)
    .map((descriptor) =>
      toRecentProviderSessionDescriptorPayload(descriptor, {
        providerLabel: providerSnapshotManager.getProviderLabel(descriptor.provider),
      }),
    );

  return {
    entries,
    filteredAlreadyImportedCount,
    providerErrors: listing.providerErrors,
    ...(listing.previewsUnavailable ? { previewsUnavailable: true } : {}),
  };
}

function normalizeImportSessionQuery(query: string | undefined): string | null {
  const normalized = query?.trim().toLowerCase();
  return normalized ? normalized : null;
}

export async function importProviderSession(
  input: ImportProviderSessionInput,
): Promise<ImportProviderSessionResult> {
  const cwd = input.request.cwd;
  if (!cwd) {
    throw new Error("Import requires cwd from the selected provider session");
  }
  const { provider, providerHandleId } = input.request;
  const signal = input.signal;
  if (signal?.aborted) {
    throw new ImportCancelledError(provider, providerHandleId);
  }
  const key = await resolveProviderSessionImportMutationKey(input);
  const work = serializeProviderSessionImport(input.agentManager, key, async () => {
    const placement = await input.workspaceProvisioning.runInImportWorkspace(
      { cwd, requestedWorkspaceId: input.request.workspaceId },
      (workspace) => importProviderSessionNow(input, cwd, workspace.workspaceId),
    );
    return { ...placement.value, createdWorkspace: placement.createdWorkspace };
  });
  if (!signal) {
    return await work;
  }
  return await raceImportCancellation(input, work, signal);
}

/**
 * Answers a cancelled import immediately and stops the provider, which is what
 * `Promise.race` alone cannot do: ACP keeps loading the session in its process
 * until that process is ended.
 */
function raceImportCancellation(
  input: ImportProviderSessionInput,
  work: Promise<ImportProviderSessionResult>,
  signal: AbortSignal,
): Promise<ImportProviderSessionResult> {
  const { provider, providerHandleId } = input.request;
  return new Promise<ImportProviderSessionResult>((resolve, reject) => {
    let settled = false;
    const cancel = () => {
      if (settled) return;
      settled = true;
      signal.removeEventListener("abort", cancel);
      // The provider holds the session in memory while it replays the history;
      // ending the process is the part of cancelling that frees anything.
      void cancelInFlightImportProcess(provider, providerHandleId).catch((error: unknown) => {
        input.logger.warn(
          { err: error, provider, providerHandleId },
          "Failed to stop the provider process importing a session",
        );
      });
      reject(new ImportCancelledError(provider, providerHandleId));
    };
    signal.addEventListener("abort", cancel, { once: true });
    work.then(
      (result) => {
        signal.removeEventListener("abort", cancel);
        if (settled || signal.aborted) {
          // The import finished after the client gave up, so the agent it
          // registered is an import nobody asked for: drop it instead of
          // leaving a half-imported agent behind.
          return discardCancelledImport(input, result).catch(() => undefined);
        }
        settled = true;
        resolve(result);
        return undefined;
      },
      (error: unknown) => {
        signal.removeEventListener("abort", cancel);
        if (!settled) {
          settled = true;
          reject(error);
        }
        return undefined;
      },
    );
  });
}

async function discardCancelledImport(
  input: ImportProviderSessionInput,
  result: ImportProviderSessionResult,
): Promise<void> {
  const agentId = result.snapshot.id;
  try {
    if (input.agentManager.getAgent(agentId)) {
      await input.agentManager.closeAgent(agentId);
    }
    await input.agentManager.archiveSnapshot(agentId, new Date().toISOString());
  } catch (error) {
    input.logger.warn(
      { err: error, agentId },
      "Failed to discard the agent registered by a cancelled import",
    );
  }
}

async function importProviderSessionNow(
  input: ImportProviderSessionInput,
  cwd: string,
  workspaceId: string,
): Promise<ImportedProviderSession> {
  const { provider, providerHandleId, labels } = input.request;
  assertImportNotCancelled(input);

  const matchingRecords = await input.agentStorage.listByProviderSession(
    provider,
    providerHandleId,
  );
  const activeRecord = matchingRecords.find((record) => !record.archivedAt);
  if (activeRecord) {
    throw new Error(`Provider session is already imported: ${providerHandleId}`);
  }
  const archivedRecord = matchingRecords.find((record) => record.archivedAt);
  if (archivedRecord?.persistence && archivedRecord.archivedAt) {
    if (!createRealpathAwarePathMatcher(cwd)(archivedRecord.cwd)) {
      throw new Error(`Provider session cwd does not match import cwd: ${providerHandleId}`);
    }
    const requestedParentAgentId = getParentAgentIdFromLabels(input.request.labels);
    const labelPatch: Record<string, string | null> = { ...input.request.labels };
    if (
      Object.hasOwn(archivedRecord.labels, PARENT_AGENT_ID_LABEL) ||
      Object.hasOwn(input.request.labels ?? {}, PARENT_AGENT_ID_LABEL)
    ) {
      labelPatch[PARENT_AGENT_ID_LABEL] = requestedParentAgentId;
    }
    await unarchiveAgentState(input.agentStorage, input.agentManager, archivedRecord.id, {
      workspaceId,
      labels: Object.keys(labelPatch).length > 0 ? labelPatch : undefined,
    });
    try {
      assertImportNotCancelled(input);
      const snapshot = await ensureAgentLoaded(archivedRecord.id, {
        agentManager: input.agentManager,
        agentStorage: input.agentStorage,
        logger: input.logger,
      });
      return {
        snapshot,
        timelineSize: input.agentManager.getTimeline(snapshot.id).length,
      };
    } catch (error) {
      await rollbackArchivedImport(input, archivedRecord, archivedRecord.archivedAt);
      throw error;
    }
  }

  // The session/load behind this call is the expensive part of an import, so a
  // client that already gave up must not start it.
  assertImportNotCancelled(input);
  const snapshot = await input.agentManager.importProviderSession({
    provider,
    providerHandleId,
    cwd,
    workspaceId,
    labels,
  });
  await unarchiveAgentState(input.agentStorage, input.agentManager, snapshot.id);

  return {
    snapshot,
    timelineSize: input.agentManager.getTimeline(snapshot.id).length,
  };
}

function assertImportNotCancelled(input: ImportProviderSessionInput): void {
  if (input.signal?.aborted) {
    throw new ImportCancelledError(input.request.provider, input.request.providerHandleId);
  }
}

async function serializeProviderSessionImport<T>(
  agentManager: ImportSessionAgentManager,
  key: string,
  operation: () => Promise<T>,
): Promise<T> {
  let mutations = providerSessionImportMutations.get(agentManager);
  if (!mutations) {
    mutations = new Map();
    providerSessionImportMutations.set(agentManager, mutations);
  }

  const previous = mutations.get(key) ?? Promise.resolve();
  const next = previous.catch(() => undefined).then(operation);
  mutations.set(key, next);
  try {
    return await next;
  } finally {
    if (mutations.get(key) === next) {
      mutations.delete(key);
    }
  }
}

async function resolveProviderSessionImportMutationKey(
  input: ImportProviderSessionInput,
): Promise<string> {
  const matchingRecord = (
    await input.agentStorage.listByProviderSession(
      input.request.provider,
      input.request.providerHandleId,
    )
  ).at(0);
  return matchingRecord
    ? `agent\0${matchingRecord.id}`
    : `handle\0${toProviderSessionHandleKey(
        input.request.provider,
        input.request.providerHandleId,
      )}`;
}

async function rollbackArchivedImport(
  input: ImportProviderSessionInput,
  archivedRecord: StoredAgentRecord,
  archivedAt: string,
): Promise<void> {
  try {
    if (input.agentManager.getAgent(archivedRecord.id)) {
      await input.agentManager.closeAgent(archivedRecord.id);
    }
    await input.agentManager.archiveSnapshot(archivedRecord.id, archivedAt);
  } catch (error) {
    input.logger.error(
      { err: error, agentId: archivedRecord.id },
      "Failed to re-archive provider session after import failure",
    );
  }

  try {
    await input.agentStorage.upsert(archivedRecord);
  } catch (error) {
    input.logger.error(
      { err: error, agentId: archivedRecord.id },
      "Failed to restore archived agent record after import failure",
    );
  }
}

function parseRecentProviderSessionsSince(since: string | undefined): number | null {
  if (!since) {
    return null;
  }
  const timestamp = Date.parse(since);
  if (Number.isNaN(timestamp)) {
    throw new ImportSessionsRequestError("invalid_since", "Invalid recent provider sessions since");
  }
  return timestamp;
}

async function collectImportedProviderSessions(
  agentManager: Pick<AgentManager, "listAgents">,
  agentStorage: Pick<AgentStorage, "listAll">,
  providerFilter: Set<string> | undefined,
): Promise<{ handles: Set<string>; count: number }> {
  const handles = new Set<string>();
  const sessions = new Set<string>();
  // Explicit provider-history import deduplicates against archived as well as active sessions.
  const records = await agentStorage.listAll();
  const storedRecordsById = new Map(records.map((record) => [record.id, record]));

  const collect = (
    provider: AgentProvider | StoredAgentRecord["provider"] | string,
    persistence: AgentPersistenceHandle | null | undefined,
  ) => {
    if (!persistence || (providerFilter && !providerFilter.has(provider))) return;
    sessions.add(toProviderSessionHandleKey(provider, persistence.sessionId));
    collectProviderSessionHandleKeys(handles, provider, persistence);
  };

  for (const agent of agentManager.listAgents()) {
    if (storedRecordsById.get(agent.id)?.archivedAt) {
      continue;
    }
    collect(agent.provider, agent.persistence);
  }

  for (const record of records) {
    if (record.archivedAt) {
      continue;
    }
    collect(record.provider, record.persistence);
  }

  return { handles, count: sessions.size };
}

function toProviderSessionHandleKey(provider: string, providerHandleId: string): string {
  return `${provider}\0${providerHandleId}`;
}

export function isMetadataGenerationSession(input: { firstPromptPreview: string | null }): boolean {
  return (
    input.firstPromptPreview?.trimStart().startsWith(METADATA_GENERATION_PROMPT_PREFIX) ?? false
  );
}

function collectProviderSessionHandleKeys(
  target: Set<string>,
  provider: AgentProvider | StoredAgentRecord["provider"] | string,
  persistence: AgentPersistenceHandle | null | undefined,
): void {
  if (!persistence) {
    return;
  }

  target.add(toProviderSessionHandleKey(provider, persistence.sessionId));
  if (persistence.nativeHandle) {
    target.add(toProviderSessionHandleKey(provider, persistence.nativeHandle));
  }
}
