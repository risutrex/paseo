import React, { type ReactNode, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Pressable, type PressableStateCallbackType, Text, View } from "react-native";
import {
  CancelledError,
  keepPreviousData,
  useMutation,
  useQueries,
  useQueryClient,
} from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import type {
  DaemonClient,
  FetchRecentProviderSessionEntry,
} from "@getpaseo/client/internal/daemon-client";
import type { AgentProvider } from "@getpaseo/protocol/agent-types";
import { ChevronDown, Inbox, Layers, RotateCw } from "lucide-react-native";
import { StyleSheet, useUnistyles } from "react-native-unistyles";
import { AdaptiveModalSheet, type SheetHeader } from "@/components/adaptive-modal-sheet";
import { Button } from "@/components/ui/button";
import { ImportSessionList } from "@/components/import-session-list";
import { LoadingSpinner } from "@/components/ui/loading-spinner";
import { Combobox, ComboboxItem, type ComboboxOption } from "@/components/ui/combobox";
import { useProviderIcon, useProviderIcons } from "@/components/provider-icons";
import { useTimeAgo } from "@/hooks/use-time-ago";
import { useDebouncedValue } from "@/hooks/use-debounced-value";
import { useProvidersSnapshot } from "@/hooks/use-providers-snapshot";
import { useHostProjects } from "@/projects/host-projects";
import { useHostFeature } from "@/runtime/host-features";
import { useHosts } from "@/runtime/host-runtime";
import { i18n } from "@/i18n/i18next";
import {
  aggregateSessionEntries,
  ALL_FILTER_VALUE,
  buildProviderLabelMap,
  collectProviderErrorRows,
  computeEmptyState,
  type DirectoryProject,
  formatDirectoryLabel,
  getPromptPreview,
  getSessionTitle,
  hasMoreSessions,
  hasUnavailablePreviews,
  resolveDirectoryLabel,
  nextPageLimit,
  PER_PROVIDER_LIMIT,
  type ProviderErrorRow,
  resolveImportTarget,
  resolveProvidersToFetch,
  requiresImportSessionsHostUpgrade,
  sumFilteredAlreadyImportedCount,
} from "@/components/import-session-sheet-view-model";

const IMPORT_SHEET_SNAP_POINTS = ["70%", "92%"];
const DISABLED_ACCESSIBILITY_STATE = { disabled: true };
/** Long enough that a typed word is one request, short enough to feel live. */
const SEARCH_DEBOUNCE_MS = 200;

/**
 * A scope the user picked by hand: the current workspace or every directory on
 * the host.
 */
type ScopePreference = "workspace" | "all";

type RecentProviderSessionsClient = Pick<
  DaemonClient,
  "fetchRecentProviderSessions" | "importAgent"
>;

type ImportedAgent = Awaited<ReturnType<RecentProviderSessionsClient["importAgent"]>>;

interface ImportSessionSheetProps {
  visible: boolean;
  client: RecentProviderSessionsClient | null;
  serverId: string | null;
  cwd?: string | null;
  workspaceId?: string | null;
  onClose: () => void;
  /** The agent belongs to the workspace the sheet was opened from; open it here. */
  onImportedAgent?: (agentId: string) => void;
  /** The agent belongs to its own workspace; open that workspace and navigate. */
  onImported?: (agent: ImportedAgent) => void;
}

type RecentSessionsResponse = Awaited<
  ReturnType<RecentProviderSessionsClient["fetchRecentProviderSessions"]>
>;

type SessionsQueryKey = ReadonlyArray<string | number | null>;

function buildSessionsQueryKey(input: {
  serverId: string | null;
  cwd: string | null;
  query: string;
  limit: number;
  provider?: string;
}): SessionsQueryKey {
  return [
    "recent-provider-sessions",
    // The host is part of the identity of the answer: two hosts can serve the
    // same directory, and a key without it mixes their listings.
    input.serverId,
    input.cwd,
    input.query,
    input.limit,
    ...(input.provider === undefined ? [] : [input.provider]),
  ];
}

interface SessionsQueryConfig {
  queryKey: SessionsQueryKey;
  enabled: boolean;
  // A provider that cannot answer stays broken until the user asks again. React
  // Query's default retry would keep the header spinner turning and pile up
  // in-flight daemon requests behind a dead provider (#2512).
  retry: false;
  // The app-wide default keeps a query fresh forever and never refetches it
  // on mount, which is wrong for a listing that changes while the user works:
  // sessions come and go, and a sheet opened later must not show what the
  // host knew hours ago. A listing is therefore always stale, so opening the
  // sheet asks the host again; the cached rows stay on screen meanwhile.
  staleTime: number;
  refetchOnMount: boolean | "always";
  placeholderData: typeof keepPreviousData;
  queryFn: (context: { signal?: AbortSignal }) => Promise<RecentSessionsResponse>;
}

/**
 * A listing is never considered fresh: the provider's session list changes
 * while the user works, and a sheet opened later must not show a list from
 * hours ago.
 */
const IMPORT_SESSIONS_STALE_TIME_MS = 0;

function buildSessionsQueriesConfig(args: {
  serverId: string | null;
  providersToFetch: AgentProvider[] | null;
  visible: boolean;
  client: RecentProviderSessionsClient | null;
  cwd: string | null;
  query: string;
  limit: number;
  hostDisconnectedMessage?: string;
}): SessionsQueryConfig[] {
  const {
    serverId,
    providersToFetch,
    visible,
    client,
    cwd,
    query,
    limit,
    hostDisconnectedMessage,
  } = args;
  if (providersToFetch === null) return [];
  const enabled = visible && Boolean(client);
  return providersToFetch.map((provider) => ({
    queryKey: buildSessionsQueryKey({ serverId, cwd, query, limit, provider }),
    enabled,
    retry: false as const,
    staleTime: IMPORT_SESSIONS_STALE_TIME_MS,
    refetchOnMount: "always",
    placeholderData: keepPreviousData,
    queryFn: async ({ signal }: { signal?: AbortSignal }) => {
      if (!client) {
        throw new Error(hostDisconnectedMessage ?? i18n.t("workspace.terminal.hostDisconnected"));
      }
      try {
        return await client.fetchRecentProviderSessions(
          {
            ...(cwd ? { cwd } : {}),
            providers: [provider],
            limit,
            ...(query ? { query } : {}),
          },
          // A newer request replaces this one. Handing the signal over tells the
          // daemon to stop the listing, so an abandoned search leaves no
          // provider process behind.
          signal,
        );
      } catch (error) {
        if (signal?.aborted) {
          // The request was replaced, not failed. Reporting it as an error
          // would show the user a failure for a search they did not wait for.
          throw new CancelledError();
        }
        throw error;
      }
    },
  }));
}

interface SheetStatusMessagesProps {
  isClientReady: boolean;
  isSnapshotUnsupported: boolean;
  hasNoImportableProviders: boolean;
  isLoadingSessions: boolean;
  hasRows: boolean;
  importErrored: boolean;
}

function SheetStatusMessages({
  isClientReady,
  isSnapshotUnsupported,
  hasNoImportableProviders,
  isLoadingSessions,
  hasRows,
  importErrored,
}: SheetStatusMessagesProps) {
  const { theme } = useUnistyles();
  const { t } = useTranslation();
  if (!isClientReady) {
    return <Text style={styles.statusText}>{t("importSession.status.connectHost")}</Text>;
  }
  if (isSnapshotUnsupported) {
    return <Text style={styles.statusText}>{t("importSession.status.updateHost")}</Text>;
  }
  return (
    <>
      {hasNoImportableProviders ? (
        <Text style={styles.statusText}>{t("importSession.status.noProviders")}</Text>
      ) : null}
      {isLoadingSessions && !hasRows ? (
        <View style={styles.statusRow}>
          <LoadingSpinner color={theme.colors.foregroundMuted} />
          <Text style={styles.statusText}>{t("importSession.status.loading")}</Text>
        </View>
      ) : null}
      {importErrored ? (
        <Text style={styles.statusText}>{t("importSession.status.failedImport")}</Text>
      ) : null}
    </>
  );
}

function ProviderErrorBanner({
  rows,
  onRetry,
}: {
  rows: ReadonlyArray<ProviderErrorRow>;
  onRetry: (provider: string) => void;
}) {
  return (
    <View style={styles.errorBanner} testID="import-session-provider-errors">
      {rows.map((row) => (
        <ProviderErrorBannerRow key={row.provider} row={row} onRetry={onRetry} />
      ))}
    </View>
  );
}

function ProviderErrorBannerRow({
  row,
  onRetry,
}: {
  row: ProviderErrorRow;
  onRetry: (provider: string) => void;
}) {
  const { t } = useTranslation();
  const handleRetry = useCallback(() => onRetry(row.provider), [onRetry, row.provider]);
  return (
    <View style={styles.errorRow}>
      <Text style={styles.errorText}>
        {t("importSession.status.failedProvider", { provider: row.label })}
      </Text>
      <Button
        variant="ghost"
        size="sm"
        onPress={handleRetry}
        testID={`import-session-retry-${row.provider}`}
      >
        {t("common.actions.retry")}
      </Button>
    </View>
  );
}

function RefreshAction({ isRefreshing, onPress }: { isRefreshing: boolean; onPress: () => void }) {
  const { theme } = useUnistyles();
  const { t } = useTranslation();
  const pressableStyle = useCallback(
    ({ pressed }: PressableStateCallbackType) => [
      styles.refreshButton,
      pressed && styles.refreshButtonPressed,
    ],
    [],
  );
  return (
    <Pressable
      onPress={onPress}
      disabled={isRefreshing}
      accessibilityLabel={t("importSession.actions.refresh")}
      accessibilityRole="button"
      testID="import-session-refresh"
      style={pressableStyle}
    >
      <View style={styles.refreshIconSlot}>
        {isRefreshing ? (
          <LoadingSpinner color={theme.colors.foregroundMuted} />
        ) : (
          <RotateCw size={16} color={theme.colors.foregroundMuted} />
        )}
      </View>
    </Pressable>
  );
}

function ScopeSubtitle({
  hostLabel,
  isScoped,
  canScope,
  onShowAll,
  onScopeToWorkspace,
}: {
  hostLabel: string;
  isScoped: boolean;
  /** Without a workspace directory there is nothing to narrow the list to. */
  canScope: boolean;
  onShowAll: () => void;
  onScopeToWorkspace: () => void;
}) {
  const { t } = useTranslation();
  return (
    <View style={styles.subtitleRow}>
      <Text style={styles.subtitleText} numberOfLines={1} testID="import-session-scope">
        {isScoped
          ? t("importSession.scope.workspace")
          : t("importSession.scope.host", { host: hostLabel })}
      </Text>
      {canScope ? (
        <Button
          variant="ghost"
          size="sm"
          onPress={isScoped ? onShowAll : onScopeToWorkspace}
          testID={isScoped ? "import-session-show-all" : "import-session-scope-workspace"}
          style={styles.subtitleAction}
        >
          {isScoped
            ? t("importSession.actions.showAll")
            : t("importSession.actions.scopeToWorkspace")}
        </Button>
      ) : null}
    </View>
  );
}

/**
 * The host answered from an index that holds names and folders but no prompt
 * text (generic ACP providers such as minimax-code or cursor load previews
 * lazily). The search therefore never looked inside prompts, and a session the
 * user knows by its wording can legitimately be missing from the list.
 */
function PreviewsUnavailableNote() {
  const { t } = useTranslation();
  return (
    <Text style={styles.previewsNote} testID="import-session-previews-unavailable">
      {t("importSession.previews.unavailable")}
    </Text>
  );
}

/**
 * Warns before the expensive half of an import. ACP providers answer an import
 * with session/load, which rebuilds the whole conversation in memory; a large
 * session takes minutes and gigabytes of RAM. Nothing reports its size —
 * session/list has none, and bytes on disk are not a RAM forecast — so the
 * warning states the risk instead of inventing an estimate.
 */
function ImportCostWarning({
  onConfirm,
  onCancel,
}: {
  onConfirm: () => void;
  onCancel: () => void;
}) {
  const { t } = useTranslation();
  return (
    <View style={styles.importNotice} testID="import-session-import-warning">
      <Text style={styles.importNoticeTitle}>{t("importSession.warning.title")}</Text>
      <Text style={styles.importNoticeBody}>{t("importSession.warning.body")}</Text>
      <View style={styles.importNoticeActions}>
        <Button
          variant="ghost"
          size="sm"
          onPress={onCancel}
          testID="import-session-import-warning-cancel"
        >
          {t("importSession.warning.cancel")}
        </Button>
        <Button
          variant="default"
          size="sm"
          onPress={onConfirm}
          testID="import-session-import-warning-confirm"
        >
          {t("importSession.warning.confirm")}
        </Button>
      </View>
    </View>
  );
}

/**
 * The import is running. It says so and counts the seconds, because the honest
 * signal during a long load is "still working", and it offers the one action
 * that matters: stopping it.
 */
function ImportProgressNote({
  elapsedSeconds,
  onCancel,
}: {
  elapsedSeconds: number;
  onCancel: () => void;
}) {
  const { theme } = useUnistyles();
  const { t } = useTranslation();
  return (
    <View style={styles.importNotice} testID="import-session-import-progress">
      <View style={styles.importProgressHeader}>
        <LoadingSpinner color={theme.colors.foregroundMuted} />
        <Text style={styles.importNoticeTitle}>{t("importSession.progress.title")}</Text>
        <Text style={styles.rowMeta} testID="import-session-import-elapsed">
          {t("importSession.progress.elapsed", { seconds: elapsedSeconds })}
        </Text>
      </View>
      <Text style={styles.importNoticeBody}>{t("importSession.progress.body")}</Text>
      <View style={styles.importNoticeActions}>
        <Button variant="ghost" size="sm" onPress={onCancel} testID="import-session-import-cancel">
          {t("importSession.progress.cancel")}
        </Button>
      </View>
    </View>
  );
}

/** The client stopped the import; the provider is no longer loading it. */
function ImportCancelledNote() {
  const { t } = useTranslation();
  return (
    <Text style={styles.importCancelledNote} testID="import-session-import-cancelled">
      {t("importSession.cancelled")}
    </Text>
  );
}

function isImportCancelledError(error: unknown): boolean {
  // daemon-client answers a stopped import with an AbortError, whether the
  // caller aborted first or the daemon confirmed the cancellation.
  return error instanceof Error && error.name === "AbortError";
}

function SheetEmptyState({ title }: { title: string }) {
  const { theme } = useUnistyles();
  return (
    <View style={styles.emptyState} testID="import-session-empty-state">
      <View style={styles.emptyStateIcon}>
        <Inbox size={theme.iconSize.lg} color={theme.colors.foregroundMuted} strokeWidth={1.5} />
      </View>
      <Text style={styles.emptyStateTitle}>{title}</Text>
    </View>
  );
}

function ImportSessionSheetRow({
  serverId,
  entry,
  disabled,
  importing,
  folder,
  onImportSession,
}: {
  serverId: string | null;
  entry: FetchRecentProviderSessionEntry;
  disabled: boolean;
  importing: boolean;
  /** The row's directory, shown only when rows can come from more than one. */
  folder: string | null;
  onImportSession: (entry: FetchRecentProviderSessionEntry) => void;
}) {
  const { theme } = useUnistyles();
  const { t } = useTranslation();
  const title = getSessionTitle(entry);
  const promptPreview = getPromptPreview(entry);
  const ProviderIcon = useProviderIcon(entry.providerId, serverId);
  const accessibilityState = useMemo(
    () => (disabled ? DISABLED_ACCESSIBILITY_STATE : undefined),
    [disabled],
  );
  const handlePress = useCallback(() => {
    onImportSession(entry);
  }, [entry, onImportSession]);
  const pressableStyle = useCallback(
    ({ pressed, hovered = false }: PressableStateCallbackType & { hovered?: boolean }) => [
      styles.row,
      Boolean(hovered) && styles.rowHovered,
      pressed && styles.rowPressed,
    ],
    [],
  );

  return (
    <Pressable
      disabled={disabled}
      onPress={handlePress}
      accessibilityRole="button"
      accessibilityState={accessibilityState}
      style={pressableStyle}
      testID={`import-session-session-${entry.providerId}-${entry.providerHandleId}`}
    >
      <View style={styles.rowIconWrap}>
        <ProviderIcon size={theme.iconSize.md} color={theme.colors.foregroundMuted} />
      </View>
      <View style={styles.rowContent}>
        <View style={styles.rowHeader}>
          <Text style={styles.rowTitle} numberOfLines={1}>
            {title}
          </Text>
          {importing ? (
            <Text style={styles.rowMeta}>{t("importSession.row.importing")}</Text>
          ) : (
            <ImportSessionActivityTime date={entry.lastActivityAt} />
          )}
        </View>
        <Text style={styles.rowPreview} numberOfLines={2}>
          {promptPreview}
        </Text>
        {folder ? (
          <Text
            style={styles.rowFolder}
            numberOfLines={1}
            testID={`import-session-row-folder-${entry.providerId}-${entry.providerHandleId}`}
          >
            {folder}
          </Text>
        ) : null}
      </View>
    </Pressable>
  );
}

function ImportSessionActivityTime({ date }: { date: string }) {
  const label = useTimeAgo(new Date(date));
  return <Text style={styles.rowMeta}>{label}</Text>;
}

function getSessionRowKey(entry: FetchRecentProviderSessionEntry): string {
  return `${entry.providerId}:${entry.providerHandleId}`;
}

function hasListHeader(
  input: SheetStatusMessagesProps & {
    showFilter: boolean;
    hasProviderErrors: boolean;
    hasPreviewsUnavailable: boolean;
  },
): boolean {
  return (
    input.showFilter ||
    !input.isClientReady ||
    input.isSnapshotUnsupported ||
    input.hasNoImportableProviders ||
    (input.isLoadingSessions && !input.hasRows) ||
    input.importErrored ||
    input.hasProviderErrors ||
    input.hasPreviewsUnavailable
  );
}

function SessionRows({
  visible,
  onClose,
  header,
  listHeader,
  listFooter,
  serverId,
  entries,
  disabled,
  importingSessionKey,
  resolveFolder,
  onImportSession,
}: {
  visible: boolean;
  onClose: () => void;
  header: SheetHeader;
  listHeader: ReactNode;
  listFooter: ReactNode;
  serverId: string | null;
  entries: ReadonlyArray<FetchRecentProviderSessionEntry>;
  disabled: boolean;
  importingSessionKey: string | null;
  resolveFolder: (entry: FetchRecentProviderSessionEntry) => string | null;
  onImportSession: (entry: FetchRecentProviderSessionEntry) => void;
}) {
  // Keep short results content-sized; only long lists need a bounded viewport.
  const virtualized = ImportSessionList !== null && entries.length > 8;
  const renderSessionRow = useCallback(
    (entry: FetchRecentProviderSessionEntry) => (
      <ImportSessionSheetRow
        key={getSessionRowKey(entry)}
        serverId={serverId}
        entry={entry}
        disabled={disabled}
        importing={importingSessionKey === getSessionRowKey(entry)}
        folder={resolveFolder(entry)}
        onImportSession={onImportSession}
      />
    ),
    [serverId, disabled, importingSessionKey, resolveFolder, onImportSession],
  );
  const renderItem = useCallback(
    (item: FetchRecentProviderSessionEntry) => (
      <View style={styles.listRow}>{renderSessionRow(item)}</View>
    ),
    [renderSessionRow],
  );
  const virtualHeader = useMemo(
    () => (listHeader ? <View style={styles.listHeader}>{listHeader}</View> : null),
    [listHeader],
  );
  const virtualFooter = useMemo(
    () => <View style={styles.listFooter}>{listFooter}</View>,
    [listFooter],
  );

  return (
    <AdaptiveModalSheet
      visible={visible}
      onClose={onClose}
      header={header}
      testID="import-session-sheet"
      desktopMaxWidth={560}
      desktopHeight={virtualized ? "85%" : undefined}
      snapPoints={IMPORT_SHEET_SNAP_POINTS}
      scrollable={!virtualized}
      contentStyle={virtualized ? styles.listBody : undefined}
    >
      {ImportSessionList && virtualized ? (
        <View style={styles.list}>
          <ImportSessionList
            data={entries}
            keyExtractor={getSessionRowKey}
            renderItem={renderItem}
            header={virtualHeader}
            footer={virtualFooter}
          />
        </View>
      ) : (
        <>
          {listHeader}
          {entries.length > 0 ? (
            <View style={styles.smallList}>{entries.map(renderSessionRow)}</View>
          ) : null}
          {listFooter}
        </>
      )}
    </AdaptiveModalSheet>
  );
}

export function ImportSessionSheet({
  visible,
  client,
  serverId,
  cwd,
  workspaceId,
  onClose,
  onImportedAgent,
  onImported,
}: ImportSessionSheetProps) {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const { theme } = useUnistyles();

  // The list starts on the sheet's own workspace: a listing scoped to it is the
  // only one the daemon matched with realpaths resolved, so it is the only one
  // where "this row is ours" can be answered exactly. Widening to every
  // directory is the user's own choice, and the sheet keeps it until they
  // narrow it back.
  const [scopePreference, setScopePreference] = useState<ScopePreference | null>(null);
  const [searchInput, setSearchInput] = useState("");
  const [pageLimit, setPageLimit] = useState(PER_PROVIDER_LIMIT);
  const [selectedProvider, setSelectedProvider] = useState<string>(ALL_FILTER_VALUE);

  const supportsSearch = useHostFeature(serverId, "importSessionSearch");
  const query = useDebouncedValue(supportsSearch ? searchInput : "", SEARCH_DEBOUNCE_MS).trim();

  const isSearchActive = query.length > 0;
  const isShowingAllDirectories = scopePreference === "all";
  // The sheet's own `cwd` still decides where an imported agent lands; the
  // scope only changes which sessions are listed.
  // The entry waiting for the "this import is expensive" confirmation, and the
  // controller that stops a running import on the provider side.
  const [pendingWarning, setPendingWarning] = useState<FetchRecentProviderSessionEntry | null>(
    null,
  );
  const [importElapsedSeconds, setImportElapsedSeconds] = useState(0);
  const importAbortRef = useRef<AbortController | null>(null);

  const scopeCwd = isShowingAllDirectories ? null : (cwd ?? null);

  useEffect(() => {
    if (visible) return;
    setScopePreference(null);
    setSearchInput("");
    // A warning for a row the user no longer sees must not survive a reopen.
    setPendingWarning(null);
  }, [visible]);

  // A narrower or wider list starts at page one; keeping a grown limit would
  // fetch 200 rows for a query that matches three.
  useEffect(() => {
    setPageLimit(PER_PROVIDER_LIMIT);
  }, [query, scopeCwd]);

  const { entries: snapshotEntries, supportsSnapshot } = useProvidersSnapshot(serverId, {
    cwd: scopeCwd,
    enabled: visible,
  });
  const supportsWorkspaceTarget = useHostFeature(serverId, "importSessionWorkspaceTarget");
  const requiresHostUpgrade = requiresImportSessionsHostUpgrade({
    supportsSnapshot,
    workspaceId,
    supportsWorkspaceTarget,
  });

  const providersToFetch = useMemo(
    () => (requiresHostUpgrade ? null : resolveProvidersToFetch(supportsSnapshot, snapshotEntries)),
    [requiresHostUpgrade, supportsSnapshot, snapshotEntries],
  );

  const providerLabelById = useMemo(
    () => buildProviderLabelMap(snapshotEntries),
    [snapshotEntries],
  );

  const sessionsQueryRoot = useMemo(
    () => ["recent-provider-sessions", serverId, scopeCwd] as const,
    [serverId, scopeCwd],
  );

  // Hiding the sheet keeps its observers mounted, and turning `enabled` off does
  // not stop a fetch already in flight (TanStack Query v5). Cancelling aborts
  // the query's signal, which is what tells the daemon to stop walking
  // directories for a list nobody is looking at any more.
  useEffect(() => {
    if (visible) return;
    void queryClient.cancelQueries({ queryKey: sessionsQueryRoot, exact: false });
  }, [visible, queryClient, sessionsQueryRoot]);

  const queriesConfig = useMemo(
    () =>
      buildSessionsQueriesConfig({
        serverId,
        providersToFetch,
        visible,
        client,
        cwd: scopeCwd,
        query,
        limit: pageLimit,
        hostDisconnectedMessage: t("workspace.terminal.hostDisconnected"),
      }),
    [serverId, providersToFetch, visible, client, scopeCwd, query, pageLimit, t],
  );

  const queries = useQueries({ queries: queriesConfig });

  const aggregatedEntries = useMemo(() => aggregateSessionEntries(queries), [queries]);
  // A provider that listed without prompt previews cannot read a session
  // cheaply either: its import has to rebuild the history on the provider side.
  // Providers that preview cheaply import cheaply, so they get no warning.
  const expensiveImportProviders = useMemo(() => {
    const providers = new Set<string>();
    providersToFetch?.forEach((provider, index) => {
      if (queries[index]?.data?.previewsUnavailable === true) {
        providers.add(provider);
      }
    });
    return providers;
  }, [providersToFetch, queries]);
  // Any provider answering without prompt previews makes that provider's rows
  // a names-and-folders match, so the note is a property of the whole list.
  // Without a search there is nothing to explain: no prompt was looked for.
  const previewsUnavailable = useMemo(
    () => isSearchActive && hasUnavailablePreviews(queries),
    [isSearchActive, queries],
  );
  const totalAlreadyImportedCount = useMemo(
    () => sumFilteredAlreadyImportedCount(queries),
    [queries],
  );

  const filterProviders = useMemo(() => [...(providersToFetch ?? [])].sort(), [providersToFetch]);
  const [isFilterOpen, setIsFilterOpen] = useState(false);
  const filterAnchorRef = useRef<View>(null);

  useEffect(() => {
    if (
      !visible ||
      (selectedProvider !== ALL_FILTER_VALUE && !filterProviders.includes(selectedProvider))
    ) {
      setSelectedProvider(ALL_FILTER_VALUE);
    }
  }, [visible, filterProviders, selectedProvider]);

  const visibleEntries = useMemo(() => {
    if (selectedProvider === ALL_FILTER_VALUE) return aggregatedEntries;
    return aggregatedEntries.filter((entry) => entry.providerId === selectedProvider);
  }, [aggregatedEntries, selectedProvider]);

  const projectServerIds = useMemo(() => (serverId ? [serverId] : []), [serverId]);
  const hostProjects = useHostProjects(projectServerIds);
  const directoryProjects = useMemo<DirectoryProject[]>(
    () =>
      hostProjects.map((project) => ({
        rootPath: project.iconWorkingDir,
        name: project.projectName,
      })),
    [hostProjects],
  );

  // A scoped sheet only lists one directory, so naming it on every row is noise.
  const showRowFolders = scopeCwd === null;
  const resolveFolder = useCallback(
    (entry: FetchRecentProviderSessionEntry) =>
      showRowFolders
        ? formatDirectoryLabel(resolveDirectoryLabel(entry.cwd, directoryProjects))
        : null,
    [showRowFolders, directoryProjects],
  );

  const filterComboboxOptions = useMemo<ComboboxOption[]>(
    () => [
      { id: ALL_FILTER_VALUE, label: t("importSession.filters.all") },
      ...filterProviders.map((provider) => ({
        id: provider,
        label: providerLabelById.get(provider) ?? provider,
      })),
    ],
    [filterProviders, providerLabelById, t],
  );

  const selectedProviderLabel = useMemo(
    () =>
      filterComboboxOptions.find((opt) => opt.id === selectedProvider)?.label ??
      t("importSession.filters.all"),
    [filterComboboxOptions, selectedProvider, t],
  );

  const handleFilterOpen = useCallback(() => setIsFilterOpen(true), []);

  const filterTriggerStyle = useCallback(
    ({ pressed, hovered = false }: PressableStateCallbackType & { hovered?: boolean }) => [
      styles.filterTrigger,
      Boolean(hovered) && styles.filterTriggerHovered,
      pressed && styles.filterTriggerPressed,
    ],
    [],
  );

  const handleFilterSelect = useCallback((id: string) => {
    setSelectedProvider(id);
    setIsFilterOpen(false);
  }, []);

  const getProviderIcon = useProviderIcons(serverId);
  const SelectedProviderIcon = useProviderIcon(selectedProvider, serverId);
  const filterOptionIcons = useMemo(() => {
    const map = new Map<string, React.ReactNode>();
    map.set(ALL_FILTER_VALUE, <Layers size={14} color={theme.colors.foregroundMuted} />);
    for (const provider of filterProviders) {
      const ProviderIcon = getProviderIcon(provider);
      map.set(provider, <ProviderIcon size={14} color={theme.colors.foregroundMuted} />);
    }
    return map;
  }, [filterProviders, getProviderIcon, theme.colors.foregroundMuted]);

  const renderFilterOption = useCallback(
    ({
      option,
      selected,
      active,
      onPress,
    }: {
      option: ComboboxOption;
      selected: boolean;
      active: boolean;
      onPress: () => void;
    }) => (
      <ComboboxItem
        label={option.label}
        selected={selected}
        active={active}
        onPress={onPress}
        leadingSlot={filterOptionIcons.get(option.id)}
      />
    ),
    [filterOptionIcons],
  );

  const importMutation = useMutation({
    mutationFn: async (entry: FetchRecentProviderSessionEntry) => {
      if (!client) {
        throw new Error(t("workspace.terminal.hostDisconnected"));
      }
      if (!entry.cwd) {
        throw new Error("Session is missing a working directory");
      }
      const target = resolveImportTarget({
        entryCwd: entry.cwd,
        workspaceCwd: cwd,
        workspaceId,
        isScopedListing: scopeCwd !== null,
      });
      // Cancelling has to reach the provider: the signal ends this request and
      // asks the daemon to stop the process that is loading the session.
      const controller = new AbortController();
      importAbortRef.current = controller;
      try {
        const agent = await client.importAgent(
          {
            providerId: entry.providerId,
            providerHandleId: entry.providerHandleId,
            cwd: entry.cwd,
            ...(target.workspaceId ? { workspaceId: target.workspaceId } : {}),
          },
          controller.signal,
        );
        return { agent, target };
      } finally {
        if (importAbortRef.current === controller) {
          importAbortRef.current = null;
        }
      }
    },
    onMutate: () => {
      setImportElapsedSeconds(0);
    },
    onSuccess: ({ agent, target }) => {
      onClose();
      if (target.crossWorkspace) {
        onImported?.(agent);
      } else {
        onImportedAgent?.(agent.id);
      }
      void queryClient.invalidateQueries({
        queryKey: sessionsQueryRoot,
        refetchType: "none",
      });
    },
  });

  const importing = importMutation.isPending;
  // The provider reports no progress, so elapsed time is the honest sign that
  // the import is working instead of stuck.
  useEffect(() => {
    if (!importing) return;
    const startedAt = Date.now();
    const timer = setInterval(() => {
      setImportElapsedSeconds(Math.floor((Date.now() - startedAt) / 1000));
    }, 1_000);
    return () => clearInterval(timer);
  }, [importing]);

  const importCancelled = importMutation.isError && isImportCancelledError(importMutation.error);

  const importingSessionKey =
    importMutation.isPending && importMutation.variables
      ? `${importMutation.variables.providerId}:${importMutation.variables.providerHandleId}`
      : null;

  const handleImportSession = useCallback(
    (entry: FetchRecentProviderSessionEntry) => {
      if (expensiveImportProviders.has(entry.providerId)) {
        // The provider rebuilds this session's history for the import, so the
        // warning comes before that work starts, not after it looks hung.
        setPendingWarning(entry);
        return;
      }
      importMutation.mutate(entry);
    },
    [expensiveImportProviders, importMutation],
  );

  const handleConfirmImport = useCallback(() => {
    if (!pendingWarning) return;
    const entry = pendingWarning;
    setPendingWarning(null);
    importMutation.mutate(entry);
  }, [pendingWarning, importMutation]);

  const handleDismissImportWarning = useCallback(() => setPendingWarning(null), []);

  const handleCancelImport = useCallback(() => {
    importAbortRef.current?.abort();
  }, []);

  const providerErrorRows = useMemo(
    () => collectProviderErrorRows(providersToFetch, queries, providerLabelById),
    [queries, providersToFetch, providerLabelById],
  );

  // Every query settles, errors included, so this stops turning (#2512).
  const isRefreshing = queries.some((providerQuery) => providerQuery.isFetching);

  const handleRefresh = useCallback(() => {
    void queryClient.invalidateQueries({ queryKey: sessionsQueryRoot });
  }, [queryClient, sessionsQueryRoot]);

  const handleRetryProvider = useCallback(
    (provider: string) => {
      void queryClient.refetchQueries({
        queryKey: buildSessionsQueryKey({
          serverId,
          cwd: scopeCwd,
          query,
          limit: pageLimit,
          provider,
        }),
      });
    },
    [pageLimit, query, queryClient, scopeCwd, serverId],
  );

  const handleShowAll = useCallback(() => setScopePreference("all"), []);
  const handleScopeToWorkspace = useCallback(() => setScopePreference("workspace"), []);
  const handleLoadMore = useCallback(() => setPageLimit(nextPageLimit), []);

  const hosts = useHosts();
  const hostLabel = useMemo(
    () => hosts.find((host) => host.serverId === serverId)?.label ?? serverId ?? "",
    [hosts, serverId],
  );

  const header = useMemo<SheetHeader>(
    () => ({
      title: t("importSession.title"),
      subtitle: (
        <ScopeSubtitle
          hostLabel={hostLabel}
          isScoped={scopeCwd !== null}
          canScope={Boolean(cwd)}
          onShowAll={handleShowAll}
          onScopeToWorkspace={handleScopeToWorkspace}
        />
      ),
      ...(supportsSearch
        ? {
            search: {
              onChange: setSearchInput,
              placeholder: t("importSession.searchPlaceholder"),
              // The compact sheet keeps its content mounted while hidden, so the
              // field has to be told to drop the text the state already dropped.
              resetKey: visible ? "open" : "closed",
              testID: "import-session-search",
            },
          }
        : {}),
      actions: <RefreshAction isRefreshing={isRefreshing} onPress={handleRefresh} />,
    }),
    [
      handleRefresh,
      handleShowAll,
      handleScopeToWorkspace,
      cwd,
      hostLabel,
      isRefreshing,
      scopeCwd,
      supportsSearch,
      t,
      visible,
    ],
  );

  const isSnapshotUnsupported = requiresHostUpgrade;
  const isWaitingForSnapshot = supportsSnapshot && snapshotEntries === undefined;
  const hasNoImportableProviders = providersToFetch !== null && providersToFetch.length === 0;
  const isQueryingProviders = queries.length > 0;
  const isLoadingSessions =
    isWaitingForSnapshot ||
    (isQueryingProviders &&
      queries.some((providerQuery) => providerQuery.isLoading || providerQuery.isPending));
  const allQueriesErrored =
    isQueryingProviders && queries.every((providerQuery) => providerQuery.isError);
  const allQueriesSettled =
    isQueryingProviders &&
    queries.every((providerQuery) => !providerQuery.isLoading && !providerQuery.isPending);
  const { showEmptyState, emptyStateTitle } = computeEmptyState({
    isLoadingSessions,
    allQueriesErrored,
    isQueryingProviders,
    allQueriesSettled,
    selectedProvider,
    hasQuery: query.length > 0,
    aggregatedCount: aggregatedEntries.length,
    visibleCount: visibleEntries.length,
    totalAlreadyImportedCount,
    providerLabelById,
  });
  const showFilter = filterProviders.length > 1;
  const showLoadMore = hasMoreSessions(queries, pageLimit);
  const importNotice = useMemo(() => {
    if (pendingWarning) {
      return (
        <ImportCostWarning onConfirm={handleConfirmImport} onCancel={handleDismissImportWarning} />
      );
    }
    if (importing) {
      return (
        <ImportProgressNote elapsedSeconds={importElapsedSeconds} onCancel={handleCancelImport} />
      );
    }
    if (importCancelled) {
      return <ImportCancelledNote />;
    }
    return null;
  }, [
    pendingWarning,
    importing,
    importElapsedSeconds,
    importCancelled,
    handleConfirmImport,
    handleDismissImportWarning,
    handleCancelImport,
  ]);
  const showListHeader =
    hasListHeader({
      showFilter,
      isClientReady: Boolean(client),
      isSnapshotUnsupported,
      hasNoImportableProviders,
      isLoadingSessions,
      hasRows: visibleEntries.length > 0,
      // A stopped import is not a failure, so it must not read like one.
      importErrored: importMutation.isError && !importCancelled,
      hasProviderErrors: providerErrorRows.length > 0,
      hasPreviewsUnavailable: previewsUnavailable,
    }) || importNotice !== null;
  const listHeader = useMemo(
    () =>
      showListHeader ? (
        <View style={styles.listHeaderContent}>
          {importNotice}
          {showFilter ? (
            <View ref={filterAnchorRef} collapsable={false} style={styles.filterTriggerWrap}>
              <Pressable
                onPress={handleFilterOpen}
                style={filterTriggerStyle}
                testID="import-session-filter-trigger"
                accessibilityRole="button"
                accessibilityLabel={`Filter: ${selectedProviderLabel}`}
              >
                {selectedProvider === ALL_FILTER_VALUE ? (
                  <Layers size={14} color={theme.colors.foregroundMuted} />
                ) : (
                  <SelectedProviderIcon size={14} color={theme.colors.foregroundMuted} />
                )}
                <Text style={styles.filterTriggerText} numberOfLines={1}>
                  {selectedProviderLabel}
                </Text>
                <ChevronDown size={14} color={theme.colors.foregroundMuted} />
              </Pressable>
              <Combobox
                options={filterComboboxOptions}
                value={selectedProvider}
                onSelect={handleFilterSelect}
                renderOption={renderFilterOption}
                searchable={false}
                title="Filter by provider"
                open={isFilterOpen}
                onOpenChange={setIsFilterOpen}
                anchorRef={filterAnchorRef}
                desktopPlacement="bottom-start"
                desktopPreventInitialFlash
              />
            </View>
          ) : null}
          <SheetStatusMessages
            isClientReady={Boolean(client)}
            isSnapshotUnsupported={isSnapshotUnsupported}
            hasNoImportableProviders={hasNoImportableProviders}
            isLoadingSessions={isLoadingSessions}
            hasRows={visibleEntries.length > 0}
            importErrored={importMutation.isError && !importCancelled}
          />
          {providerErrorRows.length > 0 ? (
            <ProviderErrorBanner rows={providerErrorRows} onRetry={handleRetryProvider} />
          ) : null}
          {previewsUnavailable ? <PreviewsUnavailableNote /> : null}
        </View>
      ) : null,
    [
      showListHeader,
      showFilter,
      filterTriggerStyle,
      handleFilterOpen,
      selectedProvider,
      theme.colors.foregroundMuted,
      SelectedProviderIcon,
      selectedProviderLabel,
      filterComboboxOptions,
      handleFilterSelect,
      renderFilterOption,
      isFilterOpen,
      client,
      isSnapshotUnsupported,
      hasNoImportableProviders,
      isLoadingSessions,
      visibleEntries.length,
      importMutation.isError,
      providerErrorRows,
      handleRetryProvider,
      previewsUnavailable,
      importNotice,
      importCancelled,
    ],
  );
  const listFooter = useMemo(
    () => (
      <>
        {showLoadMore ? (
          <View style={styles.footer}>
            <Button
              variant="ghost"
              onPress={handleLoadMore}
              disabled={isRefreshing}
              testID="import-session-load-more"
            >
              {t("importSession.actions.loadMore")}
            </Button>
          </View>
        ) : null}
        {showEmptyState ? <SheetEmptyState title={emptyStateTitle} /> : null}
      </>
    ),
    [showLoadMore, handleLoadMore, isRefreshing, t, showEmptyState, emptyStateTitle],
  );
  return (
    <SessionRows
      visible={visible}
      onClose={onClose}
      header={header}
      listHeader={listHeader}
      listFooter={listFooter}
      serverId={serverId}
      entries={visibleEntries}
      disabled={importMutation.isPending}
      importingSessionKey={importingSessionKey}
      resolveFolder={resolveFolder}
      onImportSession={handleImportSession}
    />
  );
}

const styles = StyleSheet.create((theme) => ({
  subtitleRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: theme.spacing[2],
    minWidth: 0,
  },
  subtitleText: {
    flexShrink: 1,
    minWidth: 0,
    color: theme.colors.foregroundMuted,
    fontSize: theme.fontSize.sm,
  },
  subtitleAction: {
    marginHorizontal: -theme.spacing[2],
  },
  filterTriggerWrap: {
    paddingBottom: theme.spacing[2],
  },
  filterTrigger: {
    flexDirection: "row",
    alignItems: "center",
    gap: theme.spacing[1.5],
    alignSelf: "flex-start",
    maxWidth: "100%",
    paddingVertical: theme.spacing[1.5],
    paddingHorizontal: theme.spacing[3],
    borderRadius: theme.borderRadius.md,
    backgroundColor: theme.colors.surface1,
    borderWidth: theme.borderWidth[1],
    borderColor: theme.colors.border,
  },
  filterTriggerHovered: {
    backgroundColor: theme.colors.surface2,
  },
  filterTriggerPressed: {
    backgroundColor: theme.colors.surface3,
  },
  filterTriggerText: {
    flexShrink: 1,
    minWidth: 0,
    color: theme.colors.foreground,
    fontSize: theme.fontSize.base,
    fontWeight: theme.fontWeight.medium,
  },
  listBody: {
    gap: 0,
  },
  list: {
    flex: 1,
    minHeight: 0,
    marginHorizontal: -theme.spacing[2],
  },
  listHeader: {
    marginHorizontal: theme.spacing[2],
    paddingBottom: theme.spacing[4],
  },
  listHeaderContent: {
    gap: theme.spacing[4],
  },
  smallList: {
    gap: theme.spacing[1],
  },
  listFooter: {
    marginHorizontal: theme.spacing[2],
  },
  listRow: {
    marginHorizontal: theme.spacing[2],
    paddingBottom: theme.spacing[1],
  },
  footer: {
    paddingTop: theme.spacing[2],
    alignItems: "center",
  },
  row: {
    flexDirection: "row",
    alignItems: "flex-start",
    gap: theme.spacing[2],
    paddingVertical: theme.spacing[2],
    paddingHorizontal: theme.spacing[2],
    marginHorizontal: -theme.spacing[2],
    borderRadius: theme.borderRadius.lg,
  },
  rowHovered: {
    backgroundColor: theme.colors.surface1,
  },
  rowPressed: {
    backgroundColor: theme.colors.surface2,
  },
  rowIconWrap: {
    width: theme.iconSize.md,
    paddingTop: 2,
    alignItems: "center",
    justifyContent: "center",
  },
  rowContent: {
    flex: 1,
    minWidth: 0,
    gap: theme.spacing[1],
  },
  rowHeader: {
    flexDirection: "row",
    alignItems: "baseline",
    justifyContent: "space-between",
    gap: theme.spacing[2],
  },
  rowTitle: {
    flex: 1,
    minWidth: 0,
    color: theme.colors.foreground,
    fontSize: theme.fontSize.base,
  },
  rowMeta: {
    color: theme.colors.foregroundMuted,
    fontSize: theme.fontSize.sm,
  },
  rowPreview: {
    color: theme.colors.foregroundMuted,
    fontSize: theme.fontSize.base,
    lineHeight: 20,
  },
  rowFolder: {
    color: theme.colors.foregroundMuted,
    fontSize: theme.fontSize.sm,
  },
  statusRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: theme.spacing[2],
    paddingVertical: theme.spacing[2],
  },
  statusText: {
    color: theme.colors.foregroundMuted,
    fontSize: theme.fontSize.base,
  },
  previewsNote: {
    color: theme.colors.foregroundMuted,
    fontSize: theme.fontSize.sm,
  },
  importNotice: {
    gap: theme.spacing[2],
    marginBottom: theme.spacing[2],
    borderWidth: theme.borderWidth[1],
    borderColor: theme.colors.border,
    borderRadius: theme.borderRadius.lg,
    backgroundColor: theme.colors.surface1,
    paddingVertical: theme.spacing[3],
    paddingHorizontal: theme.spacing[3],
  },
  importProgressHeader: {
    flexDirection: "row",
    alignItems: "center",
    gap: theme.spacing[2],
  },
  importNoticeTitle: {
    flexShrink: 1,
    color: theme.colors.foreground,
    fontSize: theme.fontSize.sm,
    fontWeight: theme.fontWeight.medium,
  },
  importNoticeBody: {
    color: theme.colors.foregroundMuted,
    fontSize: theme.fontSize.sm,
  },
  importNoticeActions: {
    flexDirection: "row",
    justifyContent: "flex-end",
    gap: theme.spacing[2],
  },
  importCancelledNote: {
    color: theme.colors.foregroundMuted,
    fontSize: theme.fontSize.sm,
  },
  errorBanner: {
    marginBottom: theme.spacing[2],
    borderWidth: theme.borderWidth[1],
    borderColor: theme.colors.border,
    borderRadius: theme.borderRadius.lg,
    paddingVertical: theme.spacing[1],
    paddingHorizontal: theme.spacing[3],
  },
  errorRow: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    gap: theme.spacing[2],
  },
  errorText: {
    flexShrink: 1,
    minWidth: 0,
    color: theme.colors.palette.red[300],
    fontSize: theme.fontSize.sm,
  },
  emptyState: {
    alignItems: "center",
    justifyContent: "center",
    gap: theme.spacing[2],
    paddingVertical: theme.spacing[8],
    paddingHorizontal: theme.spacing[4],
  },
  emptyStateIcon: {
    opacity: 0.6,
    marginBottom: theme.spacing[1],
  },
  emptyStateTitle: {
    color: theme.colors.foreground,
    fontSize: theme.fontSize.base,
    textAlign: "center",
  },
  refreshButton: {
    padding: theme.spacing[2],
    marginRight: theme.spacing[1],
    borderRadius: theme.borderRadius.lg,
  },
  refreshButtonPressed: {
    backgroundColor: theme.colors.surface2,
  },
  refreshIconSlot: {
    width: 16,
    height: 16,
    alignItems: "center",
    justifyContent: "center",
  },
}));
