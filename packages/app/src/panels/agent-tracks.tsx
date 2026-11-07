import type { AgentAddress } from "@getpaseo/protocol/addresses";
import { memo, useCallback, useMemo, useState, type ReactElement } from "react";
import { BackgroundWorkTrack, createHostBackgroundWorkModel } from "@/background-work";
import { WorkspaceDiffStatPill } from "@/composer/diff-stat-pill";
import { AgentTaskList } from "@/composer/task-list";
import { ComposerTrackBar } from "@/composer/tracks";
import { useIsCompactFormFactor } from "@/constants/layout";
import { usePaneContext } from "@/panels/pane-context";
import { useSettings } from "@/hooks/use-settings";
import { PluginComposerPills } from "@/plugins";
import { useSessionStore } from "@/stores/session-store";
import { useToast } from "@/contexts/toast-context";
import { i18n } from "@/i18n/i18next";
import { confirmDialog } from "@/utils/confirm-dialog";
import { toErrorMessage } from "@/utils/error-messages";
import {
  useArchiveFinishedSubagents,
  useArchiveSubagent,
  useDetachSubagent,
  useSubagentSheetTab,
  useSubagentsForParent,
} from "@/subagents";
import {
  IDLE_PROVIDER_DELEGATION_STOP_STATUS,
  requestStopProviderDelegations,
  type ProviderDelegationStopStatus,
} from "@/subagents/provider-delegation-stop";
import { countRunningProviderSubagents } from "@/subagents/provider-store";
import { formatProviderName } from "@/subagents/track-presentation";
import { ParentAgentPill } from "@/subagents/parent";
import { SubagentsTrack } from "@/subagents/track";
import type { TodoEntry } from "@/types/stream";
import { navigateToAgent } from "@/utils/navigate-to-agent";
import { buildWorkspaceTabPersistenceKey, type WorkspaceTabTarget } from "@/workspace-tabs/model";
import { openPreferredWorkspaceTarget } from "@/workspace-tabs/open-beside";
import { openComposerChanges } from "@/workspace-tabs/open-supporting-view";

/**
 * The pane's ambient context — workspace changes, subagents, and tasks — as a row of pills above
 * the composer.
 *
 * The row shares the composer's keyboard transform and owns the space between itself and the
 * transcript. Each pill owns its action while tab placement stays behind the workspace boundary.
 */
export const AgentTracks = memo(function AgentTracks({
  serverId,
  workspaceId,
  agentId,
  parentAgentId,
}: {
  serverId: string;
  workspaceId: string;
  agentId: string;
  parentAgentId: string | null;
}): ReactElement {
  const subagentRows = useSubagentsForParent({ serverId, parentAgentId: agentId });
  const tasks = useSessionStore((state): TodoEntry[] | undefined =>
    state.sessions[serverId]?.agentTasks.get(agentId),
  );
  const archiveFinishedSubagents = useArchiveFinishedSubagents({
    serverId,
    parentAgentId: agentId,
    rows: subagentRows,
  });
  const { tabId } = usePaneContext();
  const backgroundWorkCount = useSessionStore((state) => {
    const session = state.sessions[serverId];
    const agent = session?.agents.get(agentId) ?? session?.agentDetails.get(agentId);
    return agent?.backgroundWork?.count ?? 0;
  });
  const backgroundModel = useMemo(
    () => createHostBackgroundWorkModel(serverId, agentId),
    [serverId, agentId],
  );
  const isCompact = useIsCompactFormFactor();
  const openInSidePane = useSettings((settings) => settings.openInSidePane);
  const workspaceKey = buildWorkspaceTabPersistenceKey({ serverId, workspaceId });
  const canDetachSubagents = useSessionStore(
    (state) => state.sessions[serverId]?.serverInfo?.features?.agentDetach === true,
  );
  // Two things have to hold: the daemon accepts the request, and this agent's
  // provider advertised the stop. Anything less leaves the rows read-only rather
  // than offering a stop that cannot be carried out.
  const canStopProviderDelegations = useSessionStore((state) => {
    const session = state.sessions[serverId];
    if (session?.serverInfo?.features?.providerDelegationStop !== true) {
      return false;
    }
    const agent = session.agents.get(agentId) ?? session.agentDetails.get(agentId);
    return agent?.capabilities.supportsProviderDelegationStop === true;
  });
  const toast = useToast();
  const archiveSubagent = useArchiveSubagent();
  const detachSubagent = useDetachSubagent();
  const openSubagentTarget = useCallback(
    (target: WorkspaceTabTarget) => {
      openPreferredWorkspaceTarget({
        workspaceKey,
        target,
        source: "subagents",
        preferences: openInSidePane,
        parentTabId: tabId,
      });
    },
    [openInSidePane, tabId, workspaceKey],
  );
  const handleOpenSubagent = useCallback(
    ({ serverId: targetServerId, agentId: subagentId }: AgentAddress) => {
      const session = useSessionStore.getState().sessions[targetServerId];
      const agent = session?.agents.get(subagentId) ?? session?.agentDetails.get(subagentId);
      // A subagent in another workspace lives in that workspace's tabs.
      if (
        targetServerId !== serverId ||
        (agent?.workspaceId && agent.workspaceId !== workspaceId)
      ) {
        navigateToAgent({ serverId: targetServerId, agentId: subagentId });
        return;
      }
      openSubagentTarget({ kind: "agent", agentId: subagentId });
    },
    [openSubagentTarget, serverId, workspaceId],
  );
  const [providerDelegationStopStatus, setProviderDelegationStopStatus] =
    useState<ProviderDelegationStopStatus>(IDLE_PROVIDER_DELEGATION_STOP_STATUS);
  const handleStopProviderDelegations = useCallback(async () => {
    const activeTaskCount = subagentRows.reduce(
      (count, row) => (row.kind === "provider" && row.status === "running" ? count + 1 : count),
      0,
    );
    const provider = subagentRows.find((row) => row.kind === "provider")?.provider ?? agentId;
    setProviderDelegationStopStatus({ kind: "requesting" });
    const status = await requestStopProviderDelegations(
      {
        serverId,
        agentId,
        providerLabel: formatProviderName(provider),
        activeTaskCount,
      },
      {
        confirm: confirmDialog,
        t: (key, options) => i18n.t(key, options as never) as unknown as string,
        stopProviderDelegations: async ({ serverId: targetServerId, agentId: targetAgentId }) => {
          const client = useSessionStore.getState().sessions[targetServerId]?.client;
          if (!client) {
            throw new Error(i18n.t("workspaceSetup.errors.hostDisconnected"));
          }
          return client.stopProviderDelegations(targetAgentId);
        },
        // Read after the stop: the daemon refreshes the tree before answering, so
        // the rows already carry the statuses the user is about to see.
        countActiveTasks: () => countRunningProviderSubagents(serverId, agentId),
        reportError: (error) => {
          toast.error(toErrorMessage(error));
        },
      },
    );
    setProviderDelegationStopStatus(status);
  }, [agentId, serverId, subagentRows, toast]);
  const handleOpenProviderSubagent = useCallback(
    (ownerAgentId: string, subagentId: string) => {
      openSubagentTarget({
        kind: "provider_subagent",
        parentAgentId: ownerAgentId,
        subagentId,
      });
    },
    [openSubagentTarget],
  );
  const sheetTabFor = useSubagentSheetTab(workspaceKey, tabId);
  const handleOpenChanges = useCallback(() => {
    if (!workspaceKey) {
      return;
    }
    openComposerChanges({
      isCompact,
      workspaceKey,
      preferences: openInSidePane,
    });
  }, [isCompact, openInSidePane, workspaceKey]);

  return (
    <ComposerTrackBar>
      <BackgroundWorkTrack model={backgroundModel} count={backgroundWorkCount} />
      {parentAgentId ? <ParentAgentPill serverId={serverId} parentAgentId={parentAgentId} /> : null}
      <AgentTaskList tasks={tasks} />
      <SubagentsTrack
        serverId={serverId}
        rows={subagentRows}
        onOpenSubagent={handleOpenSubagent}
        onOpenProviderSubagent={handleOpenProviderSubagent}
        sheetTabFor={sheetTabFor}
        onArchiveSubagent={archiveSubagent}
        onArchiveFinished={archiveFinishedSubagents.archiveFinished}
        archiveFinishedStatus={archiveFinishedSubagents.status}
        onDetachSubagent={canDetachSubagents ? detachSubagent : undefined}
        onStopProviderDelegations={
          canStopProviderDelegations ? handleStopProviderDelegations : undefined
        }
        providerDelegationStopStatus={providerDelegationStopStatus}
      />
      <PluginComposerPills
        serverId={serverId}
        workspaceId={workspaceId}
        agentId={agentId}
        compact={isCompact}
      />
      <WorkspaceDiffStatPill
        serverId={serverId}
        workspaceId={workspaceId}
        onPress={handleOpenChanges}
      />
    </ComposerTrackBar>
  );
});
