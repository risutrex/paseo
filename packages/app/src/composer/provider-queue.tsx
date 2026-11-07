import type { ReactElement } from "react";
import { Text, View } from "react-native";
import { StyleSheet } from "react-native-unistyles";
import type { Theme } from "@/styles/theme";
import type { ProviderQueueItemState } from "@/stores/session-store";

export interface ProviderQueueTrackProps {
  /** The messages the provider holds for later turns, as it reports them. */
  items: readonly ProviderQueueItemState[];
  /** Names the owner of these rows, so they are never read as the composer queue. */
  label: string;
  /** Renders the provider's own status word for one item. */
  statusLabel: (status: string) => string;
}

/**
 * Shows the messages a provider keeps for later turns.
 *
 * They sit in the composer next to its own queue but are never mixed with it: a
 * position here belongs to the agent process, and the provider answers ACP with
 * no way to edit or promote one. A row is therefore read-only - offering an
 * action the provider cannot carry out would be worse than showing none.
 *
 * An empty or unsupported queue renders nothing, so a provider without a queue
 * leaves the composer exactly as it was.
 */
export function ProviderQueueTrack({
  items,
  label,
  statusLabel,
}: ProviderQueueTrackProps): ReactElement | null {
  if (items.length === 0) {
    return null;
  }
  return (
    <View style={styles.track} testID="composer-provider-queue">
      <Text style={styles.label}>{label}</Text>
      {items.map((item, index) => (
        <View key={item.itemId} style={styles.row} testID="composer-provider-queue-item">
          <Text style={styles.badge} numberOfLines={1}>
            {`${index + 1} · ${statusLabel(item.status)}`}
          </Text>
          <Text style={styles.text} numberOfLines={2} ellipsizeMode="tail">
            {item.text}
          </Text>
        </View>
      ))}
    </View>
  );
}

const styles = StyleSheet.create((theme: Theme) => ({
  track: {
    flexDirection: "column",
    gap: theme.spacing[2],
  },
  label: {
    color: theme.colors.foregroundMuted,
    fontSize: theme.fontSize.sm,
  },
  row: {
    flexDirection: "row",
    alignItems: "center",
    paddingHorizontal: theme.spacing[3],
    paddingVertical: theme.spacing[2],
    backgroundColor: theme.colors.surface1,
    borderRadius: theme.borderRadius.lg,
    borderWidth: theme.borderWidth[1],
    borderColor: theme.colors.border,
    gap: theme.spacing[2],
  },
  badge: {
    color: theme.colors.foregroundMuted,
    fontSize: theme.fontSize.sm,
  },
  text: {
    flex: 1,
    color: theme.colors.foreground,
    fontSize: theme.fontSize.base,
  },
}));
