import { Text, View } from "react-native";
import { StyleSheet } from "react-native-unistyles";
import { useTranslation } from "react-i18next";

/**
 * What a provider sub-agent panel shows when the provider does not share the
 * child's journal.
 *
 * An empty transcript would read as a broken chat. The panel says instead that
 * the log is unavailable and that only the status is known: this child has no
 * plain turn that could run outside its provider.
 */
export function ProviderSubagentJournalUnavailable() {
  const { t } = useTranslation();
  return (
    <View style={styles.unsupported} testID="provider-subagent-panel-unsupported">
      <Text style={styles.unsupportedText}>{t("subagents.providerLogUnavailable")}</Text>
    </View>
  );
}

const styles = StyleSheet.create((theme) => ({
  unsupported: { flex: 1, alignItems: "center", justifyContent: "center", padding: 24 },
  unsupportedText: { color: theme.colors.foregroundMuted, textAlign: "center" },
}));
