import type { OutputColumn } from "./types";

export function buildPrompt(
  inputColumns: string[],
  rowData: Record<string, string>,
  outputColumns: OutputColumn[],
  enrichmentDescription: string,
  customPrompt?: string,
  useWebSearch: boolean = true
): string {
  if (customPrompt) {
    let prompt = customPrompt;
    for (const col of inputColumns) {
      prompt = prompt.replace(new RegExp(`\\{${col}\\}`, "g"), rowData[col] || "");
    }
    return prompt;
  }

  const inputData = inputColumns
    .map((col) => `"${col}":"${rowData[col] || ""}"`)
    .join(",");

  const outputFields = outputColumns
    .map((f) => `"${f.key}":"string"`)
    .join(",");

  // Trimmed prompt: ~60% fewer tokens than the original verbose version.
  // Every token here is multiplied by every row in the batch.
  const searchInstruction = useWebSearch
    ? "Use web search for current data."
    : "Use your knowledge only (no web search).";

  return `${enrichmentDescription}

Data: {${inputData}}

Return ONLY valid JSON: {${outputFields}}
${searchInstruction} Be concise. Make your best educated guess from available evidence; only use "N/A" when you truly have too little information to guess. No extra text.`;
}

/**
 * Builds an EDITABLE prompt template with `{column_name}` placeholders instead of
 * a concrete row's values. This is what seeds the Advanced-mode textarea: at run
 * time buildPrompt() substitutes each row's real values into these placeholders.
 * (Seeding with a concrete row would make every row reuse that row's data.)
 */
export function buildPromptTemplate(
  inputColumns: string[],
  outputColumns: OutputColumn[],
  enrichmentDescription: string,
  useWebSearch: boolean = true
): string {
  const inputData = inputColumns.map((col) => `"${col}":"{${col}}"`).join(",");
  const outputFields = outputColumns.map((f) => `"${f.key}":"string"`).join(",");
  const searchInstruction = useWebSearch
    ? "Use web search for current data."
    : "Use your knowledge only (no web search).";

  return `${enrichmentDescription}

Data: {${inputData}}

Return ONLY valid JSON: {${outputFields}}
${searchInstruction} Be concise. Make your best educated guess from available evidence; only use "N/A" when you truly have too little information to guess. No extra text.`;
}
