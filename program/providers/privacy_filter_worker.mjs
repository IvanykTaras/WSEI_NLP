import readline from "node:readline";
import { env, pipeline } from "@huggingface/transformers";

const modelDir = process.env.PRIVACY_FILTER_MODEL_DIR;
if (!modelDir) {
  process.stdout.write(JSON.stringify({ ready: false, error: "Brak PRIVACY_FILTER_MODEL_DIR." }) + "\n");
  process.exit(1);
}

env.allowLocalModels = true;
env.allowRemoteModels = false;
env.useBrowserCache = false;

let classifier;
try {
  classifier = await pipeline("token-classification", modelDir, {
    device: "cpu",
    dtype: "q4",
    local_files_only: true,
  });
  process.stdout.write(JSON.stringify({ ready: true }) + "\n");
} catch (error) {
  process.stdout.write(JSON.stringify({ ready: false, error: String(error?.stack || error) }) + "\n");
  process.exit(1);
}

const input = readline.createInterface({ input: process.stdin, terminal: false });
for await (const line of input) {
  try {
    const request = JSON.parse(line);
    const rows = await classifier(String(request.text || ""), {
      aggregation_strategy: "simple",
    });
    process.stdout.write(JSON.stringify({ id: request.id, rows }) + "\n");
  } catch (error) {
    process.stdout.write(JSON.stringify({
      id: null,
      error: String(error?.stack || error),
    }) + "\n");
  }
}
