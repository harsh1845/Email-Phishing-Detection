import * as ort from 'onnxruntime-web';

type Meta = {
  features: string[];
  threshold: number;
  calibration?: { intercept: number; coef: number };
};

let session: ort.InferenceSession | null = null;
let meta: Meta | null = null;

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

function calibrate(raw: number): number {
  if (!meta?.calibration) return raw;
  return sigmoid(meta.calibration.coef * raw + meta.calibration.intercept);
}

async function init(): Promise<void> {
  const wasmPath = chrome.runtime.getURL('wasm/');
  ort.env.wasm.wasmPaths = wasmPath;
  ort.env.wasm.numThreads = 1;
  const [modelBuf, metaRes] = await Promise.all([
    fetch(chrome.runtime.getURL('model/model.onnx')).then((r) => {
      if (!r.ok) throw new Error('model missing');
      return r.arrayBuffer();
    }),
    fetch(chrome.runtime.getURL('model/model_meta.json')).then((r) => r.json() as Promise<Meta>),
  ]);
  meta = metaRes;
  session = await ort.InferenceSession.create(modelBuf, {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all',
  });
  chrome.runtime.sendMessage({ type: 'MODEL_READY' }).catch(() => undefined);
}

async function predict(features: number[]): Promise<number> {
  if (!session) throw new Error('model not ready');
  const input = new ort.Tensor('float32', Float32Array.from(features), [1, features.length]);
  const feeds: Record<string, ort.Tensor> = {};
  feeds[session.inputNames[0]] = input;
  const out = await session.run(feeds);
  const tensor = out[session.outputNames[0]];
  const data = tensor.data as Float32Array;
  let raw = 0;
  if (data.length >= 2) raw = data[1];
  else raw = data[0];
  return calibrate(raw);
}

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message?.type !== 'MODEL_PREDICT') return false;
  predict(message.features as number[])
    .then((score) => sendResponse({ score }))
    .catch((err) => sendResponse({ score: null, error: String(err) }));
  return true;
});

void init().catch((err) => {
  console.warn('PhishGuard model init failed; using heuristics only.', err);
});
