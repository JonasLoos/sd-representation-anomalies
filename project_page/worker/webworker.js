import * as wasm from './pkg/worker.js';

const wasmInit = wasm.default({}).then(_ => null);

function handleFetchRepr(id, data) {
  const { url, n, m } = data;
  wasm.fetch_repr(url, n, m)
    .then(() => postMessage({ id, data: { status: 'success' } }))
    .catch((e) => postMessage({ id, data: { status: 'error', msg: e } }));
}

function handleCalcSimilarities(id, data) {
  const { func, repr1_str, repr2_str, row, col } = data;
  try {
    const similarities = wasm.calc_similarities(func, repr1_str, repr2_str, row, col);
    postMessage({ id, data: { status: 'success', similarities } });
  } catch (e) {
    postMessage({ id, data: { status: 'error', msg: e } });
  }
}

onmessage = function(e) {
  wasmInit.then(_ => {
    const { id, task, data } = e.data;
    switch(task) {
      case 'fetch_repr':
        handleFetchRepr(id, data);
        break;
      case 'calc_similarities':
        handleCalcSimilarities(id, data);
        break;
      default:
        console.error(`Unknown task: ${task}`);
        postMessage({ id, data: { status: 'error', msg: `Unknown task: ${task}` } });
    }
  });
};
