const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const source = fs.readFileSync(path.join(__dirname, '../src/gui/replay_web/app.js'), 'utf8');
const flush = () => new Promise(resolve => setImmediate(resolve));

function harness() {
  const elements = new Map();
  function element(id) {
    if (!elements.has(id)) {
      const classes = new Set();
      elements.set(id, {
        value: '', style: {}, dataset: {}, parentElement: {},
        classList: {toggle(name, enabled) { enabled ? classes.add(name) : classes.delete(name); },
          contains(name) { return classes.has(name); }},
        addEventListener() {}, getContext() { return {}; },
      });
    }
    return elements.get(id);
  }
  const buttons = [60, 5, 360].map(tf => Object.assign(element(`tf-${tf}`), {dataset: {tf: String(tf)}}));
  buttons[0].classList.toggle('selected', true);
  const requests = [];
  const context = vm.createContext({
    document: {getElementById: element, addEventListener() {}, querySelector() { return null; },
      querySelectorAll(selector) { return selector === '[data-tf]' ? buttons : []; }},
    window: {addEventListener() {}}, ResizeObserver: class { observe() {} },
    performance: {now() { return 0; }}, setTimeout() {}, clearTimeout() {},
    fetch(url, options) {
      if (url === '/api/captures') return Promise.resolve({ok: true, json: async () => ({captures: []})});
      return new Promise((resolve, reject) => {
        requests.push({body: JSON.parse(options.body), resolve(data) { resolve({ok: true, json: async () => data}); }, reject});
      });
    },
  });
  vm.runInContext(source, context, {filename: 'app.js'});
  function apply(data, reset = false) {
    context.incoming = data;
    vm.runInContext(`applyState(incoming, ${reset})`, context);
  }
  apply(response(60, [{time: 120, close: 100}], [{actor: 'system', action: 'attempt'}]), true);
  return {requests, buttons, apply,
    click(tf) { buttons.find(button => Number(button.dataset.tf) === tf).onclick(); },
    advance() { vm.runInContext("enqueue(() => requestAction('advance', {seconds: 1}))", context); },
    settled() { return vm.runInContext('queue', context); },
    read() { return JSON.parse(vm.runInContext('JSON.stringify({state, candles, events, timeframe})', context)); },
  };
}

function response(timeframe, candles, events = []) {
  return {id: 'attempt', timeframe, candles, events, utc: '2026-11-26T04:00:00Z',
    account: {position: 0, equity: 10000, unrealized_pnl: 0, realized_pnl: 0, fees: 0},
    layers: [], orders: [], drawings: [], progress: 0, capture: 'capture', price_source: 'trades',
    book_valid: true, finished: false};
}

test('timeframe transition waits for an in-flight advance and replaces candles only on its response', async () => {
  const app = harness();
  app.advance();
  await flush();
  app.click(5);
  assert.equal(app.read().timeframe, 60);
  assert.deepEqual(app.read().candles.map(bar => bar.time), [120]);
  app.requests[0].resolve(response(60, [{time: 180, close: 101}], [{actor: 'human', action: 'wait'}]));
  await flush();
  assert.equal(app.requests[1].body.timeframe, 5);
  assert.equal(app.requests[1].body.since, 0);
  assert.equal(app.requests[1].body.event_offset, 2);
  assert.deepEqual(app.read().candles.map(bar => bar.time), [120, 180]);
  app.requests[1].resolve(response(5, [{time: 185, close: 102}]));
  await app.settled();
  assert.deepEqual(app.read().candles.map(bar => bar.time), [185]);
  assert.equal(app.read().state.timeframe, 5);
  assert.equal(app.read().events.length, 2);
  app.advance();
  await flush();
  app.requests[2].resolve(response(5, [{time: 185, close: 103}, {time: 190, close: 104}]));
  await app.settled();
  assert.deepEqual(app.read().candles, [{time: 185, close: 103}, {time: 190, close: 104}]);
});

test('responses for an inactive timeframe cannot alter visible state or candles', () => {
  const app = harness();
  const before = app.read();
  app.apply(response(5, [{time: 125, close: 999}], [{actor: 'human', action: 'skip'}]));
  assert.deepEqual(app.read(), before);
});

test('rapid timeframe clicks each keep their transition and state request together', async () => {
  const app = harness();
  app.click(5);
  app.click(360);
  await flush();
  assert.equal(app.requests.length, 1);
  assert.equal(app.requests[0].body.timeframe, 5);
  app.requests[0].resolve(response(5, [{time: 125, close: 101}]));
  await flush();
  assert.equal(app.requests[1].body.timeframe, 360);
  assert.deepEqual(app.read().candles.map(bar => bar.time), [125]);
  app.requests[1].resolve(response(360, [{time: 0, close: 102}]));
  await app.settled();
  assert.deepEqual(app.read().candles.map(bar => bar.time), [0]);
  assert.equal(app.read().state.timeframe, 360);
});

test('a failed timeframe request preserves candles and restores the previous timeframe', async () => {
  const app = harness();
  app.click(5);
  await flush();
  app.requests[0].reject(new Error('state request failed'));
  await app.settled();
  assert.equal(app.read().timeframe, 60);
  assert.equal(app.read().state.timeframe, 60);
  assert.deepEqual(app.read().candles.map(bar => bar.time), [120]);
  assert.ok(app.buttons[0].classList.contains('selected'));
});
