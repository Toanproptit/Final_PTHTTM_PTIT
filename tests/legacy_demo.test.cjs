// Unit tests execute the inline demo with a small DOM adapter; no browser rendering is claimed.
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

function initialize() {
  const html = fs.readFileSync(path.join(__dirname, '..', 'index.html'), 'utf8');
  const elements = {};
  const context = new Proxy({}, { get: () => () => {} });
  for (const [, id] of html.matchAll(/id="([^"]+)"/g)) {
    elements[id] = { value: id === 'stepsInput' ? '20' : id === 'episodesInput' ? '100' : '',
      textContent: '', innerHTML: '', disabled: false,
      classList: { toggle() {} }, getBoundingClientRect: () => ({ width: 600, height: 180 }),
      getContext: () => context };
  }
  const sandbox = { document: { getElementById: id => elements[id] },
    window: { devicePixelRatio: 1, addEventListener() {} }, setTimeout: callback => callback() };
  vm.runInNewContext(html.match(/<script>([\s\S]*?)<\/script>/)[1], sandbox);
  return elements;
}

test('demo initializes and an episode stops at its boundary', () => {
  const elements = initialize();
  assert.equal(elements.rlTotal.textContent, '$0');
  elements.stepBtn.onclick();
  assert.match(elements.episodeText.textContent, /Episode 1 · Lượt 1\/20/);
  elements.episodeBtn.onclick();
  assert.match(elements.episodeText.textContent, /Episode 1 · Lượt 20\/20/);
  elements.stepBtn.onclick();
  assert.match(elements.episodeText.textContent, /Episode 2 · Lượt 1\/20/);
  assert.ok(Number.parseInt(elements.discountRate.textContent) <= 100);
});

test('training creates Q values and reset restores an untrained demo', () => {
  const elements = initialize();
  elements.trainBtn.onclick();
  assert.match(elements.trainedTag.textContent, /100 EPISODE/);
  assert.doesNotMatch(elements.qBody.innerHTML, /NaN/);
  assert.ok([...elements.qBody.innerHTML.matchAll(/>(\d+\.\d+)</g)].some(match => Number(match[1]) > 0));
  elements.resetBtn.onclick();
  assert.equal(elements.trainedTag.textContent, 'CHƯA HUẤN LUYỆN');
  assert.equal(elements.rlTotal.textContent, '$0');
});
