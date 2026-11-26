'use strict';

const $ = id => document.getElementById(id);
const number = value => value === null || value === undefined ? null : Number(value);
const price = value => value == null ? '—' : Number(value).toLocaleString('en-US', {minimumFractionDigits: 1, maximumFractionDigits: 1});
const money = value => value == null ? '—' : Number(value).toLocaleString('en-US', {style: 'currency', currency: 'USD'});
const escapeHTML = value => String(value ?? '').replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
let state = null, candles = [], events = [], timeframe = 60, orderType = 'market', tab = 'positions';
let playing = false, generation = 0, queue = Promise.resolve(), currentCapture = null, showRange = true;
let toastTimer, lastResponseMs = 0;

function toast(message, error = false) {
  $('toast').textContent = message; $('toast').classList.toggle('error', error); $('toast').style.display = 'block';
  clearTimeout(toastTimer); toastTimer = setTimeout(() => $('toast').style.display = 'none', 4500);
}

async function api(path, body) {
  const started = performance.now();
  const response = await fetch(path, body === undefined ? {} : {method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(body)});
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || 'The replay request failed');
  lastResponseMs = performance.now() - started;
  return result;
}

function enqueue(fn) {
  const operation = queue.then(fn);
  queue = operation.catch(error => { pause(); toast(error.message, true); });
  return operation;
}

function payload(extra = {}) {
  return {timeframe, since: candles.length ? candles.at(-1).time : 0, event_offset: events.length, ...extra};
}

async function requestAction(action, body = {}) {
  if (!state) throw new Error('Open a recording first');
  const result = await api(`/api/sessions/${state.id}/${action}`, payload(body));
  applyState(result);
  return result;
}

function pause() { playing = false; generation++; $('play').textContent = '▶'; }

function applyState(next, reset = false) {
  if (next.timeframe !== timeframe) return;
  const timeframeChanged = state && state.timeframe !== next.timeframe;
  state = next;
  $('price-source').textContent = next.price_source === 'trades' ?
    (next.trade_history_provenance ? 'Last traded price · recovered tape' : 'Last traded price · recorded tape') :
    'Legacy book midpoint · no trade tape';
  if (reset || timeframeChanged) { candles = []; chart.offset = 0; }
  if (reset) events = [];
  if (reset && next.calibration) {
    const observed=Object.entries(next.calibration.sources).find(([,view])=>view.quote.price!=null||view.trade.price!=null);
    if(observed){$('calibration-source').value=observed[0];$('calibration-stream').value=observed[1].quote.price!=null?'quote':'trade';}
  }
  const merged = new Map(candles.map(bar => [bar.time, bar]));
  for (const bar of next.candles || []) merged.set(bar.time, bar);
  candles = Array.from(merged.values()).sort((a,b) => a.time-b.time).slice(-2000);
  if(next.finished)events=[];
  events.push(...(next.events || []));
  $('empty-chart').style.display = candles.length ? 'none' : 'flex';
  $('bid-price').textContent = price(next.bid); $('ask-price').textContent = price(next.ask);
  $('spread').textContent = next.bid && next.ask ? price(Number(next.ask) - Number(next.bid)) : '—';
  $('equity').textContent = money(next.account.equity); $('open-pnl').textContent = money(next.account.unrealized_pnl);
  $('realized-pnl').textContent = money(next.account.realized_pnl); $('fees').textContent = money(next.account.fees);
  for (const [id, value] of [['open-pnl', next.account.unrealized_pnl], ['realized-pnl', next.account.realized_pnl]]) {
    $(id).classList.toggle('positive', Number(value) > 0); $(id).classList.toggle('negative', Number(value) < 0);
  }
  $('position-count').textContent = Number(next.account.position) ? '1' : '0';
  $('replay-clock').textContent = new Date(next.utc).toISOString().replace('T', ' ').slice(0, 19);
  $('progress-fill').style.width = `${next.progress * 100}%`; $('progress-text').textContent = `${(next.progress * 100).toFixed(1)}%`;
  $('capture-name').textContent = next.capture;
  $('capture-name').title = next.trade_history_provenance ? 'Production trade history; original trade receipt times are unavailable. Execution uses recorded API depth.' : next.price_source === 'trades' ? 'Recorded production transaction prices; execution uses recorded API depth.' : 'Legacy diagnostic recording: book midpoint is not the last traded price.';
  $('feed-health').textContent = next.finished ? 'Attempt complete' : next.book_valid ? 'Recorded API book available' : 'Waiting for fresh depth';
  $('feed-health').classList.toggle('fresh', next.book_valid && !next.finished);
  $('latency-label').textContent = `${lastResponseMs.toFixed(0)} ms round trip · ${next.server_ms?.toFixed(0) || '—'} ms engine`;
  $('export').disabled = !next.finished; $('finish').disabled = next.finished;
  if (next.finished) pause();
  const plan = next.plan;
  if (document.activeElement !== $('stop-price')) $('stop-price').value = plan?.stop ?? '';
  if (document.activeElement !== $('target-price')) $('target-price').value = plan?.target ?? '';
  const entry = next.layers.find(layer => layer.key === 'entry')?.price;
  if (entry && plan?.stop && plan?.target) {
    const risk = Math.abs(entry - Number(plan.stop)), reward = Math.abs(Number(plan.target) - entry);
    $('risk-reward').textContent = `${risk ? (reward / risk).toFixed(2) : '—'} R · ${money(risk * Math.abs(Number(next.account.position) || Number($('quantity').value)))}`;
  } else $('risk-reward').textContent = '—';
  renderCalibration(); chart.draw(); renderJournal();
}

function renderCalibration() {
  const source = $('calibration-source').value;
  const view = state?.calibration?.sources?.[source];
  const stream = view?.[$('calibration-stream').value];
  const values = $('calibration-values');
  $('save-calibration-label').disabled = !state || state.finished;
  if (!stream) { values.textContent = 'No reached observations.'; return; }
  const fmt = value => value == null ? '—' : Number(value).toFixed(2);
  const horizons = Object.entries(stream.horizons || {}).map(([seconds, row]) =>
    `<tr><td>${escapeHTML(seconds)}s</td><td>${fmt(row.speed_bps_s)}</td><td>${fmt(row.abs_percentile)}</td><td>${row.baseline_samples || 0}</td></tr>`).join('');
  const encounter = view.encounters?.at(-1);
  values.innerHTML = `<div class="calibration-status">${escapeHTML(stream.reason || 'Observed')} · age ${fmt(stream.age_s)}s</div>` +
    `<table><thead><tr><th>Window</th><th>bps/s</th><th>|v| %ile</th><th>Prior n</th></tr></thead><tbody>${horizons}</tbody></table>` +
    `<p>OR: ${escapeHTML(view.opening_range?.status || 'missing')} · ${escapeHTML(view.opening_range?.reason || state.calibration.config.range_start + ' / ' + state.calibration.config.range_minutes + 'm ' + state.calibration.config.session_timezone)}</p>` +
    (encounter ? `<p>${escapeHTML(encounter.level)} · ${escapeHTML(encounter.state.replaceAll('_', ' '))}<br>` +
      `Beyond band ${fmt(encounter.outside_s)}s · occupancy ${fmt(encounter.outside_fraction == null ? null : 100 * encounter.outside_fraction)}%</p>` : '<p>No eligible level encounter observed.</p>') +
    '<small>Provisional measurements. Percentile is not reversal odds. Overlay uses raw selected-market prices.</small>';
}

async function advance(seconds, jump = {}) {
  const token = generation;
  let result, target;
  do {
    result = await requestAction('advance', {seconds, ...(target == null ? jump : {target_ns: target})});
    target = result.work?.target_ns;
    if (result.work?.more) await new Promise(resolve => requestAnimationFrame(resolve));
  } while (result.work?.more && !result.finished && generation === token);
}

function togglePlay() {
  if (playing) { pause(); return; }
  if (!state || state.finished) { toast('Load a recording to practise', true); return; }
  playing = true; $('play').textContent = 'Ⅱ';
  async function tick() {
    if (!playing) return;
    await enqueue(() => advance(Number($('timestep').value)));
    if (playing) setTimeout(tick, 1000 / Number($('speed').value));
  }
  tick();
}

function action(name, body = {}) { pause(); return enqueue(() => requestAction(name, body)); }

function defaultLevels(side, reference) {
  const visible = candles.slice(-30);
  const spread = state.bid && state.ask ? Math.abs(Number(state.ask) - Number(state.bid)) : 0;
  const range = visible.length ? Math.max(...visible.map(b => b.high)) - Math.min(...visible.map(b => b.low)) : 0;
  const distance = Math.max(reference * 0.002, range / 2, spread * 2);
  const direction = side === 'buy' ? 1 : -1;
  return {stop: Number((reference - distance * direction).toFixed(1)), target: Number((reference + distance * 2 * direction).toFixed(1))};
}

function place(side, overridePrice = null) {
  if (!state) { toast('Open a recording first', true); return; }
  const kind = overridePrice != null ? 'limit' : orderType;
  const reference = kind === 'limit' ? Number(overridePrice ?? $('entry').value) : Number(side === 'buy' ? state.ask : state.bid);
  if (!reference) { toast('No executable quote yet. Step forward.', true); return; }
  const levels = defaultLevels(side, reference);
  const body = {side, quantity: $('quantity').value, order_type: kind, price: kind === 'limit' ? reference : null,
                ...levels, reason: $('note').value.trim() || `Chart ${side} entry`};
  action('order', body).then(() => { $('note').value = ''; chart.offset = 0; });
}

function saveNote(kind = 'note') {
  const reason = $('note').value.trim();
  if (!reason) { $('note').focus(); toast('Add a short reason for your decision'); return; }
  action('note', {action: kind, reason}).then(() => { $('note').value = ''; toast('Decision saved'); });
}

function table(headers, rows) {
  return `<table><thead><tr>${headers.map(h => `<th>${escapeHTML(h)}</th>`).join('')}</tr></thead><tbody>${rows.map(row => `<tr>${row.map(cell => `<td>${cell}</td>`).join('')}</tr>`).join('')}</tbody></table>`;
}

function renderJournal() {
  if (!state) return;
  let html;
  if (tab === 'positions') {
    const account = state.account, size = Number(account.position);
    html = size ? table(['Symbol','Side','Quantity','Entry','Stop','Target','Open P&L'], [[
      'BTC-USD-PERP', `<span class="${size > 0 ? 'positive' : 'negative'}">${size > 0 ? 'LONG' : 'SHORT'}</span>`,
      Math.abs(size), price(account.average_entry), price(state.plan?.stop), price(state.plan?.target),
      `<span class="${Number(account.unrealized_pnl) >= 0 ? 'positive' : 'negative'}">${money(account.unrealized_pnl)}</span>`]])
      : '<div class="table-empty">No open position. Buy or sell to practise, or record why you are waiting.</div>';
  } else if (tab === 'orders') {
    const rows = state.orders.filter(order => order.actor === 'human').slice().reverse();
    html = rows.length ? table(['ID','Side','Type','Quantity','Remaining','Price','Status'], rows.map(o => [o.id, escapeHTML(o.side), escapeHTML(o.order_type), escapeHTML(o.quantity), escapeHTML(o.remaining), price(o.price), `<span class="tag">${escapeHTML(o.status)}</span>`])) : '<div class="table-empty">No orders placed.</div>';
  } else if (tab === 'journal') {
    const rows = events.filter(e => e.actor === 'human' && !['drawing'].includes(e.action)).slice(-100).reverse();
    html = rows.length ? table(['Replay time','Action','Details'], rows.map(e => [escapeHTML(e.utc.slice(11, 23)), escapeHTML(e.action.replaceAll('_',' ')), escapeHTML(eventDetails(e))])) : '<div class="table-empty">Your decisions and fills are saved here. Press N to add your read.</div>';
  } else {
    html = !state.finished ? '<div class="table-empty">Bot decisions stay hidden during practice. Choose Finish & review when you are ready.</div>' :
      table(['Trader','Position','Realized P&L','Open P&L','Fees','Equity'], Object.entries(state.comparison || {}).map(([actor, a]) => [escapeHTML(actor), escapeHTML(a.position), money(a.realized_pnl), money(a.unrealized_pnl), money(a.fees), money(a.equity)])) +
      table(['Time','Trader','Action','Decision'], events.filter(e => ['order','fill','wait','skip','move_level','calibration_label'].includes(e.action)).slice(-100).map(e => [escapeHTML(e.utc.slice(11,19)), escapeHTML(e.actor), escapeHTML(e.action), escapeHTML(eventDetails(e))]));
  }
  $('journal-content').innerHTML = html;
}

function eventDetails(e) {
  if(e.action==='calibration_label')return `${e.source} / ${e.stream} · ${e.label.replaceAll('_',' ')} · ${e.reason}`;
  return e.rationale || (e.action==='fill' ? `${e.side} ${e.quantity} @ ${price(e.price)}` : JSON.stringify(e.order || e));
}

class CandleChart {
  constructor(canvas) {
    this.canvas = canvas; this.ctx = canvas.getContext('2d'); this.count = 105; this.offset = 0;
    this.tool = 'cursor'; this.cross = null; this.drag = null; this.trendStart = null;
    new ResizeObserver(() => this.resize()).observe(canvas.parentElement);
    canvas.addEventListener('pointerdown', e => this.down(e));
    canvas.addEventListener('pointermove', e => this.move(e));
    canvas.addEventListener('pointerup', e => this.up(e));
    canvas.addEventListener('pointercancel', () => { this.drag = null; });
    canvas.addEventListener('pointerleave', () => { if (!this.drag) { this.cross = null; this.draw(); } });
    canvas.addEventListener('wheel', e => { e.preventDefault(); this.count = Math.max(15, Math.min(600, this.count * (e.deltaY > 0 ? 1.13 : 0.88))); this.draw(); }, {passive: false});
    canvas.addEventListener('contextmenu', e => { e.preventDefault(); this.context(e); });
    canvas.addEventListener('dblclick', e => this.removeDrawing(e));
  }
  resize() {
    this.width = this.canvas.parentElement.clientWidth; this.height = this.canvas.parentElement.clientHeight;
    const ratio = window.devicePixelRatio || 1; this.canvas.width = this.width * ratio; this.canvas.height = this.height * ratio;
    this.ctx.setTransform(ratio, 0, 0, ratio, 0, 0); this.draw();
  }
  point(e) { const rect = this.canvas.getBoundingClientRect(); return {x: e.clientX - rect.left, y: e.clientY - rect.top}; }
  coordinate(point) {
    if (!this.visible?.length) return null;
    const index = Math.max(0, Math.min(this.visible.length - 1, Math.round((point.x - 12) / this.step)));
    return {time: this.visible[index].time, price: this.unprice(point.y)};
  }
  unprice(y) { return this.max - (y - 12) / Math.max(1, this.plotHeight) * (this.max - this.min); }
  y(value) { return 12 + (this.max - Number(value)) / (this.max - this.min) * this.plotHeight; }
  xTime(time) {
    if (!this.visible?.length) return 0;
    let low = 0, high = this.visible.length - 1;
    while (low < high) { const mid = Math.floor((low + high) / 2); if (this.visible[mid].time < time) low = mid + 1; else high = mid; }
    return 12 + low * this.step;
  }
  draw() {
    if (!this.width || !this.height) return;
    const c = this.ctx, w = this.width, h = this.height, right = w - 83;
    c.clearRect(0,0,w,h); c.fillStyle = '#0d1117'; c.fillRect(0,0,w,h);
    this.offset = Math.max(0, Math.min(this.offset, Math.max(0,candles.length - 8)));
    const end = candles.length - Math.round(this.offset), start = Math.max(0, end - Math.round(this.count));
    const rows = candles.slice(start,end); this.visible = rows; this.plotHeight = Math.max(1,h - 46); this.step = (right - 12) / (this.count + 10);
    if (!rows.length) return;
    let low = Math.min(...rows.map(b => b.low)), high = Math.max(...rows.map(b => b.high));
    const levels = state?.layers || [];
    for (const level of levels) { low = Math.min(low,level.price); high = Math.max(high,level.price); }
    const span = Math.max(high - low, high * 0.0003);
    this.min = this.drag?.kind==='level' ? this.drag.min : low - span * 0.16;
    this.max = this.drag?.kind==='level' ? this.drag.max : high + span * 0.16;
    c.font = '10px ui-monospace, monospace'; c.textAlign = 'left'; c.textBaseline = 'middle';
    c.strokeStyle = '#1b2330'; c.lineWidth = 1;
    for (let i = 0; i <= 6; i++) {
      const y = 12 + this.plotHeight * i/6, value = this.max - (this.max - this.min)*i/6;
      c.beginPath(); c.moveTo(0,Math.round(y)+.5); c.lineTo(right,y); c.stroke(); c.fillStyle = '#67768b'; c.fillText(price(value),right+10,y);
    }
    const labelEvery = Math.max(1,Math.floor(this.count/7));
    rows.forEach((bar,index) => {
      const x = 12 + index*this.step;
      if (index % labelEvery === 0) {
        c.strokeStyle='#18202b'; c.beginPath(); c.moveTo(x,0); c.lineTo(x,h-29); c.stroke();
        c.fillStyle='#637389'; c.fillText(new Date(bar.time*1000).toISOString().slice(11,timeframe<60?19:16),x-15,h-14);
      }
      const up = bar.close >= bar.open, color = up?'#55c9a9':'#df748a'; c.strokeStyle=color; c.fillStyle=color;
      c.beginPath(); c.moveTo(Math.round(x)+.5,this.y(bar.high)); c.lineTo(Math.round(x)+.5,this.y(bar.low)); c.stroke();
      const bodyWidth=Math.max(1,Math.min(14,this.step*.65)), bodyTop=this.y(Math.max(bar.open,bar.close));
      c.fillRect(Math.round(x-bodyWidth/2),bodyTop,Math.round(bodyWidth),Math.max(1,Math.abs(this.y(bar.open)-this.y(bar.close))));
    });
    c.save(); c.beginPath(); c.rect(0,0,right,h-29); c.clip();
    const range=state?.opening_range;
    if (showRange && range?.high != null && range?.low != null) {
      const y1=this.y(range.high),y2=this.y(range.low);
      c.fillStyle='#789cff08';c.fillRect(0,y1,right,y2-y1);
      this.line(range.high,'#637dad','OR HIGH',right,false,true);this.line(range.low,'#637dad','OR LOW',right,false,true);
    }
    const featureRange = state?.calibration?.sources?.[$('calibration-source').value]?.opening_range;
    if (showRange && featureRange?.status === 'complete') {
      const prefix = $('calibration-source').selectedOptions[0].textContent;
      for (const [name, value] of Object.entries(featureRange.levels || {})) {
        this.line(value, '#b29adb', `${prefix} ${name}`, right, false, true);
      }
    }
    for(const drawing of state?.drawings || []) this.drawing(drawing);
    if(this.trendStart && this.cross) this.drawing({kind:'trend',a:this.trendStart,b:this.coordinate(this.cross)});
    const entry=levels.find(l=>l.key==='entry'),stop=levels.find(l=>l.key==='stop'),target=levels.find(l=>l.key==='target');
    const tradeX=12+Math.max(0,rows.length-20)*this.step;
    if(entry&&stop){c.fillStyle='#b7597214';c.fillRect(tradeX,Math.min(this.y(entry.price),this.y(stop.price)),right-tradeX,Math.abs(this.y(entry.price)-this.y(stop.price)));}
    if(entry&&target){c.fillStyle='#3d9d8513';c.fillRect(tradeX,Math.min(this.y(entry.price),this.y(target.price)),right-tradeX,Math.abs(this.y(entry.price)-this.y(target.price)));}
    for(const e of events.filter(e=>e.action==='fill')) {
      const t=Math.floor(Date.parse(e.utc)/1000/timeframe)*timeframe;
      if(t<rows[0].time||t>rows.at(-1).time)continue;
      c.fillStyle=e.actor==='bot'?'#c8acff':e.side==='buy'?'#6fe3b5':'#ff98ac';
      c.beginPath();c.arc(this.xTime(t),this.y(Number(e.price)),3,0,Math.PI*2);c.fill();
    }
    c.restore();
    if(rows.length) this.line(rows.at(-1).close,rows.at(-1).close>=rows.at(-1).open?'#55c9a9':'#df748a',price(rows.at(-1).close),right,true,true);
    for(const level of levels) this.line(level.price,level.key==='stop'?'#dc8298':level.key==='target'?'#68ccad':'#8eaaff',`${level.draggable?'↕ ':''}${level.label}  ${price(level.price)}`,right,true,false);
    if(this.cross && this.cross.x<right && this.cross.y<h-29) {
      c.setLineDash([3,4]);c.strokeStyle='#65799088';c.beginPath();c.moveTo(this.cross.x,0);c.lineTo(this.cross.x,h-29);c.moveTo(0,this.cross.y);c.lineTo(right,this.cross.y);c.stroke();c.setLineDash([]);
      c.fillStyle='#34455d';c.fillRect(right,this.cross.y-10,83,20);c.fillStyle='#d7e4f7';c.fillText(price(this.unprice(this.cross.y)),right+7,this.cross.y);
      const index=Math.max(0,Math.min(rows.length-1,Math.round((this.cross.x-12)/this.step))),bar=rows[index];
      $('bar-values').textContent=`O ${price(bar.open)}   H ${price(bar.high)}   L ${price(bar.low)}   C ${price(bar.close)}`;
    } else {const b=rows.at(-1);$('bar-values').textContent=`O ${price(b.open)}   H ${price(b.high)}   L ${price(b.low)}   C ${price(b.close)}`;}
  }
  line(value,color,label,right,box=false,dashed=false) {
    const c=this.ctx,y=this.y(value);if(y<0||y>this.height-29)return;
    c.strokeStyle=color;c.lineWidth=1;c.setLineDash(dashed?[3,4]:[]);c.beginPath();c.moveTo(0,Math.round(y)+.5);c.lineTo(right,y);c.stroke();c.setLineDash([]);
    if(box){const width=Math.max(77,c.measureText(label).width+15);c.fillStyle=color;c.fillRect(right-width+77,y-10,width,20);c.fillStyle='#111822';c.fillText(label,right-width+84,y);}
    else{c.fillStyle=color;c.fillText(label,9,y-9);}
  }
  drawing(d) {
    if(!d)return;const c=this.ctx;c.strokeStyle='#c2a16d';c.lineWidth=1;
    if(d.kind==='horizontal'){c.setLineDash([6,4]);c.beginPath();c.moveTo(0,this.y(d.price));c.lineTo(this.width-83,this.y(d.price));c.stroke();c.setLineDash([]);}
    else if(d.a&&d.b){c.beginPath();c.moveTo(this.xTime(d.a.time),this.y(d.a.price));c.lineTo(this.xTime(d.b.time),this.y(d.b.price));c.stroke();}
  }
  down(e) {
    if(e.button!==0||!state||!this.visible.length)return;
    pause();const p=this.point(e);this.canvas.setPointerCapture(e.pointerId);
    const hit=state.layers.find(l=>l.draggable&&Math.abs(this.y(l.price)-p.y)<8);
    if(hit){this.drag={kind:'level',key:hit.key,original:hit.price,min:this.min,max:this.max};return;}
    const coordinate=this.coordinate(p);
    if(this.tool==='horizontal'){state.drawings.push({kind:'horizontal',price:coordinate.price});this.persistDrawings();return;}
    if(this.tool==='trend'){if(!this.trendStart)this.trendStart=coordinate;else{state.drawings.push({kind:'trend',a:this.trendStart,b:coordinate});this.trendStart=null;this.persistDrawings();}this.draw();return;}
    const drawing=state.drawings.find(d=>d.kind==='horizontal'&&Math.abs(this.y(d.price)-p.y)<6);
    this.drag=drawing?{kind:'drawing',drawing}:{kind:'pan',x:p.x,offset:this.offset};
  }
  move(e) {
    const p=this.point(e);this.cross=p;
    if(this.drag?.kind==='level'){const layer=state.layers.find(l=>l.key===this.drag.key);if(layer)layer.price=Number(this.unprice(p.y).toFixed(1));}
    else if(this.drag?.kind==='drawing')this.drag.drawing.price=this.unprice(p.y);
    else if(this.drag?.kind==='pan')this.offset=this.drag.offset+(p.x-this.drag.x)/this.step;
    this.draw();
  }
  up() {
    if(this.drag?.kind==='level'){
      const field=this.drag.key,level=state.layers.find(l=>l.key===field),original=this.drag.original;
      if(level)action('level',{field,price:level.price}).then(()=>chart.draw());
      if(level)level.price=original;
    } else if(this.drag?.kind==='drawing')this.persistDrawings();
    this.drag=null;this.draw();
  }
  persistDrawings(){this.draw();action('drawing',{drawings:state.drawings});}
  removeDrawing(e){if(!state)return;const p=this.point(e);const index=state.drawings.findIndex(d=>d.kind==='horizontal'&&Math.abs(this.y(d.price)-p.y)<7);if(index>=0){state.drawings.splice(index,1);this.persistDrawings();}}
  context(e){
    if(!state||!this.visible.length)return;pause();const at=this.coordinate(this.point(e));
    document.getElementById('context-menu')?.remove();const menu=document.createElement('div');menu.id='context-menu';
    Object.assign(menu.style,{position:'fixed',left:`${Math.min(e.clientX,window.innerWidth-200)}px`,top:`${Math.min(e.clientY,window.innerHeight-130)}px`,background:'#192334',border:'1px solid #3b4c67',padding:'6px',borderRadius:'7px',zIndex:50,boxShadow:'0 8px 35px #0008'});
    for(const side of ['buy','sell']){const button=document.createElement('button');button.textContent=`${side==='buy'?'Buy':'Sell'} limit @ ${price(at.price)}`;button.style.display='block';button.onclick=()=>{menu.remove();place(side,Number(at.price.toFixed(1)));};menu.append(button);}
    const line=document.createElement('button');line.textContent='Draw horizontal level';line.onclick=()=>{state.drawings.push({kind:'horizontal',price:at.price});menu.remove();this.persistDrawings();};menu.append(line);document.body.append(menu);
    setTimeout(()=>document.addEventListener('pointerdown',event=>{if(!menu.contains(event.target))menu.remove();},{once:true}),0);
  }
}

const chart = new CandleChart($('chart'));

async function recordings() {
  pause();const result=await api('/api/captures');
  const options=result.captures;
  currentCapture=currentCapture || result.default || options[0]?.id;
  $('recording-list').innerHTML=options.length?options.map(item=>`<label class="recording-option"><input type="radio" name="capture" value="${escapeHTML(item.id)}" ${item.id===currentCapture?'checked':''}><div><span>${escapeHTML(item.name)}</span><small>${escapeHTML(item.price_source || 'legacy quotes').toUpperCase()} · ${item.closed?(item.checkpoint?'CHECKPOINT':'CLOSED'):'RECORDING NOW'} · ${(item.bytes/1048576).toFixed(1)} MB</small></div></label>`).join(''):'<p>No capture yet. Start the public Paradex recorder first.</p>';
  $('recording-dialog').showModal();
}

async function loadRecording() {
  const selected=document.querySelector('input[name="capture"]:checked');if(!selected){toast('Choose a recording',true);return;}
  currentCapture=selected.value;pause();$('recording-dialog').close();$('loading').hidden=false;
  try {
    await enqueue(async()=>{
      if(state&&!state.finished)await api(`/api/sessions/${state.id}/close`,{});
      const body={capture:currentCapture,timeframe,warmup_seconds:Number($('warmup').value),config:{session_timezone:$('session-zone').value,range_start:$('session-start').value,range_minutes:Number($('range-minutes').value),order_latency_ms:Number($('order-latency').value),fee_bps:$('fee-bps').value,starting_equity:$('starting-equity').value}};
      const next=await api('/api/sessions',body);currentCapture=next.capture_path;chart.count=Math.min(120,Math.max(35,next.candles.length+5));applyState(next,true);toast(`Practice ready · ${(next.load_ms/1000).toFixed(2)}s load${next.index_built?' · index built':' · cached index'}`);
    });
  } finally {$('loading').hidden=true;}
}

$('open-recording').onclick=()=>recordings().catch(e=>toast(e.message,true));$('empty-load').onclick=$('open-recording').onclick;
$('load-recording').onclick=loadRecording;$('settings-button').onclick=()=>{$('settings-dialog').showModal();pause();};$('settings-done').onclick=()=>$('settings-dialog').close();
$('help-button').onclick=()=>$('help-dialog').showModal();$('play').onclick=togglePlay;
$('step').onclick=()=>{pause();enqueue(()=>advance(Number($('timestep').value)));};
$('buy').onclick=()=>place('buy');$('sell').onclick=()=>place('sell');$('close-position').onclick=()=>action('flatten');$('cancel-orders').onclick=()=>action('cancel');
$('save-note').onclick=()=>saveNote();$('wait-note').onclick=()=>saveNote('wait');$('skip-note').onclick=()=>saveNote('skip');
$('calibration-source').onchange=()=>{renderCalibration();chart.draw();};
$('calibration-stream').onchange=renderCalibration;
$('save-calibration-label').onclick=()=>{
  const reason=$('note').value.trim();
  if(!reason){$('note').focus();toast('Add your reason, then save the interpretation');return;}
  action('calibration-label',{label:$('calibration-label').value,source:$('calibration-source').value,
    stream:$('calibration-stream').value,expected_clock_ns:state.clock_ns,
    expected_sequence:state.calibration.capture_sequence,reason})
    .then(()=>{$('note').value='';toast('Interpretation and current features saved');});
};
$('finish').onclick=()=>{if(!state)return;action('finish').then(()=>{tab='review';document.querySelectorAll('[data-tab]').forEach(b=>b.classList.toggle('selected',b.dataset.tab===tab));renderJournal();toast('Attempt saved. Compare your decisions with the bot.');});};
$('export').onclick=()=>{if(state?.finished)window.location.href=`/api/sessions/${state.id}/export`;};
$('restart').onclick=()=>{if(!state)return;recordings().catch(e=>toast(e.message,true));};
$('fit-chart').onclick=()=>{chart.offset=0;chart.count=105;chart.draw();};
$('go-to').onclick=()=>{if(!state)return;pause();$('goto-time').value=new Date(Date.parse(state.utc)+60000).toISOString().slice(0,19);$('goto-dialog').showModal();};
$('goto-date').onclick=()=>{$('goto-dialog').close();pause();enqueue(()=>advance(0,{utc:$('goto-time').value+'Z'}));};
$('goto-range').onclick=()=>{$('goto-dialog').close();pause();enqueue(()=>advance(0,{jump:'opening_range'}));};
$('toggle-range').onclick=()=>{showRange=!showRange;$('toggle-range').classList.toggle('range-on',showRange);chart.draw();};
$('clear-drawings').onclick=()=>{if(state){state.drawings=[];chart.persistDrawings();}};
for(const field of ['stop','target'])$(field+'-price').onchange=()=>{if(state)action('level',{field,price:$(field+'-price').value});};
document.querySelectorAll('[data-tf]').forEach(button=>button.onclick=()=>{
  pause();
  enqueue(async()=>{
    const previousTimeframe=timeframe;
    timeframe=Number(button.dataset.tf);
    try {
      if(state)await requestAction('state',{since:0});
    } catch(error) {
      timeframe=previousTimeframe;
      throw error;
    } finally {
      document.querySelectorAll('[data-tf]').forEach(b=>b.classList.toggle('selected',Number(b.dataset.tf)===timeframe));
    }
  });
});
document.querySelectorAll('[data-order]').forEach(button=>button.onclick=()=>{orderType=button.dataset.order;document.querySelectorAll('[data-order]').forEach(b=>b.classList.toggle('selected',b===button));$('entry-label').hidden=orderType!=='limit';if(orderType==='limit'&&state)$('entry').value=Number(state.last_price).toFixed(1);});
document.querySelectorAll('[data-tab]').forEach(button=>button.onclick=()=>{tab=button.dataset.tab;document.querySelectorAll('[data-tab]').forEach(b=>b.classList.toggle('selected',b===button));renderJournal();});
function selectTool(tool){chart.tool=tool;chart.trendStart=null;document.querySelectorAll('[data-tool]').forEach(b=>b.classList.toggle('active',b.dataset.tool===tool));chart.draw();}
document.querySelectorAll('[data-tool]').forEach(button=>button.onclick=()=>selectTool(button.dataset.tool));
document.addEventListener('keydown',event=>{
  if(['INPUT','TEXTAREA','SELECT'].includes(event.target.tagName)||document.querySelector('dialog[open]'))return;
  const key=event.key.toLowerCase();
  if(event.code==='Space'){event.preventDefault();togglePlay();}else if(key==='arrowright'){event.preventDefault();$('step').click();}
  else if(key==='b')place('buy');else if(key==='s')place('sell');else if(key==='x')action('flatten');else if(key==='h')selectTool('horizontal');else if(key==='t')selectTool('trend');else if(key==='escape')selectTool('cursor');else if(key==='f')$('fit-chart').click();else if(key==='n'){$('note').focus();pause();}
});
window.addEventListener('unhandledrejection',event=>{event.preventDefault();toast(event.reason?.message||'Replay request failed',true);});
api('/api/captures').then(result=>{currentCapture=result.default;if(result.captures.length){$('capture-name').textContent=`${result.captures.length} recordings available · open one to start`;$('empty-chart').querySelector('p').textContent='Your recorded Paradex market is ready. Choose a session and take control.';}}).catch(e=>toast(e.message,true));
