const elements = {
  connection: document.querySelector('#connection'),
  controllerState: document.querySelector('#controller-state'),
  mode: document.querySelector('#mode'),
  arbitrationSource: document.querySelector('#arbitration-source'),
  recorderState: document.querySelector('#recorder-state'),
  emergencyState: document.querySelector('#emergency-state'),
  lastEvent: document.querySelector('#last-event'),
  automationState: document.querySelector('#automation-state'),
  error: document.querySelector('#error'),
  emergencyStop: document.querySelector('#emergency-stop'),
  pumpButtons: [...document.querySelectorAll('.pump')],
  scheduleForm: document.querySelector('#schedule-form'),
  order: document.querySelector('#order'),
  burstSeconds: document.querySelector('#burst-seconds'),
  totalMinutes: document.querySelector('#total-minutes'),
  startSchedule: document.querySelector('#start-schedule'),
  stopSchedule: document.querySelector('#stop-schedule'),
  scheduleProgress: document.querySelector('#schedule-progress'),
  activePump: document.querySelector('#active-pump'),
  elapsed: document.querySelector('#elapsed'),
  remaining: document.querySelector('#remaining'),
  progressFill: document.querySelector('#progress-fill'),
  toggleAutomations: document.querySelector('#toggle-automations'),
  localClock: document.querySelector('#local-clock'),
  nextAutomation: document.querySelector('#next-automation'),
  automationList: document.querySelector('#automation-list'),
  automationForm: document.querySelector('#automation-form'),
  automationId: document.querySelector('#automation-id'),
  automationName: document.querySelector('#automation-name'),
  automationStart: document.querySelector('#automation-start'),
  automationEnd: document.querySelector('#automation-end'),
  automationOn: document.querySelector('#automation-on'),
  automationBlock: document.querySelector('#automation-block'),
  automationEnabled: document.querySelector('#automation-enabled'),
  cancelAutomation: document.querySelector('#cancel-automation'),
  tabs: [...document.querySelectorAll('.tab')],
  controlsView: document.querySelector('#controls-view'),
  visualizationView: document.querySelector('#visualization-view'),
  logView: document.querySelector('#log-view'),
  visualizationDate: document.querySelector('#visualization-date'),
  previousDay: document.querySelector('#previous-day'),
  today: document.querySelector('#today'),
  nextDay: document.querySelector('#next-day'),
  visualizationSummary: document.querySelector('#visualization-summary'),
  selectedDayLabel: document.querySelector('#selected-day-label'),
  visualizationCurrentOutput: document.querySelector('#visualization-current-output'),
  visualizationNextWindow: document.querySelector('#visualization-next-window'),
  visualizationAutomationState: document.querySelector('#visualization-automation-state'),
  timelineHours: document.querySelector('#timeline-hours'),
  timelineRows: document.querySelector('#timeline-rows'),
  visualizationEmpty: document.querySelector('#visualization-empty'),
  logHealth: document.querySelector('#log-health'),
  logFilterForm: document.querySelector('#log-filter-form'),
  logPump: document.querySelector('#log-pump'),
  logFrom: document.querySelector('#log-from'),
  logTo: document.querySelector('#log-to'),
  clearLogFilters: document.querySelector('#clear-log-filters'),
  exportLogCsv: document.querySelector('#export-log-csv'),
  logResultCount: document.querySelector('#log-result-count'),
  logOpenCount: document.querySelector('#log-open-count'),
  logTableBody: document.querySelector('#log-table-body'),
  logPrevious: document.querySelector('#log-previous'),
  logNext: document.querySelector('#log-next'),
  logPage: document.querySelector('#log-page'),
};

let latestStatus = null;
let requestPending = false;
const dayNames = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
const pumpColors = ['pump-color-1', 'pump-color-2', 'pump-color-3', 'pump-color-4'];

function displayPumpNumber(pump) {
  if (pump === 1) return 4;
  if (pump === 4) return 1;
  return pump;
}

function localDateKey(date) {
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, '0');
  const day = String(date.getDate()).padStart(2, '0');
  return `${year}-${month}-${day}`;
}

function dateFromKey(key) {
  const [year, month, day] = key.split('-').map(Number);
  return new Date(year, month - 1, day);
}

let visualizationDateKey = localDateKey(new Date());
elements.visualizationDate.value = visualizationDateKey;
let activeTab = 'controls';
let logOffset = 0;
const logLimit = 100;

async function api(path, body = null) {
  const options = { cache: 'no-store' };
  if (body !== null) {
    options.method = 'POST';
    options.headers = { 'Content-Type': 'application/json' };
    options.body = JSON.stringify(body);
  }
  const response = await fetch(path, options);
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || `Request failed (${response.status})`);
  }
  return payload;
}

function showError(message) {
  elements.error.textContent = message;
  elements.error.hidden = !message;
}

function formatDuration(seconds) {
  const rounded = Math.max(0, Math.ceil(Number(seconds) || 0));
  const minutes = Math.floor(rounded / 60);
  return `${minutes}:${String(rounded % 60).padStart(2, '0')}`;
}

function formatDateTime(value) {
  if (!value) return '—';
  const date = new Date(value);
  return Number.isNaN(date.getTime())
    ? '—'
    : date.toLocaleString([], { weekday: 'short', hour: 'numeric', minute: '2-digit' });
}

function selectedValues(name) {
  return [...document.querySelectorAll(`input[name="${name}"]:checked`)]
    .map((input) => Number(input.value));
}

function resetAutomationForm() {
  elements.automationForm.reset();
  elements.automationId.value = '';
  elements.automationName.value = 'Day dosing';
  elements.automationStart.value = '00:00';
  elements.automationEnd.value = '18:00';
  elements.automationOn.value = '5';
  elements.automationBlock.value = '25';
  elements.automationEnabled.checked = true;
  for (const input of document.querySelectorAll('[name="automation-pump"], [name="weekday"]')) {
    input.checked = true;
  }
  elements.cancelAutomation.hidden = true;
}

function editAutomation(rule) {
  elements.automationId.value = rule.id;
  elements.automationName.value = rule.name;
  elements.automationStart.value = rule.start_time;
  elements.automationEnd.value = rule.end_time;
  elements.automationOn.value = rule.on_minutes;
  elements.automationBlock.value = rule.block_minutes;
  elements.automationEnabled.checked = rule.enabled;
  for (const input of document.querySelectorAll('[name="automation-pump"]')) {
    const bit = 1 << (Number(input.value) - 1);
    input.checked = Boolean(rule.mask & bit);
  }
  for (const input of document.querySelectorAll('[name="weekday"]')) {
    input.checked = rule.weekdays.includes(Number(input.value));
  }
  elements.cancelAutomation.hidden = false;
  elements.automationForm.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

function renderAutomationList(automation) {
  elements.automationList.replaceChildren();
  if (!automation.rules?.length) {
    const empty = document.createElement('p');
    empty.className = 'empty-state';
    empty.textContent = 'No saved automations.';
    elements.automationList.append(empty);
    return;
  }

  for (const rule of automation.rules) {
    const card = document.createElement('article');
    card.className = `automation-card${rule.active ? ' active' : ''}${rule.enabled ? '' : ' disabled'}`;

    const header = document.createElement('div');
    header.className = 'automation-card-header';
    const title = document.createElement('h4');
    title.textContent = rule.name;
    const state = document.createElement('span');
    state.className = 'rule-status';
    state.textContent = rule.active ? 'RUNNING' : (rule.enabled ? 'READY' : 'DISABLED');
    header.append(title, state);

    const summary = document.createElement('p');
    const pumps = [1, 2, 3, 4].filter((pump) => rule.mask & (1 << (pump - 1)));
    const days = rule.weekdays.map((day) => dayNames[day]).join(' ');
    summary.textContent = `Pumps ${pumps.map(displayPumpNumber).join(', ')} together · ${days} · ${rule.start_time}–${rule.end_time} · wait ${rule.block_minutes} min, then ON ${rule.on_minutes} min`;

    const footer = document.createElement('div');
    footer.className = 'automation-card-footer';
    const next = document.createElement('span');
    next.textContent = rule.active
      ? `Until ${formatDateTime(rule.active_until)}`
      : `Next ${formatDateTime(rule.next_start)}`;
    const actions = document.createElement('div');
    actions.className = 'automation-card-actions';
    const edit = document.createElement('button');
    edit.type = 'button';
    edit.className = 'secondary';
    edit.textContent = 'Edit';
    edit.addEventListener('click', () => editAutomation(rule));
    const enabled = document.createElement('button');
    enabled.type = 'button';
    enabled.className = 'secondary';
    enabled.textContent = rule.enabled ? 'Disable' : 'Enable';
    enabled.addEventListener('click', () => mutate('/api/automations/enabled', {
      id: rule.id,
      enabled: !rule.enabled,
    }));
    const remove = document.createElement('button');
    remove.type = 'button';
    remove.className = 'danger';
    remove.textContent = 'Delete';
    remove.addEventListener('click', () => {
      if (window.confirm(`Delete automation "${rule.name}"?`)) {
        mutate('/api/automations/delete', { id: rule.id });
      }
    });
    actions.append(edit, enabled, remove);
    footer.append(next, actions);
    card.append(header, summary, footer);
    elements.automationList.append(card);
  }
}

function minuteLabel(totalMinutes) {
  const bounded = Math.max(0, Math.min(1440, Math.round(totalMinutes)));
  if (bounded === 1440) return '24:00';
  return `${String(Math.floor(bounded / 60)).padStart(2, '0')}:${String(bounded % 60).padStart(2, '0')}`;
}

function minutesFromTime(value) {
  const [hours, minutes] = value.split(':').map(Number);
  return hours * 60 + minutes;
}

function renderTimelineHours() {
  elements.timelineHours.replaceChildren();
  for (let hour = 0; hour <= 24; hour += 3) {
    const label = document.createElement('span');
    label.className = 'hour-label';
    label.style.left = `${(hour / 24) * 100}%`;
    label.textContent = hour === 24 ? '24:00' : `${String(hour).padStart(2, '0')}:00`;
    elements.timelineHours.append(label);
  }
}

function renderScheduleVisualization(automation, status) {
  const selectedDate = dateFromKey(visualizationDateKey);
  if (Number.isNaN(selectedDate.getTime())) return;
  const weekday = (selectedDate.getDay() + 6) % 7;
  const selectedRules = (automation.rules || []).filter(
    (rule) => rule.enabled && rule.weekdays.includes(weekday),
  );
  const localNow = automation.local_time ? new Date(automation.local_time) : new Date();
  const todayKey = localDateKey(localNow);
  const isToday = visualizationDateKey === todayKey;
  const nowMinute = localNow.getHours() * 60 + localNow.getMinutes()
    + localNow.getSeconds() / 60;

  elements.selectedDayLabel.textContent = selectedDate.toLocaleDateString([], {
    weekday: 'long',
    month: 'short',
    day: 'numeric',
    year: 'numeric',
  });
  const activePumps = [1, 2, 3, 4].filter(
    (pump) => status.acknowledged_mask & (1 << (pump - 1)),
  );
  elements.visualizationCurrentOutput.textContent = activePumps.length
    ? `Pump${activePumps.length > 1 ? 's' : ''} ${activePumps.map(displayPumpNumber).join(', ')} ON`
    : 'All pumps OFF';
  elements.visualizationNextWindow.textContent = formatDateTime(automation.next_start);
  elements.visualizationAutomationState.textContent = automation.last_error
    ? 'ERROR'
    : (automation.paused ? 'PAUSED' : 'RUNNING');

  const ruleWord = selectedRules.length === 1 ? 'automation' : 'automations';
  elements.visualizationSummary.textContent = automation.paused
    ? `${selectedRules.length} enabled ${ruleWord} shown; output is globally paused.`
    : `${selectedRules.length} enabled ${ruleWord} scheduled for this day.`;

  elements.timelineRows.replaceChildren();
  let blockCount = 0;
  for (let pump = 1; pump <= 4; pump += 1) {
    const row = document.createElement('div');
    row.className = 'timeline-row';
    const rowLabel = document.createElement('div');
    rowLabel.className = 'timeline-row-label';
    const title = document.createElement('strong');
    title.textContent = `Pump ${displayPumpNumber(pump)}`;
    const count = document.createElement('span');
    const pumpRules = selectedRules.filter((rule) => rule.mask & (1 << (pump - 1)));
    count.textContent = `${pumpRules.length} rule${pumpRules.length === 1 ? '' : 's'}`;
    rowLabel.append(title, count);

    const track = document.createElement('div');
    track.className = 'timeline-track';
    for (const rule of pumpRules) {
      const start = minutesFromTime(rule.start_time);
      const end = minutesFromTime(rule.end_time);
      const cycleMinutes = rule.block_minutes + rule.on_minutes;
      for (
        let blockStart = start + rule.block_minutes;
        blockStart + rule.on_minutes <= end && blockCount < 8000;
        blockStart += cycleMinutes
      ) {
        const duration = rule.on_minutes;
        const blockEnd = blockStart + duration;
        const block = document.createElement('div');
        block.className = `timeline-block ${pumpColors[pump - 1]}`;
        block.style.left = `${(blockStart / 1440) * 100}%`;
        block.style.width = `${(duration / 1440) * 100}%`;
        block.title = `${rule.name}: Pump ${displayPumpNumber(pump)} ON ${minuteLabel(blockStart)}–${minuteLabel(blockEnd)}`;
        if (isToday && blockStart <= nowMinute && nowMinute < blockEnd && !automation.paused) {
          block.classList.add('current');
        }
        track.append(block);
        blockCount += 1;
      }
    }
    if (isToday) {
      const marker = document.createElement('div');
      marker.className = 'now-marker';
      marker.style.left = `${(nowMinute / 1440) * 100}%`;
      marker.title = `Now ${minuteLabel(nowMinute)}`;
      track.append(marker);
    }
    row.append(rowLabel, track);
    elements.timelineRows.append(row);
  }
  elements.visualizationEmpty.hidden = blockCount > 0;
}

function logFilterParameters({ includePaging = true } = {}) {
  const params = new URLSearchParams();
  if (elements.logPump.value) params.set('pump', elements.logPump.value);
  if (elements.logFrom.value) {
    params.set('from', new Date(elements.logFrom.value).toISOString());
  }
  if (elements.logTo.value) {
    params.set('to', new Date(elements.logTo.value).toISOString());
  }
  if (includePaging) {
    params.set('limit', String(logLimit));
    params.set('offset', String(logOffset));
  }
  return params;
}

function formatLogDuration(seconds) {
  if (seconds === null || seconds === undefined) return 'Still ON';
  const total = Math.max(0, Math.round(seconds));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const remainder = total % 60;
  if (hours) return `${hours}h ${minutes}m ${remainder}s`;
  if (minutes) return `${minutes}m ${remainder}s`;
  return `${remainder}s`;
}

function appendLogCell(row, text, className = '') {
  const cell = document.createElement('td');
  cell.textContent = text;
  if (className) cell.className = className;
  row.append(cell);
}

function renderPumpLog(data) {
  elements.logTableBody.replaceChildren();
  if (!data.rows.length) {
    const row = document.createElement('tr');
    const cell = document.createElement('td');
    cell.colSpan = 6;
    cell.className = 'empty-cell';
    cell.textContent = 'No confirmed pump intervals match these filters.';
    row.append(cell);
    elements.logTableBody.append(row);
  } else {
    for (const interval of data.rows) {
      const row = document.createElement('tr');
      const pumpCell = document.createElement('td');
      const pump = document.createElement('span');
      pump.className = `log-pump ${pumpColors[interval.pump_number - 1]}`;
      pump.textContent = `Pump ${displayPumpNumber(interval.pump_number)}`;
      pumpCell.append(pump);
      row.append(pumpCell);
      appendLogCell(row, new Date(interval.started_at_utc).toLocaleString());
      appendLogCell(
        row,
        interval.ended_at_utc
          ? new Date(interval.ended_at_utc).toLocaleString()
          : 'Currently ON',
        interval.ended_at_utc ? '' : 'open-interval',
      );
      const liveDuration = interval.duration_seconds ?? (
        (Date.now() - new Date(interval.started_at_utc).getTime()) / 1000
      );
      appendLogCell(
        row,
        formatLogDuration(interval.ended_at_utc ? interval.duration_seconds : liveDuration),
        interval.ended_at_utc ? '' : 'open-interval',
      );
      appendLogCell(row, interval.control_mode);
      appendLogCell(row, interval.end_reason || '—');
      elements.logTableBody.append(row);
    }
  }
  elements.logResultCount.textContent = `${data.total} interval${data.total === 1 ? '' : 's'}`;
  const first = data.total ? data.offset + 1 : 0;
  const last = Math.min(data.total, data.offset + data.rows.length);
  elements.logPage.textContent = data.total
    ? `${first}–${last} of ${data.total}`
    : 'No results';
  elements.logPrevious.disabled = data.offset <= 0;
  elements.logNext.disabled = data.offset + data.rows.length >= data.total;
}

async function loadPumpLog() {
  try {
    const params = logFilterParameters();
    const data = await api(`/api/pump-log?${params}`);
    renderPumpLog(data);
    elements.exportLogCsv.href = `/api/pump-log.csv?${logFilterParameters({ includePaging: false })}`;
  } catch (error) {
    showError(error.message);
  }
}

function render(status) {
  latestStatus = status;
  const online = status.connected && status.ready;
  const automation = status.automations || {
    paused: true,
    mask: 0,
    active_ids: [],
    rules: [],
    next_start: null,
    local_time: null,
    timezone: null,
    last_error: null,
  };
  elements.connection.textContent = online ? 'Arduino connected' : 'Arduino offline';
  elements.connection.className = `status-pill ${online ? 'online' : 'offline'}`;
  elements.controllerState.textContent = online
    ? (status.emergency_latched
      ? 'EMERGENCY LATCHED'
      : (status.lease_active ? 'ARMED' : 'READY'))
    : 'OFFLINE';
  elements.mode.textContent = String(status.mode || 'idle').toUpperCase();
  elements.arbitrationSource.textContent = String(
    status.arbitration_source || status.mode || 'idle',
  ).toUpperCase();
  const recorder = status.recorder || {};
  elements.recorderState.textContent = recorder.connected
    ? `CONNECTED · MASK ${recorder.demand_mask || 0}`
    : (recorder.session_active ? 'SESSION STALE · FAIL-CLOSED' : 'DISCONNECTED');
  elements.emergencyState.textContent = status.emergency_latched ? 'LATCHED' : 'CLEAR';
  elements.lastEvent.textContent = status.last_event || '—';
  elements.automationState.textContent = automation.last_error
    ? 'ERROR'
    : (automation.paused
      ? 'PAUSED'
      : (automation.active_ids?.length ? 'RUNNING' : 'ENABLED'));
  elements.toggleAutomations.textContent = automation.paused
    ? 'Resume automations'
    : 'Pause automations';
  elements.localClock.textContent = automation.local_time
    ? `${new Date(automation.local_time).toLocaleString()} ${automation.timezone || ''}`
    : 'Local time —';
  elements.nextAutomation.textContent = automation.next_start
    ? `Next ${formatDateTime(automation.next_start)}`
    : 'Next —';
  renderAutomationList(automation);
  renderScheduleVisualization(automation, status);
  const logStatus = status.pump_log || {
    healthy: false,
    total_intervals: 0,
    open_intervals: 0,
    last_error: 'Pump log unavailable',
  };
  elements.logHealth.textContent = logStatus.healthy
    ? 'Database healthy'
    : 'Database error';
  elements.logHealth.className = `status-pill ${logStatus.healthy ? 'online' : 'offline'}`;
  elements.logOpenCount.textContent = `${logStatus.open_intervals} currently open`;

  const scheduleRunning = status.mode === 'schedule' && status.schedule;
  elements.pumpButtons.forEach((button) => {
    const pump = Number(button.dataset.pump);
    const active = Boolean(status.acknowledged_mask & (1 << (pump - 1)));
    button.classList.toggle('active', active);
    button.setAttribute('aria-pressed', String(active));
    button.querySelector('.pump-state').textContent = active ? 'ON' : 'OFF';
    button.disabled = !online || !status.lease_active || scheduleRunning || requestPending;
  });

  elements.startSchedule.disabled = !online || !status.lease_active || scheduleRunning || requestPending;
  elements.stopSchedule.disabled = !scheduleRunning || requestPending;
  for (const input of elements.scheduleForm.querySelectorAll('input')) {
    input.disabled = scheduleRunning || requestPending;
  }

  elements.scheduleProgress.hidden = !scheduleRunning;
  if (scheduleRunning) {
    const schedule = status.schedule;
    elements.activePump.textContent = `Pump ${displayPumpNumber(schedule.active_pump)}`;
    elements.elapsed.textContent = formatDuration(schedule.elapsed_seconds);
    elements.remaining.textContent = formatDuration(schedule.remaining_seconds);
    const percent = schedule.total_seconds > 0
      ? (schedule.elapsed_seconds / schedule.total_seconds) * 100
      : 0;
    elements.progressFill.style.width = `${Math.min(100, Math.max(0, percent))}%`;
  } else {
    elements.progressFill.style.width = '0%';
  }

  if (status.last_error || automation.last_error || logStatus.last_error) {
    showError(status.last_error || automation.last_error || logStatus.last_error);
  }
}

async function mutate(path, body = {}) {
  requestPending = true;
  let succeeded = false;
  if (latestStatus) render(latestStatus);
  try {
    const status = await api(path, body);
    showError('');
    render(status);
    succeeded = true;
  } catch (error) {
    showError(error.message);
  } finally {
    requestPending = false;
    if (latestStatus) render(latestStatus);
  }
  return succeeded;
}

async function renewLease() {
  try {
    const status = await api('/api/lease', {});
    render(status);
  } catch (error) {
    showError(error.message);
    render({
      connected: false,
      ready: false,
      lease_active: false,
      mode: 'idle',
      acknowledged_mask: 0,
      last_event: null,
      last_error: null,
      schedule: null,
      automations: {
        paused: true,
        mask: 0,
        active_ids: [],
        rules: [],
        next_start: null,
        local_time: null,
        timezone: null,
        last_error: null,
      },
      pump_log: {
        healthy: false,
        total_intervals: 0,
        open_intervals: 0,
        last_error: 'Pump log unavailable',
      },
    });
  }
}

async function refreshStatus() {
  try {
    render(await api('/api/status'));
  } catch (error) {
    showError(error.message);
  }
}

function selectTab(name) {
  activeTab = name;
  elements.controlsView.hidden = name !== 'controls';
  elements.visualizationView.hidden = name !== 'visualization';
  elements.logView.hidden = name !== 'log';
  for (const tab of elements.tabs) {
    const active = tab.dataset.tab === name;
    tab.classList.toggle('active', active);
    tab.setAttribute('aria-selected', String(active));
  }
  if (name === 'visualization' && latestStatus) {
    renderScheduleVisualization(latestStatus.automations, latestStatus);
  }
  if (name === 'log') loadPumpLog();
}

function setVisualizationDate(date) {
  visualizationDateKey = localDateKey(date);
  elements.visualizationDate.value = visualizationDateKey;
  if (latestStatus) {
    renderScheduleVisualization(latestStatus.automations, latestStatus);
  }
}

for (const tab of elements.tabs) {
  tab.addEventListener('click', () => selectTab(tab.dataset.tab));
}

elements.visualizationDate.addEventListener('change', () => {
  if (elements.visualizationDate.value) {
    visualizationDateKey = elements.visualizationDate.value;
    if (latestStatus) {
      renderScheduleVisualization(latestStatus.automations, latestStatus);
    }
  }
});

elements.previousDay.addEventListener('click', () => {
  const date = dateFromKey(visualizationDateKey);
  date.setDate(date.getDate() - 1);
  setVisualizationDate(date);
});

elements.nextDay.addEventListener('click', () => {
  const date = dateFromKey(visualizationDateKey);
  date.setDate(date.getDate() + 1);
  setVisualizationDate(date);
});

elements.today.addEventListener('click', () => {
  const now = latestStatus?.automations?.local_time
    ? new Date(latestStatus.automations.local_time)
    : new Date();
  setVisualizationDate(now);
});

elements.logFilterForm.addEventListener('submit', (event) => {
  event.preventDefault();
  logOffset = 0;
  loadPumpLog();
});

elements.clearLogFilters.addEventListener('click', () => {
  elements.logFilterForm.reset();
  logOffset = 0;
  loadPumpLog();
});

elements.logPrevious.addEventListener('click', () => {
  logOffset = Math.max(0, logOffset - logLimit);
  loadPumpLog();
});

elements.logNext.addEventListener('click', () => {
  logOffset += logLimit;
  loadPumpLog();
});

elements.toggleAutomations.addEventListener('click', () => {
  const paused = latestStatus?.automations?.paused ?? true;
  mutate(paused ? '/api/automations/resume' : '/api/automations/pause');
});

elements.cancelAutomation.addEventListener('click', resetAutomationForm);

elements.automationForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  const pumps = selectedValues('automation-pump');
  const weekdays = selectedValues('weekday');
  if (!pumps.length) {
    showError('Select at least one pump for the automation.');
    return;
  }
  if (!weekdays.length) {
    showError('Select at least one weekday.');
    return;
  }
  const mask = pumps.reduce((combined, pump) => combined | (1 << (pump - 1)), 0);
  const payload = {
    name: elements.automationName.value.trim(),
    enabled: elements.automationEnabled.checked,
    mask,
    weekdays,
    start_time: elements.automationStart.value,
    end_time: elements.automationEnd.value,
    on_minutes: Number(elements.automationOn.value),
    block_minutes: Number(elements.automationBlock.value),
  };
  if (elements.automationId.value) {
    payload.id = elements.automationId.value;
  }
  if (await mutate('/api/automations/save', payload)) {
    resetAutomationForm();
  }
});

elements.pumpButtons.forEach((button) => {
  button.addEventListener('click', () => {
    const pump = Number(button.dataset.pump);
    const bit = 1 << (pump - 1);
    const currentMask = latestStatus?.requested_mask || 0;
    mutate('/api/manual', { mask: currentMask ^ bit });
  });
});

elements.emergencyStop.addEventListener('click', () => {
  mutate('/api/emergency-stop');
});

elements.stopSchedule.addEventListener('click', () => {
  mutate('/api/schedule/stop');
});

elements.scheduleForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  const rawOrder = elements.order.value.trim();
  if (!/^[1-4](\s*,\s*[1-4])*$/.test(rawOrder)) {
    showError('Pump order must be comma-separated pump numbers from 1 through 4.');
    return;
  }
  const order = rawOrder.split(',').map((value) => Number(value.trim()));
  const burstSeconds = Number(elements.burstSeconds.value);
  const totalSeconds = Number(elements.totalMinutes.value) * 60;
  await mutate('/api/schedule/start', {
    order,
    burst_seconds: burstSeconds,
    total_seconds: totalSeconds,
  });
});

renderTimelineHours();
await renewLease();
setInterval(renewLease, 750);
setInterval(refreshStatus, 400);
setInterval(() => {
  if (activeTab === 'log') loadPumpLog();
}, 5000);
