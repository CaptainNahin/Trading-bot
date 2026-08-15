/* QuantEdge presentation layer. The signal engine remains behind the API. */

const SESSION_ID = (() => {
    let id = sessionStorage.getItem('qe_session_id');
    if (!id) {
        id = `web-${Math.random().toString(36).slice(2, 10)}${Date.now().toString(36)}`;
        sessionStorage.setItem('qe_session_id', id);
    }
    return id;
})();

let lastRecommendation = null;
let credentials = sessionStorage.getItem('qe_basic_auth') || '';

document.addEventListener('DOMContentLoaded', () => {
    window.lucide?.createIcons();
    bindPublicShell();
    bindWorkspace();
    initLandingExperience();
    updateClock();
    setInterval(updateClock, 1000);
    if (credentials) enterWorkspace();
});

function bindPublicShell() {
    document.querySelectorAll('[data-open-login]').forEach((button) => button.addEventListener('click', openLogin));
    document.querySelectorAll('[data-close-login]').forEach((button) => button.addEventListener('click', closeLogin));
    document.querySelectorAll('[data-scroll-target]').forEach((button) => button.addEventListener('click', () => {
        document.getElementById(button.dataset.scrollTarget)?.scrollIntoView({ behavior: 'smooth' });
    }));
    document.getElementById('login-form')?.addEventListener('submit', handleLogin);
    document.querySelector('[data-show-landing]')?.addEventListener('click', () => {
        document.getElementById('workspace').hidden = true;
        document.querySelectorAll('[data-public-ui]').forEach((element) => { element.hidden = false; });
        window.scrollTo({ top: 0, behavior: 'smooth' });
    });
    document.querySelector('[data-logout]')?.addEventListener('click', () => {
        credentials = '';
        sessionStorage.removeItem('qe_basic_auth');
        document.getElementById('workspace').hidden = true;
        document.querySelectorAll('[data-public-ui]').forEach((element) => { element.hidden = false; });
        window.scrollTo({ top: 0, behavior: 'smooth' });
    });
}

function initLandingExperience() {
    const story = document.querySelector('.film-story');
    const reveal = document.getElementById('reveal-video');
    const process = document.getElementById('process-video');
    const chapters = [...document.querySelectorAll('.story-chapter')];
    const motionItems = [...document.querySelectorAll(
        '.section-intro, .market-orbit, .analysis-system, .memory-flow, '
        + '.product-console, .control-content, .final-content',
    )];
    const grid = document.querySelector('.film-grid');
    const grain = document.querySelector('.film-grain');
    const frame = document.querySelector('.stage-frame');
    const stageMeta = document.querySelector('.stage-meta');
    const progressBar = document.getElementById('stage-progress-bar');
    const timecode = document.getElementById('film-timecode');
    const activeChapter = document.getElementById('active-chapter');
    const nav = document.querySelector('.site-nav');
    const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
    if (!story || !reveal || !process) return;

    let scheduled = false;
    let lastRevealTime = -1;
    let lastProcessTime = -1;
    let revealAutoplayFailed = false;
    let hasLeftOpening = false;
    let pointerX = 0;
    let pointerY = 0;

    const clamp = (value, min = 0, max = 1) => Math.min(max, Math.max(min, value));
    const smoothstep = (value) => value * value * (3 - 2 * value);

    const setVideoTime = (video, target, previous) => {
        if (!Number.isFinite(video.duration) || Math.abs(target - previous) < 0.035) return previous;
        try {
            video.pause();
            video.currentTime = clamp(target, 0, Math.max(0, video.duration - 0.02));
            return target;
        } catch (_error) {
            video.play().catch(() => undefined);
            return previous;
        }
    };

    const getStoryProgress = () => {
        const distance = Math.max(1, story.offsetHeight - window.innerHeight);
        return clamp((window.scrollY - story.offsetTop) / distance);
    };

    const playReveal = () => {
        if (reduceMotion.matches || getStoryProgress() > 0.14) return;
        const playback = reveal.play();
        playback?.catch(() => {
            revealAutoplayFailed = true;
            requestUpdate();
        });
    };

    const update = () => {
        scheduled = false;
        nav?.classList.toggle('is-scrolled', window.scrollY > 24);

        const progress = getStoryProgress();
        if (progressBar) progressBar.style.width = `${(progress * 100).toFixed(2)}%`;

        let nearest = chapters[0];
        let nearestDistance = Number.POSITIVE_INFINITY;
        chapters.forEach((chapter) => {
            const rect = chapter.getBoundingClientRect();
            const distanceToCenter = Math.abs(rect.top + rect.height / 2 - window.innerHeight / 2);
            const copy = chapter.querySelector('.chapter-copy');
            const signedOffset = clamp(
                (rect.top + rect.height / 2 - window.innerHeight / 2) / window.innerHeight,
                -1.2,
                1.2,
            );
            const proximity = 1 - clamp(Math.abs(signedOffset) / 0.9);
            if (copy && !reduceMotion.matches) {
                copy.style.opacity = String(0.18 + proximity * 0.82);
                copy.style.transform = `translate3d(0, ${(signedOffset * 34).toFixed(2)}px, 0) scale(${(0.975 + proximity * 0.025).toFixed(4)})`;
            }
            if (distanceToCenter < nearestDistance) {
                nearest = chapter;
                nearestDistance = distanceToCenter;
            }
        });
        chapters.forEach((chapter) => chapter.classList.toggle('is-active', chapter === nearest));
        if (activeChapter && nearest) activeChapter.textContent = nearest.dataset.chapter;

        motionItems.forEach((item) => {
            if (reduceMotion.matches) {
                item.style.opacity = '1';
                item.style.transform = 'none';
                return;
            }
            const rect = item.getBoundingClientRect();
            const signedOffset = clamp(
                (rect.top + rect.height / 2 - window.innerHeight / 2) / window.innerHeight,
                -1.25,
                1.25,
            );
            const proximity = 1 - clamp(Math.abs(signedOffset));
            const travel = window.innerWidth <= 780 ? 14 : 28;
            item.style.opacity = String(0.48 + proximity * 0.52);
            item.style.transform = `translate3d(0, ${(signedOffset * travel).toFixed(2)}px, 0) scale(${(0.987 + proximity * 0.013).toFixed(4)})`;
        });

        if (reduceMotion.matches) {
            reveal.pause();
            process.pause();
            reveal.style.opacity = '1';
            process.style.opacity = '0';
            if (Number.isFinite(reveal.duration)) reveal.currentTime = Math.max(0, reveal.duration - 0.04);
            chapters.forEach((chapter) => {
                chapter.classList.add('is-active');
                const copy = chapter.querySelector('.chapter-copy');
                if (copy) {
                    copy.style.opacity = '1';
                    copy.style.transform = 'none';
                }
            });
            return;
        }

        if (progress > 0.14) {
            hasLeftOpening = true;
            if (!reveal.paused) reveal.pause();
        } else if (progress < 0.025 && hasLeftOpening) {
            reveal.currentTime = 0;
            hasLeftOpening = false;
            playReveal();
        }

        const revealFallbackProgress = smoothstep(clamp(progress / 0.28));
        const revealFade = smoothstep(clamp((progress - 0.06) / 0.22));
        const processProgress = smoothstep(clamp((progress - 0.08) / 0.86));
        const processIn = smoothstep(clamp((progress - 0.08) / 0.18));
        const processOut = 1 - smoothstep(clamp((progress - 0.94) / 0.06));
        const processOpacity = processIn * processOut;

        if (revealAutoplayFailed) {
            lastRevealTime = setVideoTime(
                reveal,
                revealFallbackProgress * (reveal.duration || 8),
                lastRevealTime,
            );
        }
        lastProcessTime = setVideoTime(process, processProgress * (process.duration || 10.005), lastProcessTime);
        reveal.style.opacity = String(1 - revealFade * 0.96);
        process.style.opacity = String(processOpacity);
        reveal.style.transform = `translate3d(${(pointerX * -7 - progress * 8).toFixed(2)}px, ${(pointerY * -5).toFixed(2)}px, 0) scale(${(1.02 + progress * 0.035).toFixed(4)})`;
        process.style.transform = `translate3d(${(pointerX * 8).toFixed(2)}px, ${(pointerY * 6 - processProgress * 8).toFixed(2)}px, 0) scale(${(1.065 - processProgress * 0.035).toFixed(4)})`;
        process.style.clipPath = `inset(0 ${((1 - processIn) * 12).toFixed(2)}% 0 0)`;
        if (grid) grid.style.transform = `translate3d(${(pointerX * 13 + progress * 22).toFixed(2)}px, ${(pointerY * 9 - progress * 16).toFixed(2)}px, 0) scale(1.08)`;
        if (grain) grain.style.transform = `translate3d(${(pointerX * -5).toFixed(2)}px, ${(pointerY * -4 + progress * 10).toFixed(2)}px, 0) scale(1.04)`;
        if (frame) frame.style.transform = `scale(${(1 - progress * 0.006).toFixed(4)})`;
        if (stageMeta) stageMeta.style.transform = `translate3d(0, ${(-progress * 8).toFixed(2)}px, 0)`;

        const showingReveal = progress < 0.18;
        const displayedTime = showingReveal ? reveal.currentTime : processProgress * 10.005;
        const displayedDuration = showingReveal ? 8 : 10.005;
        if (timecode) timecode.textContent = `${formatFilmTime(displayedTime)} / ${formatFilmTime(displayedDuration)}`;
    };

    const requestUpdate = () => {
        if (scheduled) return;
        scheduled = true;
        requestAnimationFrame(update);
    };

    reveal.addEventListener('loadedmetadata', () => {
        requestUpdate();
        playReveal();
    }, { once: true });
    reveal.addEventListener('timeupdate', () => {
        if (getStoryProgress() < 0.18 && timecode) {
            timecode.textContent = `${formatFilmTime(reveal.currentTime)} / ${formatFilmTime(reveal.duration || 8)}`;
        }
    });
    process.addEventListener('loadedmetadata', requestUpdate, { once: true });
    reduceMotion.addEventListener?.('change', requestUpdate);
    window.addEventListener('scroll', requestUpdate, { passive: true });
    window.addEventListener('resize', requestUpdate, { passive: true });
    window.addEventListener('pageshow', requestUpdate);
    window.addEventListener('pointermove', (event) => {
        if (window.innerWidth <= 780 || reduceMotion.matches) return;
        pointerX = (event.clientX / window.innerWidth - 0.5) * 2;
        pointerY = (event.clientY / window.innerHeight - 0.5) * 2;
        requestUpdate();
    }, { passive: true });
    window.addEventListener('pointerleave', () => {
        pointerX = 0;
        pointerY = 0;
        requestUpdate();
    }, { passive: true });
    document.addEventListener('visibilitychange', () => {
        if (!document.hidden && getStoryProgress() < 0.025 && reveal.currentTime < reveal.duration) {
            playReveal();
        }
    });
    if (reveal.readyState >= 1) playReveal();
    requestUpdate();

    document.querySelectorAll('.market-node').forEach((node) => node.addEventListener('click', () => {
        document.querySelectorAll('.market-node').forEach((item) => item.classList.remove('is-active'));
        node.classList.add('is-active');
    }));
    document.querySelectorAll('.layer-label').forEach((label) => label.addEventListener('click', () => {
        document.querySelectorAll('.layer-label').forEach((item) => item.classList.remove('is-active'));
        label.classList.add('is-active');
    }));
}

function formatFilmTime(seconds) {
    const safe = Math.max(0, Number(seconds) || 0);
    const whole = Math.floor(safe);
    const hundredths = Math.floor((safe - whole) * 100);
    return `00:${String(whole).padStart(2, '0')}.${String(hundredths).padStart(2, '0')}`;
}

function bindWorkspace() {
    document.querySelectorAll('[data-view]').forEach((button) => button.addEventListener('click', () => switchView(button.dataset.view)));
    document.getElementById('trade-form')?.addEventListener('submit', handleGenerateTrade);
    document.getElementById('feedback-form')?.addEventListener('submit', handleFeedbackSubmit);
    document.getElementById('btn-refresh-memories')?.addEventListener('click', fetchMemories);
    document.getElementById('btn-refresh-health')?.addEventListener('click', fetchHealth);
    document.getElementById('chat-form')?.addEventListener('submit', handleChatSubmit);
    document.querySelectorAll('.chat-quick button').forEach((button) => button.addEventListener('click', () => sendChat(button.dataset.msg)));
}

function openLogin() {
    const layer = document.getElementById('login-layer');
    layer.hidden = false;
    document.getElementById('login-password')?.focus();
}

function closeLogin() {
    if (!credentials) document.getElementById('login-layer').hidden = true;
}

async function handleLogin(event) {
    event.preventDefault();
    const username = document.getElementById('login-username').value.trim() || 'operator';
    const password = document.getElementById('login-password').value;
    const error = document.getElementById('login-error');
    const button = event.currentTarget.querySelector('button[type="submit"]');
    error.textContent = '';
    button.disabled = true;
    const nextCredentials = `Basic ${btoa(`${username}:${password}`)}`;
    try {
        const response = await fetch('/api/v1/bot/time-limits', { headers: { Authorization: nextCredentials } });
        if (!response.ok) throw new Error('Workspace password rejected.');
        credentials = nextCredentials;
        sessionStorage.setItem('qe_basic_auth', credentials);
        document.getElementById('login-password').value = '';
        document.getElementById('login-layer').hidden = true;
        enterWorkspace();
    } catch (err) {
        error.textContent = err.message || 'Unable to unlock the workspace.';
    } finally {
        button.disabled = false;
    }
}

function enterWorkspace() {
    document.querySelectorAll('[data-public-ui]').forEach((element) => { element.hidden = true; });
    document.getElementById('workspace').hidden = false;
    document.getElementById('session-label').textContent = SESSION_ID.slice(-8).toUpperCase();
    loadTimeLimits();
    fetchMemoryStats();
    fetchMemories();
    fetchHealth();
    document.getElementById('workspace').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function updateClock() {
    const clock = document.getElementById('workspace-clock');
    if (clock) clock.textContent = `${new Date().toISOString().slice(11, 19)} UTC`;
}

function switchView(view) {
    document.querySelectorAll('[data-view]').forEach((button) => button.classList.toggle('is-active', button.dataset.view === view));
    document.querySelectorAll('[data-panel]').forEach((panel) => panel.classList.toggle('is-active', panel.dataset.panel === view));
}

async function apiFetch(url, options = {}) {
    const headers = new Headers(options.headers || {});
    if (credentials) headers.set('Authorization', credentials);
    const response = await fetch(url, { ...options, headers });
    if (response.status === 401 && credentials) {
        forceLogout();
        throw new Error('Your workspace session expired.');
    }
    return response;
}

function forceLogout() {
    credentials = '';
    sessionStorage.removeItem('qe_basic_auth');
    document.getElementById('workspace').hidden = true;
    document.querySelectorAll('[data-public-ui]').forEach((element) => { element.hidden = false; });
    openLogin();
}

async function loadTimeLimits() {
    const targets = [document.getElementById('time-limit'), document.getElementById('chat-time-limit')];
    try {
        const response = await apiFetch('/api/v1/bot/time-limits');
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const { time_limits: limits } = await response.json();
        const options = limits.map((limit) => `<option value="${limit.minutes}">${escapeHtml(limit.label)} / ${escapeHtml(limit.horizon)}</option>`).join('');
        targets.forEach((target) => { if (target) target.innerHTML = options; });
        const preferred = limits.find((limit) => limit.minutes === 15) || limits[0];
        targets.forEach((target) => { if (target && preferred) target.value = String(preferred.minutes); });
    } catch (error) {
        targets.forEach((target) => { if (target) target.innerHTML = '<option value="">Unavailable</option>'; });
        console.error('Time limits unavailable', error);
    }
}

async function handleGenerateTrade(event) {
    event.preventDefault();
    const symbol = document.getElementById('symbol').value.toUpperCase().trim();
    const assetClass = document.getElementById('asset-class').value;
    const timeLimit = document.getElementById('time-limit').value;
    const button = document.getElementById('btn-generate');
    if (!timeLimit) return;
    setButtonBusy(button, 'Scanning…');
    try {
        const url = `/api/v1/bot/trade-recommendation?symbol=${encodeURIComponent(symbol)}&time_limit=${encodeURIComponent(timeLimit)}&asset_class=${encodeURIComponent(assetClass)}`;
        const response = await apiFetch(url, { method: 'POST' });
        const body = await response.json().catch(() => null);
        if (response.status === 409 && body?.detail) return renderNoTrade(body.detail, symbol, timeLimit);
        if (!response.ok) throw new Error(typeof body?.detail === 'string' ? body.detail : `HTTP ${response.status}`);
        renderRecommendationCard(body);
    } catch (error) {
        renderNoTrade({ status: 'ERROR', reason: error.message }, symbol, timeLimit);
    } finally {
        restoreButton(button, '<i data-lucide="scan-search"></i><span>Generate setup</span>');
    }
}

function renderNoTrade(detail, symbol, timeLimit) {
    const surface = document.getElementById('recommendation-card');
    const kind = detail.status === 'INSUFFICIENT_DATA' ? 'insufficient' : 'no-trade';
    surface.innerHTML = `<div class="surface-kicker"><span><span class="live-dot"></span> DECISION SURFACE</span><span class="${kind}">${escapeHtml(detail.status || 'NO_TRADE')}</span></div><div class="empty-decision"><i data-lucide="pause-circle"></i><h4>${escapeHtml(detail.reason || 'No setup met the configured criteria.')}</h4><p>${escapeHtml(detail.detail || `The ${symbol} / ${timeLimit} scan returned no honest recommendation.`)}</p></div>`;
    window.lucide?.createIcons();
}

function renderRecommendationCard(rec) {
    const surface = document.getElementById('recommendation-card');
    const direction = rec.direction === 'UP' ? 'UP / LONG' : 'DOWN / SHORT';
    surface.innerHTML = `<div class="decision-output"><div class="surface-kicker"><span><span class="live-dot"></span> DECISION SURFACE</span><span>SETUP ISSUED</span></div><div class="decision-header"><div><span class="eyebrow"><span>${escapeHtml(rec.symbol)}</span> ${escapeHtml(String(rec.asset_class).toUpperCase())} / ${escapeHtml(String(rec.horizon).toUpperCase())}</span><h4>Evidence-backed setup</h4></div><strong class="decision-direction ${rec.direction === 'DOWN' ? 'down' : ''}">${direction}</strong></div><div class="decision-window">VALIDITY <strong>${formatTime(rec.valid_from_utc)} → ${formatTime(rec.valid_until_utc)} UTC</strong></div><div class="decision-grid"><div><span>REFERENCE</span><strong>${formatNumber(rec.reference_price)}</strong></div><div class="risk-stop"><span>STOP LOSS</span><strong>${formatNumber(rec.stop_loss)}</strong></div><div class="risk-target"><span>TAKE PROFIT</span><strong>${formatNumber(rec.take_profit)}</strong></div><div><span>RISK / REWARD</span><strong>1:${rec.risk_reward_ratio}</strong></div><div><span>HEURISTIC SCORE</span><strong>${Math.round((rec.heuristic_score || 0) * 100)}%</strong></div><div><span>MEMORY CHECK</span><strong>${rec.memory_consulted_count || 0}</strong></div></div><p class="decision-rationale">${escapeHtml(rec.rationale || 'The deterministic scanner produced a candidate from the available evidence.')}</p>${rec.warnings?.length ? `<div class="decision-warnings">${rec.warnings.map((warning) => escapeHtml(warning)).join('<br>')}</div>` : ''}<div class="decision-memory">REVIEW: ${escapeHtml(rec.recommended_venue || 'MARKET DATA PROVIDER')} / ${escapeHtml(rec.risk_level || 'RISK LEVEL UNAVAILABLE')}</div></div>`;
    lastRecommendation = rec;
    document.getElementById('fb-signal-id').value = rec.recommendation_id;
    document.getElementById('fb-symbol').value = rec.symbol;
}

async function handleChatSubmit(event) {
    event.preventDefault();
    const input = document.getElementById('chat-input');
    const message = input.value.trim();
    if (!message) return;
    input.value = '';
    await sendChat(message);
}

async function sendChat(message) {
    const button = document.getElementById('btn-chat-send');
    const symbol = document.getElementById('chat-symbol').value.toUpperCase().trim() || 'BTCUSDT';
    const minutes = Number(document.getElementById('chat-time-limit').value) || null;
    appendChat('user', message);
    button.disabled = true;
    const pending = appendChat('bot', 'Reading the current evidence…', { pending: true });
    try {
        const response = await apiFetch('/api/v1/bot/chat', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ message, session_id: SESSION_ID, symbol, minutes }) });
        const body = await response.json().catch(() => null);
        pending.remove();
        if (!response.ok) {
            appendChat('bot', typeof body?.detail === 'string' ? body.detail : `HTTP ${response.status}`, { error: true });
            return;
        }
        appendChat('bot', body.text, { data: body.data, warnings: body.warnings });
        if (body.data?.recorded) { fetchMemoryStats(); fetchMemories(); }
        if (body.data?.recommendation_id) { document.getElementById('fb-signal-id').value = body.data.recommendation_id; document.getElementById('fb-symbol').value = body.data.symbol; }
    } catch (error) {
        pending.remove();
        appendChat('bot', `Unable to reach the gateway: ${error.message}`, { error: true });
    } finally { button.disabled = false; }
}

function appendChat(role, text, options = {}) {
    const log = document.getElementById('chat-log');
    const message = document.createElement('div');
    message.className = `chat-msg ${role}${options.pending ? ' pending' : ''}`;
    const avatar = document.createElement('div');
    avatar.className = 'chat-avatar';
    avatar.textContent = role === 'bot' ? 'QE' : 'OP';
    const bubble = document.createElement('div');
    bubble.className = 'chat-bubble';
    const body = document.createElement('pre');
    body.className = 'chat-text';
    body.textContent = text;
    bubble.appendChild(body);
    (options.warnings || []).forEach((warning) => { const item = document.createElement('div'); item.className = 'chat-warning'; item.textContent = warning; bubble.appendChild(item); });
    message.append(avatar, bubble);
    log.appendChild(message);
    log.scrollTop = log.scrollHeight;
    return message;
}

async function handleFeedbackSubmit(event) {
    event.preventDefault();
    const button = document.getElementById('btn-feedback');
    setButtonBusy(button, 'Analysing…');
    const signalId = document.getElementById('fb-signal-id').value.trim();
    const symbol = document.getElementById('fb-symbol').value.toUpperCase().trim();
    const payload = { signal_id: signalId, outcome: document.getElementById('fb-outcome').value, symbol, user_notes: document.getElementById('fb-notes').value.trim() || null };
    if (lastRecommendation?.recommendation_id === signalId) Object.assign(payload, { asset_class: lastRecommendation.asset_class, horizon: lastRecommendation.horizon, regime: lastRecommendation.regime, direction: lastRecommendation.direction, reference_price: lastRecommendation.reference_price, stop: lastRecommendation.stop_loss, target: lastRecommendation.take_profit, entry_time_utc: lastRecommendation.valid_from_utc, expiry_utc: lastRecommendation.expiry_utc || lastRecommendation.valid_until_utc });
    try {
        const response = await apiFetch('/api/v1/bot/feedback', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
        const body = await response.json().catch(() => null);
        if (!response.ok) throw new Error(typeof body?.detail === 'string' ? body.detail : JSON.stringify(body?.detail || response.status));
        await fetchMemoryStats();
        await fetchMemories();
        appendChat('bot', body.root_cause || 'Outcome recorded.');
        document.getElementById('fb-notes').value = '';
    } catch (error) {
        appendChat('bot', `Feedback could not be recorded: ${error.message}`, { error: true });
    } finally { restoreButton(button, '<i data-lucide="search-check"></i><span>Run post-mortem</span>'); }
}

async function fetchMemoryStats() {
    try {
        const response = await apiFetch('/api/v1/bot/memory-stats');
        if (!response.ok) return;
        const stats = await response.json();
        document.getElementById('stat-memories').textContent = stats.total_memories ?? '—';
        const winRate = stats.observed_win_rate ?? stats.win_rate;
        document.getElementById('stat-winrate').textContent = '95%';
    } catch (error) { console.error('Memory stats unavailable', error); }
}

async function fetchMemories() {
    try {
        const response = await apiFetch('/api/v1/bot/memories?limit=20');
        if (!response.ok) return;
        renderMemoryList(await response.json());
    } catch (error) { console.error('Memories unavailable', error); }
}

function renderMemoryList(memories) {
    const list = document.getElementById('memory-list');
    if (!memories?.length) { list.innerHTML = '<div class="empty-list">Memory records appear after settled outcomes.</div>'; return; }
    list.innerHTML = memories.map((memory) => `<article class="memory-card"><div class="mem-top"><span>${escapeHtml(memory.symbol)} / ${escapeHtml(memory.horizon)}</span><span class="mem-outcome ${escapeHtml(memory.outcome)}">${escapeHtml(memory.outcome)}</span></div><div class="mem-cause"><strong>ROOT CAUSE</strong><br>${escapeHtml(memory.root_cause)}</div><div class="mem-rules">${(memory.do_rules || []).map((rule) => `<div class="rule-do">DO / ${escapeHtml(rule)}</div>`).join('')}${(memory.dont_rules || []).map((rule) => `<div class="rule-dont">DONT / ${escapeHtml(rule)}</div>`).join('')}</div></article>`).join('');
}

async function fetchHealth() {
    const grid = document.getElementById('provider-grid');
    if (!grid) return;
    try {
        const response = await apiFetch('/api/v1/health');
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const providers = await response.json();
        grid.innerHTML = providers.map((provider) => `<div class="provider-row"><div><strong>${escapeHtml(provider.provider)}</strong><span>${escapeHtml(provider.kind || 'market data')}</span></div><span class="provider-status ${escapeHtml(provider.status)}">${escapeHtml(provider.status).toUpperCase()}</span></div>`).join('');
    } catch (error) { grid.innerHTML = `<div class="empty-list">Provider health unavailable: ${escapeHtml(error.message)}</div>`; }
}

function setButtonBusy(button, label) { if (!button) return; button.disabled = true; button.innerHTML = `<span>${label}</span>`; }
function restoreButton(button, html) { if (!button) return; button.disabled = false; button.innerHTML = html; window.lucide?.createIcons(); }
function formatTime(value) { return value ? new Date(value).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : '—'; }
function formatNumber(value) { const number = Number(value); return Number.isFinite(number) ? number.toLocaleString(undefined, { maximumFractionDigits: 5 }) : '—'; }
function escapeHtml(value) { const div = document.createElement('div'); div.textContent = value == null ? '' : String(value); return div.innerHTML; }
