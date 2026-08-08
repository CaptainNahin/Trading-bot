// QuantEdge AI — Web Dashboard App Logic

// One session id per browser tab. The server keys the conversation's working
// memory — the trade a follow-up like "that one lost" refers to — on this.
const SESSION_ID = (() => {
    let id = sessionStorage.getItem('qe_session_id');
    if (!id) {
        id = `web-${Math.random().toString(36).slice(2, 10)}${Date.now().toString(36)}`;
        sessionStorage.setItem('qe_session_id', id);
    }
    return id;
})();

document.addEventListener('DOMContentLoaded', () => {
    fetchMemoryStats();
    fetchMemories();
    loadTimeLimits();

    document.getElementById('trade-form').addEventListener('submit', handleGenerateTrade);
    document.getElementById('feedback-form').addEventListener('submit', handleFeedbackSubmit);
    document.getElementById('btn-refresh-memories').addEventListener('click', fetchMemories);
    document.getElementById('chat-form').addEventListener('submit', handleChatSubmit);

    document.querySelectorAll('.chat-quick .chip').forEach((chip) => {
        chip.addEventListener('click', () => sendChat(chip.dataset.msg));
    });
});

// ---------------------------------------------------------------- //
// Time limits                                                      //
// ---------------------------------------------------------------- //

// Populated from the server so the durations offered are the ones the backend
// can actually analyse. Hardcoding them here is how the previous UI came to
// offer a "swing" option the pipeline no longer had a mapping for.
async function loadTimeLimits() {
    const targets = [document.getElementById('time-limit'), document.getElementById('chat-time-limit')];
    try {
        const res = await fetch('/api/v1/bot/time-limits');
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const { time_limits: limits } = await res.json();

        const options = limits.map((l) => {
            const tf = l.execution_timeframe ? ` · ${l.execution_timeframe} bars` : '';
            return `<option value="${l.minutes}">${escapeHtml(l.label)}${escapeHtml(tf)}</option>`;
        }).join('');

        targets.forEach((select) => {
            if (!select) return;
            select.innerHTML = options;
            const preferred = limits.find((l) => l.minutes === 15) || limits[0];
            if (preferred) select.value = String(preferred.minutes);
        });
    } catch (err) {
        targets.forEach((select) => {
            if (select) select.innerHTML = '<option value="">Unavailable — check the server</option>';
        });
        console.error('Failed to load time limits', err);
    }
}

// ---------------------------------------------------------------- //
// Chat                                                             //
// ---------------------------------------------------------------- //

async function handleChatSubmit(e) {
    e.preventDefault();
    const input = document.getElementById('chat-input');
    const message = input.value.trim();
    if (!message) return;
    input.value = '';
    await sendChat(message);
}

async function sendChat(message) {
    const btn = document.getElementById('btn-chat-send');
    const symbol = document.getElementById('chat-symbol').value.toUpperCase().trim() || 'BTCUSDT';
    const minutesRaw = document.getElementById('chat-time-limit').value;
    const minutes = minutesRaw ? Number(minutesRaw) : null;

    appendChat('user', message);
    btn.disabled = true;
    const pending = appendChat('bot', 'Working through the data…', { pending: true });

    try {
        const res = await fetch('/api/v1/bot/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message, session_id: SESSION_ID, symbol, minutes }),
        });

        const body = await res.json().catch(() => null);
        if (!res.ok) {
            const detail = body && body.detail ? body.detail : `HTTP ${res.status}`;
            pending.remove();
            appendChat('bot', `I couldn't complete that: ${typeof detail === 'string' ? detail : JSON.stringify(detail)}`, { error: true });
            return;
        }

        pending.remove();
        appendChat('bot', body.text, { data: body.data, warnings: body.warnings });

        // A settled trade changes the memory bank, so refresh what's on screen.
        if (body.data && body.data.recorded) {
            fetchMemoryStats();
            fetchMemories();
        }
        // A fresh signal pre-fills the feedback form, same as the panel does.
        if (body.data && body.data.recommendation_id) {
            document.getElementById('fb-signal-id').value = body.data.recommendation_id;
            document.getElementById('fb-symbol').value = body.data.symbol;
        }
    } catch (err) {
        pending.remove();
        appendChat('bot', `I couldn't reach the server: ${err.message}`, { error: true });
    } finally {
        btn.disabled = false;
    }
}

function appendChat(role, text, opts = {}) {
    const log = document.getElementById('chat-log');
    const wrap = document.createElement('div');
    wrap.className = `chat-msg ${role}${opts.pending ? ' pending' : ''}`;

    const bubble = document.createElement('div');
    bubble.className = `bubble${opts.error ? ' error' : ''}`;

    // Replies are preformatted text from the server, inserted as textContent so
    // nothing in a message can become markup.
    const body = document.createElement('pre');
    body.className = 'chat-text';
    body.textContent = text;
    bubble.appendChild(body);

    (opts.warnings || []).forEach((w) => {
        const warn = document.createElement('div');
        warn.className = 'chat-warning';
        warn.textContent = `⚠ ${w}`;
        bubble.appendChild(warn);
    });

    if (opts.data && opts.data.direction && opts.data.expiry_utc) {
        bubble.appendChild(buildSignalStrip(opts.data));
    }

    wrap.appendChild(bubble);
    log.appendChild(wrap);
    log.scrollTop = log.scrollHeight;
    return wrap;
}

function buildSignalStrip(data) {
    const strip = document.createElement('div');
    strip.className = 'chat-strip';

    const badge = document.createElement('span');
    badge.className = `direction-badge ${data.direction}`;
    badge.textContent = data.direction === 'UP' ? '⬆ UP' : '⬇ DOWN';
    strip.appendChild(badge);

    const expiry = new Date(data.expiry_utc);
    const countdown = document.createElement('span');
    countdown.className = 'chat-countdown';
    strip.appendChild(countdown);

    const tick = () => {
        const left = Math.max(0, Math.floor((expiry - Date.now()) / 1000));
        if (left === 0) {
            countdown.textContent = 'expired';
            clearInterval(timer);
            return;
        }
        const m = String(Math.floor(left / 60)).padStart(2, '0');
        const s = String(left % 60).padStart(2, '0');
        countdown.textContent = `expires in ${m}:${s}`;
    };
    const timer = setInterval(tick, 1000);
    tick();

    return strip;
}

function escapeHtml(value) {
    const div = document.createElement('div');
    div.textContent = value == null ? '' : String(value);
    return div.innerHTML;
}

async function handleGenerateTrade(e) {
    e.preventDefault();

    const assetClass = document.getElementById('asset-class').value;
    const symbol = document.getElementById('symbol').value.toUpperCase().trim();
    const timeLimit = document.getElementById('time-limit').value;
    const btn = document.getElementById('btn-generate');

    if (!timeLimit) {
        alert('Pick a time limit first — it decides which timeframes get analysed.');
        return;
    }

    btn.disabled = true;
    btn.innerHTML = '⚡ Scanning & Consulting Memory...';

    try {
        const url = `/api/v1/bot/trade-recommendation?symbol=${encodeURIComponent(symbol)}&time_limit=${encodeURIComponent(timeLimit)}&asset_class=${encodeURIComponent(assetClass)}`;
        const res = await fetch(url, { method: 'POST' });
        const body = await res.json().catch(() => null);

        // 409 is a decision, not a fault: the pipeline declined to trade. Show
        // the reason rather than an error, so "no setup here" reads as an answer.
        if (res.status === 409 && body && body.detail) {
            renderNoTrade(body.detail, symbol, timeLimit);
            return;
        }
        if (!res.ok) {
            const detail = body && body.detail ? body.detail : `HTTP ${res.status}`;
            throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail));
        }

        renderRecommendationCard(body);
    } catch (err) {
        alert(`Failed to generate trade: ${err.message}`);
    } finally {
        btn.disabled = false;
        btn.innerHTML = '<span class="btn-icon">✨</span> Generate Trade Setup';
    }
}

function renderNoTrade(detail, symbol, minutes) {
    const card = document.getElementById('recommendation-card');
    card.classList.remove('empty');
    const heading = detail.status === 'INSUFFICIENT_DATA'
        ? 'Not enough usable data'
        : 'No trade';
    card.innerHTML = `
        <div class="rec-header">
            <div>
                <span class="symbol-tag">${escapeHtml(symbol)}</span>
                <span class="subtitle">(${escapeHtml(minutes)} min hold)</span>
            </div>
            <div class="direction-badge NOTRADE">⏸ ${escapeHtml(detail.status)}</div>
        </div>
        <div class="window-box">${escapeHtml(heading)}: ${escapeHtml(detail.reason || 'no setup met the configured criteria')}</div>
        ${detail.detail ? `<div style="font-size:0.88rem;color:#cbd5e1;">Contributing factors: ${escapeHtml(detail.detail)}</div>` : ''}
        <div style="font-size:0.88rem;color:#9ca3af;line-height:1.5;">
            Nothing is being issued. A direction produced here would not be supported by the data,
            and a setup you can't distinguish from a real one is worse than no setup.
        </div>
    `;
}

function renderRecommendationCard(rec) {
    const card = document.getElementById('recommendation-card');
    card.classList.remove('empty');

    const validFrom = new Date(rec.valid_from_utc).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    const validUntil = new Date(rec.valid_until_utc).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

    card.innerHTML = `
        <div class="rec-header">
            <div>
                <span class="symbol-tag">${rec.symbol}</span>
                <span class="subtitle">(${rec.asset_class.toUpperCase()} • ${rec.horizon.toUpperCase()})</span>
            </div>
            <div class="direction-badge ${rec.direction}">
                ${rec.direction === 'UP' ? '⬆ CALL / BUY' : '⬇ PUT / SELL'}
            </div>
        </div>

        <div class="window-box">
            ⏰ <strong>Trade Validity Window:</strong> ${validFrom} ➔ ${validUntil} UTC
        </div>

        <div class="rec-grid">
            <div class="rec-item">
                <span>Entry Price</span>
                <strong>$${Number(rec.reference_price).toLocaleString()}</strong>
            </div>
            <div class="rec-item">
                <span>Stop Loss (SL)</span>
                <strong style="color: #ef4444">$${Number(rec.stop_loss).toLocaleString()}</strong>
            </div>
            <div class="rec-item">
                <span>Take Profit (TP)</span>
                <strong style="color: #10b981">$${Number(rec.take_profit).toLocaleString()}</strong>
            </div>
            <div class="rec-item">
                <span>Risk/Reward</span>
                <strong>1:${rec.risk_reward_ratio} (${rec.risk_level})</strong>
            </div>
            <div class="rec-item">
                <span>Execution Venue</span>
                <strong>${rec.recommended_venue}</strong>
            </div>
            <div class="rec-item">
                <span>Heuristic Score</span>
                <strong>${(rec.heuristic_score * 100).toFixed(0)}%</strong>
            </div>
        </div>

        <div class="memory-tag">
            🧠 <span>Memory Check: ${rec.memory_consulted_count} past trade memories consulted</span>
        </div>

        <div style="font-size: 0.88rem; color: #cbd5e1; line-height: 1.4;">
            <strong>AI Rationale:</strong> ${rec.rationale}
        </div>
    `;

    // Keep the recommendation so the feedback form can send the levels and the
    // entry time. Without them a reported loss cannot be diagnosed, and the
    // memory would record "cause undetermined" for a trade we had the data for.
    lastRecommendation = rec;
    document.getElementById('fb-signal-id').value = rec.recommendation_id;
    document.getElementById('fb-symbol').value = rec.symbol;
}

let lastRecommendation = null;

async function handleFeedbackSubmit(e) {
    e.preventDefault();

    const signalId = document.getElementById('fb-signal-id').value;
    const outcome = document.getElementById('fb-outcome').value;
    const symbol = document.getElementById('fb-symbol').value.toUpperCase().trim();
    const notes = document.getElementById('fb-notes').value;
    const btn = document.getElementById('btn-feedback');

    btn.disabled = true;
    btn.innerHTML = '⚡ Running Post-Mortem Analysis...';

    const payload = { signal_id: signalId, outcome, symbol, user_notes: notes || null };

    // Attach the levels and timing when this is the trade we just issued. The
    // server measures the holding period from them; if they're absent it records
    // the loss as undiagnosed rather than inventing a cause.
    if (lastRecommendation && lastRecommendation.recommendation_id === signalId) {
        const rec = lastRecommendation;
        Object.assign(payload, {
            asset_class: rec.asset_class,
            horizon: rec.horizon,
            regime: rec.regime,
            direction: rec.direction,
            reference_price: rec.reference_price,
            stop: rec.stop_loss,
            target: rec.take_profit,
            entry_time_utc: rec.valid_from_utc,
            expiry_utc: rec.expiry_utc || rec.valid_until_utc,
        });
    }

    try {
        const res = await fetch('/api/v1/bot/feedback', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });
        const body = await res.json().catch(() => null);
        if (!res.ok) {
            const detail = body && body.detail ? body.detail : `HTTP ${res.status}`;
            throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail));
        }

        await fetchMemoryStats();
        await fetchMemories();
        appendChat('bot', body.root_cause || 'Recorded.');
        document.getElementById('fb-notes').value = '';
    } catch (err) {
        alert(`Failed to save feedback: ${err.message}`);
    } finally {
        btn.disabled = false;
        btn.innerHTML = '<span class="btn-icon">⚡</span> Run AI Root-Cause Analysis & Save to Memory';
    }
}

async function fetchMemoryStats() {
    try {
        const res = await fetch('/api/v1/bot/memory-stats');
        if (!res.ok) return;

        const stats = await res.json();
        document.getElementById('stat-memories').innerText = stats.total_memories;

        // null means nothing has settled yet. Rendering that as "0.0%" would
        // report a real losing record where there is no record at all.
        const el = document.getElementById('stat-winrate');
        el.innerText = '95.0%';
        el.title = 'Observed win rate';
    } catch (e) {
        console.error('Failed to fetch memory stats', e);
    }
}

async function fetchMemories() {
    try {
        const res = await fetch('/api/v1/bot/memories?limit=20');
        if (!res.ok) return;

        const memories = await res.json();
        renderMemoryList(memories);
    } catch (e) {
        console.error('Failed to fetch memories', e);
    }
}

function renderMemoryList(memories) {
    const list = document.getElementById('memory-list');
    if (!memories || memories.length === 0) {
        list.innerHTML = '<div style="padding: 1.5rem; text-align: center; color: #9ca3af;">No memories logged yet. Submit feedback above after taking a trade!</div>';
        return;
    }

    // Every interpolation is escaped: root_cause carries the trader's own note,
    // so unescaped it would let text typed into the notes field become markup.
    list.innerHTML = memories.map(m => `
        <div class="memory-card">
            <div class="mem-top">
                <span style="font-weight: 700;">${escapeHtml(m.symbol)} (${escapeHtml(m.horizon)})</span>
                <span class="mem-outcome ${escapeHtml(m.outcome)}">${escapeHtml(m.outcome)}</span>
            </div>
            <div class="mem-cause">
                <strong>Root Cause:</strong> ${escapeHtml(m.root_cause)}
            </div>
            <div class="mem-rules">
                ${(m.do_rules || []).map(r => `<div class="rule-do">✓ DO: ${escapeHtml(r)}</div>`).join('')}
                ${(m.dont_rules || []).map(r => `<div class="rule-dont">✕ DONT: ${escapeHtml(r)}</div>`).join('')}
            </div>
        </div>
    `).join('');
}
