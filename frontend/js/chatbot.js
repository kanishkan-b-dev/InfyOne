/**
 * Infosys AI — Chatbot Client
 * Design system: Linear (Inspo MCP) — dark, minimal, technical
 * Handles: session management, API calls, markdown rendering,
 *          typing indicator, latency tracking, keyboard shortcuts
 */
(function () {
  'use strict';

  // ── Config ──────────────────────────────────────────────────────────────
  const API_URL         = '/api/chat';
  const SESSION_KEY     = 'infosys_session_id';

  // ── State ────────────────────────────────────────────────────────────────
  let sessionId = localStorage.getItem(SESSION_KEY) || generateSessionId();
  let busy = false;

  function generateSessionId() {
    const id = 'sess_' + Date.now().toString(36) + '_' + Math.random().toString(36).slice(2, 9);
    localStorage.setItem(SESSION_KEY, id);
    return id;
  }

  // ── DOM refs ─────────────────────────────────────────────────────────────
  const feed          = document.getElementById('consoleFeed');
  const input         = document.getElementById('chatInput');
  const sendBtn       = document.getElementById('sendBtn');
  const clearBtn      = document.getElementById('clearSessionBtn');
  const latencyTag    = document.getElementById('latencyTag');
  const typingRow     = document.getElementById('typingIndicator');
  const chips         = document.querySelectorAll('.chip[data-query]');

  // ── Helpers ──────────────────────────────────────────────────────────────
  function now() {
    return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  }

  function scrollBottom() {
    feed.scrollTop = feed.scrollHeight;
  }

  // ── Minimal markdown renderer ────────────────────────────────────────────
  // Escape HTML first, then apply markdown patterns.
  function renderMarkdown(raw) {
    if (!raw) return '';

    let html = raw
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;');

    // Fenced code blocks
    html = html.replace(/```([a-z]*)\n?([\s\S]*?)```/g, (_, lang, code) =>
      `<pre><code class="lang-${lang || 'text'}">${code.trim()}</code></pre>`
    );

    // Inline code (must come after fenced blocks)
    html = html.replace(/`([^`\n]+)`/g, '<code>$1</code>');

    // Bold + italic
    html = html.replace(/\*\*\*(.+?)\*\*\*/g, '<strong><em>$1</em></strong>');
    html = html.replace(/\*\*(.+?)\*\*/g,     '<strong>$1</strong>');
    html = html.replace(/\*(.+?)\*/g,          '<em>$1</em>');

    // Headings (h3, h2 — h1 unlikely in chat)
    html = html.replace(/^### (.+)$/gm, '<h3 style="font-size:13px;font-weight:600;margin:10px 0 4px;color:var(--color-text-primary)">$1</h3>');
    html = html.replace(/^## (.+)$/gm,  '<h3 style="font-size:14px;font-weight:600;margin:12px 0 6px;color:var(--color-text-primary)">$1</h3>');

    // Unordered lists
    html = html.replace(/((?:^[ \t]*[-•*] .+\n?)+)/gm, (block) => {
      const items = block
        .trim()
        .split('\n')
        .filter(Boolean)
        .map(l => l.replace(/^[ \t]*[-•*] /, ''))
        .map(l => `<li>${l}</li>`)
        .join('');
      return `<ul>${items}</ul>`;
    });

    // Paragraphs — double newlines → paragraph breaks
    html = html
      .replace(/\n\n+/g, '\n§\n')
      .split('§')
      .map(s => s.trim())
      .filter(Boolean)
      .map(s => {
        // Don't wrap block-level elements in <p>
        if (/^<(h[1-6]|ul|ol|pre|blockquote)/.test(s)) return s;
        return `<p>${s.replace(/\n/g, '<br>')}</p>`;
      })
      .join('\n');

    return html;
  }

  // ── Append message to feed ───────────────────────────────────────────────
  function appendMsg(role, content, meta = {}) {
    // Move typing indicator to bottom before appending
    if (typingRow) feed.appendChild(typingRow);

    const row = document.createElement('div');
    row.className = `msg-row ${role}`;

    const avatar = document.createElement('div');
    avatar.className = 'msg-avatar';

    if (role === 'ai') {
      // SVG person silhouette — looks human/assistant, not a generic "AI" label
      avatar.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" aria-hidden="true">
        <circle cx="12" cy="8" r="3.5" fill="currentColor" opacity="0.9"/>
        <path d="M4 20c0-4 3.6-7 8-7s8 3 8 7" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" fill="none" opacity="0.9"/>
      </svg>`;
      avatar.setAttribute('aria-label', 'InfyOne');
    } else {
      avatar.textContent = 'U';
      avatar.setAttribute('aria-hidden', 'true');
    }

    const body = document.createElement('div');
    body.className = 'msg-body';

    const metaEl = document.createElement('div');
    metaEl.className = 'msg-meta';
    metaEl.innerHTML = `<span>${role === 'user' ? 'You' : 'InfyOne'}</span><span aria-hidden="true">·</span><span>${now()}</span>`;

    const bubble = document.createElement('div');
    bubble.className = 'msg-bubble';

    if (role === 'user') {
      bubble.textContent = content;
    } else {
      let inner = '';

      // Tool badge (shown when RAG was invoked)
      if (meta.toolUsed) {
        inner += `<div class="tool-badge" aria-label="RAG tool invoked">
          <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/></svg>
          search_infyone_knowledge_base
        </div>`;
      }

      inner += renderMarkdown(content);
      bubble.innerHTML = inner;
    }

    body.appendChild(metaEl);
    body.appendChild(bubble);
    row.appendChild(avatar);
    row.appendChild(body);

    // Insert before typing indicator
    feed.insertBefore(row, typingRow);
    scrollBottom();
    return row;
  }

  function showTyping()  { if (typingRow) { typingRow.classList.add('visible'); scrollBottom(); } }
  function hideTyping()  { if (typingRow) typingRow.classList.remove('visible'); }

  // ── Send message ─────────────────────────────────────────────────────────
  async function send(text) {
    text = (text || input.value).trim();
    if (!text || busy) return;

    input.value = '';
    input.style.height = 'auto';
    busy = true;
    sendBtn.disabled = true;

    appendMsg('user', text);
    showTyping();

    const t0 = performance.now();

    try {
      const res = await fetch(API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text, session_id: sessionId }),
      });

      const elapsed = (performance.now() - t0) / 1000;
      if (latencyTag) latencyTag.textContent = elapsed.toFixed(2) + 's';

      hideTyping();

      if (!res.ok) {
        if (res.status === 401) {
          window.location.href = '/login';
          return;
        }
        const err = await res.json().catch(() => ({}));
        appendMsg('ai', `**Error ${res.status}:** ${err.detail || 'Unexpected error from the knowledge server.'}`);
        return;
      }

      const data = await res.json();
      if (data.session_id) {
        sessionId = data.session_id;
        localStorage.setItem(SESSION_KEY, sessionId);
      }
      const answer = data.answer || 'No response returned.';
      const sources = data.sources || [];
      const toolUsed = (data.answering_node === 'generate_answer' || data.rewritten_query)
        ? 'search_infosys_knowledge_base'
        : null;

      appendMsg('ai', answer, { sources, toolUsed });

    } catch (err) {
      hideTyping();
      appendMsg('ai', `**Network error:** Cannot reach \`${API_URL}\`. Ensure the server is running.`);
      console.error('[Infosys] fetch error:', err);
    } finally {
      busy = false;
      sendBtn.disabled = false;
      input.focus();
    }
  }

  // ── Keyboard shortcuts ───────────────────────────────────────────────────
  if (input) {
    input.addEventListener('keydown', e => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        send();
      } else if (e.key === 'Escape') {
        input.value = '';
        input.style.height = 'auto';
      }
    });

    // Auto-expand textarea
    input.addEventListener('input', () => {
      input.style.height = 'auto';
      input.style.height = Math.min(input.scrollHeight, 120) + 'px';
    });
  }

  // Global "/" key to focus input (Linear-style keyboard shortcut)
  document.addEventListener('keydown', e => {
    if (e.key === '/' && document.activeElement !== input && !e.metaKey && !e.ctrlKey) {
      e.preventDefault();
      input?.focus();
      // Scroll console into view
      document.getElementById('console')?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
  });

  // Send button
  sendBtn?.addEventListener('click', () => send());

  // Clear / reset session
  clearBtn?.addEventListener('click', () => {
    sessionId = generateSessionId();
    if (latencyTag) latencyTag.textContent = '0.00s';

    // Remove all messages except welcome
    const welcome = document.getElementById('welcome-msg');
    Array.from(feed.children).forEach(el => {
      if (el.id !== 'welcome-msg' && el.id !== 'typingIndicator') el.remove();
    });

    // Update welcome message
    if (welcome) {
      const bubble = welcome.querySelector('.msg-bubble');
      if (bubble) bubble.innerHTML =
        'Session reset. New conversation started. How can I help you with <strong>Infosys</strong> today?';
      const metaEl = welcome.querySelector('.msg-meta span:last-child');
      if (metaEl) metaEl.textContent = now();
    }
  });

  // Quick topic chips
  chips.forEach(chip => {
    chip.addEventListener('click', () => {
      const query = chip.getAttribute('data-query');
      if (!query) return;
      // Scroll to console and send
      document.getElementById('console')?.scrollIntoView({ behavior: 'smooth', block: 'center' });
      setTimeout(() => send(query), 300);
    });
  });

  // Defer moving typing row to end of feed on init
  if (typingRow) feed.appendChild(typingRow);

  // ── Authentication Check & User Profile Chip ───────────────────────────
  async function checkAuth() {
    try {
      const res = await fetch('/api/auth/me');
      if (res.ok) {
        const data = await res.json();
        if (data.authenticated && data.username) {
          const userChip = document.getElementById('userChip');
          const userNameDisplay = document.getElementById('userNameDisplay');
          if (userChip && userNameDisplay) {
            userNameDisplay.textContent = '👤 ' + data.username;
            userChip.style.display = 'inline-flex';
          }
        }
      }
    } catch (_) {}
  }
  checkAuth();

  const logoutBtn = document.getElementById('logoutBtn');
  if (logoutBtn) {
    logoutBtn.addEventListener('click', async () => {
      try {
        await fetch('/api/auth/logout', { method: 'POST' });
      } finally {
        window.location.href = '/login';
      }
    });
  }

  console.info('%c⚡ Infosys AI%c · Linear design system · LangGraph + ChromaDB',
    'color:#7170ff;font-weight:700', 'color:#8a8f98;font-weight:400'
  );
})();
