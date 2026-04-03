/* Global helpers used by all pages (no backend changes required). */
(function () {
  const LS_AUTH = "authUser";
  const LS_ROLE = "selectedRole";
  const LS_FLASH = "flashMessageV1";
  const LS_CSRF = "csrfTokenV1";
  const LS_COOKIE_CONSENT = "cookieConsentV1";

  const ROLE_THEMES = {
    patient: {
      // Purple + orange accent to keep patient pages visually distinct.
      a: "#7C3AED",
      b: "#F97316",
      c: "#CFFAFE",
      blob1: "rgba(207, 250, 254, .82)",
      blob2: "rgba(251, 146, 60, .28)",
      shadow: "rgba(124, 58, 237, .14)",
      border: "rgba(124, 58, 237, .28)",
      photo: "url('/static/img/clinic-photo.svg')",
    },
    nurse: {
      // Cyan + violet accent.
      a: "#0891B2",
      b: "#A855F7",
      c: "#E0E7FF",
      blob1: "rgba(224, 231, 255, .78)",
      blob2: "rgba(165, 243, 252, .62)",
      shadow: "rgba(8, 145, 178, .14)",
      border: "rgba(8, 145, 178, .28)",
      photo: "url('/static/img/clinic-photo.svg')",
    },
    doctor: {
      // Deep blue + emerald accent.
      a: "#0B3B8C",
      b: "#10B981",
      c: "#DCFCE7",
      blob1: "rgba(220, 252, 231, .70)",
      blob2: "rgba(191, 219, 255, .70)",
      shadow: "rgba(16, 185, 129, .14)",
      border: "rgba(16, 185, 129, .28)",
      photo: "url('/static/img/auth-radiology.svg')",
    },
  };

  function qs(sel, root) { return (root || document).querySelector(sel); }
  function qsa(sel, root) { return Array.from((root || document).querySelectorAll(sel)); }

  function safeJsonParse(text) {
    try { return JSON.parse(text); } catch { return null; }
  }

  async function apiFetch(path, opts) {
    const csrf = (localStorage.getItem(LS_CSRF) || "").trim();
    const res = await fetch(path, {
      credentials: "same-origin",
      headers: {
        "Content-Type": "application/json",
        ...(csrf ? { "X-CSRF-Token": csrf } : {}),
        ...(opts && opts.headers ? opts.headers : {}),
      },
      ...opts,
    });
    const text = await res.text();
    const data = text ? safeJsonParse(text) : null;
    if (!res.ok) {
      const msg =
        (data && (data.detail || data.message)) ||
        `Request failed (${res.status})`;
      const err = new Error(msg);
      err.status = res.status;
      err.data = data;
      throw err;
    }
    return data;
  }

  function setSelectedRole(role) {
    if (typeof role === "string" && role.trim()) {
      localStorage.setItem(LS_ROLE, role.trim().toLowerCase());
    }
  }
  function getSelectedRole() {
    return (localStorage.getItem(LS_ROLE) || "").trim().toLowerCase();
  }

  function applyRoleTheme(role) {
    const r = String(role || "").trim().toLowerCase();
    const t = ROLE_THEMES[r] || ROLE_THEMES.patient;
    const root = document.documentElement;
    root.dataset.role = r || "patient";
    root.style.setProperty("--primary", t.a);
    root.style.setProperty("--primary2", t.b);
    root.style.setProperty("--role-shadow", t.shadow || "rgba(15, 40, 95, .12)");
    root.style.setProperty("--role-border", t.border || "rgba(207,224,255,.95)");

    // Also keep auth variables in sync (used by login/register pages).
    root.style.setProperty("--auth-a", t.a);
    root.style.setProperty("--auth-b", t.b);
    root.style.setProperty("--auth-c", t.c);
    root.style.setProperty("--auth-blob1", t.blob1);
    root.style.setProperty("--auth-blob2", t.blob2);
    root.style.setProperty("--auth-photo", t.photo);
  }

  function setAuthUser(user) {
    localStorage.setItem(LS_AUTH, JSON.stringify(user || null));
  }
  function getAuthUser() {
    const raw = localStorage.getItem(LS_AUTH);
    if (!raw) return null;
    try { return JSON.parse(raw); } catch { return null; }
  }
  function clearAuthUser() {
    localStorage.removeItem(LS_AUTH);
    localStorage.removeItem(LS_CSRF);
  }

  function setCsrfToken(token) {
    const t = String(token || "").trim();
    if (!t) return;
    localStorage.setItem(LS_CSRF, t);
  }

  function setFlash(message, tone, kind) {
    const msg = String(message || "").trim();
    if (!msg) return;
    const payload = {
      message: msg,
      tone: (tone || "success").toString().trim().toLowerCase() || "success",
      kind: (kind || "").toString().trim().toLowerCase(),
      at: Date.now(),
    };
    try {
      localStorage.setItem(LS_FLASH, JSON.stringify(payload));
    } catch {
      // ignore storage issues
    }
  }

  function takeFlash() {
    const raw = localStorage.getItem(LS_FLASH);
    if (!raw) return null;
    localStorage.removeItem(LS_FLASH);
    try {
      const parsed = JSON.parse(raw);
      if (!parsed || !parsed.message) return null;
      return parsed;
    } catch {
      return null;
    }
  }

  function showToast(message, tone) {
    const msg = String(message || "").trim();
    if (!msg) return;

    const existing = document.querySelector(".toast");
    if (existing) existing.remove();

    const t = (tone || "info").toString().trim().toLowerCase();
    const root = document.createElement("div");
    root.className = `toast toast-${t}`;

    const body = document.createElement("div");
    body.className = "toast-msg";
    body.textContent = msg;

    const close = document.createElement("button");
    close.type = "button";
    close.className = "toast-x";
    close.setAttribute("aria-label", "Close");
    close.textContent = "\u00D7";

    function dismiss() {
      root.classList.remove("show");
      window.setTimeout(() => root.remove(), 180);
    }
    close.addEventListener("click", dismiss);

    root.appendChild(body);
    root.appendChild(close);
    document.body.appendChild(root);

    // Animate in/out.
    requestAnimationFrame(() => root.classList.add("show"));
    window.setTimeout(dismiss, 2600);
  }

  function showConfirmPopup(message, opts) {
    const msg = String(message || "").trim();
    if (!msg) return;

    const options = opts && typeof opts === "object" ? opts : {};
    const tone = (options.tone || "success").toString().trim().toLowerCase();
    const title = String(options.title || "Success").trim();
    const confirmText = String(options.confirmText || "OK").trim();
    const cancelText = String(options.cancelText || "Cancel").trim();
    const onCancel = typeof options.onCancel === "function" ? options.onCancel : null;
    const onConfirm = typeof options.onConfirm === "function" ? options.onConfirm : null;
    const showCancel = options.showCancel !== false;

    // Remove any prior popup/toast.
    const oldToast = document.querySelector(".toast");
    if (oldToast) oldToast.remove();
    const oldPopup = document.querySelector(".popup-backdrop");
    if (oldPopup) oldPopup.remove();

    const backdrop = document.createElement("div");
    backdrop.className = "popup-backdrop";
    backdrop.setAttribute("role", "dialog");
    backdrop.setAttribute("aria-modal", "true");
    backdrop.setAttribute("aria-label", title || "Message");

    const card = document.createElement("div");
    card.className = `popup popup-${tone}`;

    const badge = document.createElement("div");
    badge.className = "popup-badge";
    badge.setAttribute("aria-hidden", "true");
    badge.textContent = "\u2713";

    const h = document.createElement("div");
    h.className = "popup-title";
    h.textContent = title || "Success";

    const p = document.createElement("div");
    p.className = "popup-msg";
    p.textContent = msg;

    const actions = document.createElement("div");
    actions.className = "popup-actions";

    const btnOk = document.createElement("button");
    btnOk.type = "button";
    btnOk.className = "popup-ok";
    btnOk.textContent = confirmText || "Continue";

    function close() {
      backdrop.classList.remove("show");
      window.setTimeout(() => backdrop.remove(), 160);
      document.removeEventListener("keydown", onKey);
    }

    function onKey(e) {
      if (e.key === "Escape") {
        e.preventDefault();
        if (showCancel) {
          const cancelBtn = backdrop.querySelector(".popup-cancel");
          if (cancelBtn) cancelBtn.click();
        } else {
          btnOk.click();
        }
      }
    }

    btnOk.addEventListener("click", () => {
      close();
      if (onConfirm) onConfirm();
    });

    backdrop.addEventListener("click", (e) => {
      if (e.target === backdrop) btnOk.click();
    });

    actions.appendChild(btnOk);

    if (showCancel) {
      const btnCancel = document.createElement("button");
      btnCancel.type = "button";
      btnCancel.className = "popup-cancel";
      btnCancel.textContent = cancelText || "Cancel";
      btnCancel.addEventListener("click", () => {
        close();
        if (onCancel) onCancel();
      });
      actions.appendChild(btnCancel);
    }

    card.appendChild(badge);
    card.appendChild(h);
    card.appendChild(p);
    card.appendChild(actions);
    backdrop.appendChild(card);
    document.body.appendChild(backdrop);

    document.addEventListener("keydown", onKey);
    requestAnimationFrame(() => backdrop.classList.add("show"));
    btnOk.focus();
  }

  function initCookieBanner() {
    try {
      const existing = localStorage.getItem(LS_COOKIE_CONSENT);
      if (existing === "accepted" || existing === "declined") return;
    } catch {
      // If storage is blocked, don't show a banner we can't persist.
      return;
    }

    if (document.querySelector(".cookie-banner")) return;

    const wrap = document.createElement("div");
    wrap.className = "cookie-banner";
    wrap.setAttribute("role", "dialog");
    wrap.setAttribute("aria-modal", "false");
    wrap.setAttribute("aria-label", "Cookie notice");

    const msg = document.createElement("div");
    msg.className = "cookie-text";
    msg.textContent =
      "This site uses cookies for secure login sessions and basic app functionality. Please accept to continue using the portal.";

    const actions = document.createElement("div");
    actions.className = "cookie-actions";

    const btnDecline = document.createElement("button");
    btnDecline.type = "button";
    btnDecline.className = "cookie-btn cookie-decline";
    btnDecline.textContent = "Decline";

    const btnAccept = document.createElement("button");
    btnAccept.type = "button";
    btnAccept.className = "cookie-btn cookie-accept";
    btnAccept.textContent = "Accept";

    function dismiss() {
      wrap.classList.remove("show");
      window.setTimeout(() => wrap.remove(), 150);
    }

    btnAccept.addEventListener("click", () => {
      try { localStorage.setItem(LS_COOKIE_CONSENT, "accepted"); } catch {}
      dismiss();
    });

    btnDecline.addEventListener("click", () => {
      try { localStorage.setItem(LS_COOKIE_CONSENT, "declined"); } catch {}
      // The app relies on a session cookie; declining means we must log out.
      try { clearAuthUser(); } catch {}
      dismiss();
      window.location.href = "/";
    });

    actions.appendChild(btnDecline);
    actions.appendChild(btnAccept);
    wrap.appendChild(msg);
    wrap.appendChild(actions);
    document.body.appendChild(wrap);
    requestAnimationFrame(() => wrap.classList.add("show"));
  }

  function consumeFlashAndToast() {
    const flash = takeFlash();
    if (!flash || !flash.message) return;

    // For login/register flows, show a modal-style popup with Continue/Cancel.
    if (flash.kind === "login" || flash.kind === "register") {
      showConfirmPopup(flash.message, {
        tone: flash.tone || "success",
        title: "Success",
        confirmText: "OK",
        cancelText: "Cancel",
        onCancel: () => {
          clearAuthUser();
          window.location.href = "/login";
        },
      });
      return;
    }

    showToast(flash.message, flash.tone || "success");
  }

  function initials(nameOrEmail) {
    const v = (nameOrEmail || "").trim();
    if (!v) return "U";
    const parts = v.split(/\s+/).filter(Boolean);
    if (parts.length === 1) return parts[0].slice(0, 2).toUpperCase();
    return (parts[0][0] + parts[parts.length - 1][0]).toUpperCase();
  }

  function setText(sel, value) {
    const el = qs(sel);
    if (!el) return;
    el.textContent = value == null || value === "" ? "-" : String(value);
  }

  function formatLocal(iso) {
    if (!iso) return "-";
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return iso;
    // Force dd/mm/yyyy format (common in India/UK) with 12-hour time.
    const text = d.toLocaleString("en-GB", {
      day: "2-digit",
      month: "2-digit",
      year: "numeric",
      hour: "2-digit",
      minute: "2-digit",
      hour12: true,
    });
    // Normalize to: dd/mm/yyyy, hh:mm AM/PM
    return text.replace(/\s*(am|pm)\b/i, (m) => m.toUpperCase());
  }

  function ensureAuth(expectedRole) {
    const user = getAuthUser();
    if (!user || !user.email) {
      window.location.href = "/login";
      return null;
    }
    if (expectedRole && user.role && user.role !== expectedRole) {
      window.location.href = "/";
      return null;
    }
    if (user.role) applyRoleTheme(user.role);
    // Server-side check to prevent "back button shows logged-in page" after logout.
    try { revalidateSession(expectedRole); } catch {}
    return user;
  }

  async function revalidateSession(expectedRole) {
    try {
      const local = getAuthUser();
      const tryAuthUserFallback = async () => {
        if (!local || !local.email) throw new Error("not-auth");
        const r2 = await fetch(`/auth/user?email=${encodeURIComponent(local.email)}`, { credentials: "same-origin" });
        if (!r2.ok) throw new Error("not-auth");
        const d2 = await r2.json().catch(() => null);
        return d2 && d2.user ? d2.user : null;
      };

      const res = await fetch("/auth/me", { credentials: "same-origin" });
      let u = null;
      if (res.status === 404) {
        u = await tryAuthUserFallback();
      } else {
        if (!res.ok) throw new Error("not-auth");
        const data = await res.json().catch(() => null);
        u = data && data.user ? data.user : null;
      }
      if (!u || !u.email) throw new Error("not-auth");
      if (expectedRole && u.role && u.role !== expectedRole) throw new Error("wrong-role");
      setAuthUser(u);
      if (u.role) applyRoleTheme(u.role);
      return u;
    } catch (e) {
      clearAuthUser();
      if (String(e && e.message) === "wrong-role") window.location.replace("/");
      else window.location.replace("/login");
      return null;
    }
  }

  function wireLogoutButtons() {
    qsa("[data-logout]").forEach((btn) => {
      btn.addEventListener("click", () => {
        (async () => {
          try {
            const csrf = (localStorage.getItem(LS_CSRF) || "").trim();
            await fetch("/auth/logout", {
              method: "POST",
              credentials: "same-origin",
              headers: csrf ? { "X-CSRF-Token": csrf } : {},
            });
          } catch {}
          clearAuthUser();
          window.location.replace("/");
        })();
      });
    });
  }

  function hydrateShellUser() {
    const user = getAuthUser();
    if (!user) return;
    if (user.role) applyRoleTheme(user.role);
    qsa("[data-user-name]").forEach((el) => (el.textContent = user.full_name || user.email));
    qsa("[data-user-role]").forEach((el) => (el.textContent = (user.role || "").toUpperCase() + " PORTAL"));
    qsa("[data-user-initials]").forEach((el) => (el.textContent = initials(user.full_name || user.email)));
    qsa("[data-user-email]").forEach((el) => (el.textContent = user.email || "-"));
  }

  function setActiveNav(path) {
    const here = path || window.location.pathname;
    qsa(".nav a").forEach((a) => {
      const href = a.getAttribute("href");
      if (!href) return;
      if (href === here) a.classList.add("active");
      else a.classList.remove("active");
    });
  }

  // Expose to pages.
  window.App = {
    qs,
    qsa,
    apiFetch,
    setSelectedRole,
    getSelectedRole,
    applyRoleTheme,
    setAuthUser,
    getAuthUser,
    clearAuthUser,
    setCsrfToken,
    flash: setFlash,
    toast: showToast,
    confirmPopup: showConfirmPopup,
    consumeFlashAndToast,
    ensureAuth,
    setText,
    formatLocal,
    hydrateShellUser,
    wireLogoutButtons,
    setActiveNav,
  };

  // Apply role color theme as early as possible.
  try {
    const u = getAuthUser();
    if (u && u.role) applyRoleTheme(u.role);
    else {
      const saved = getSelectedRole();
      if (saved) applyRoleTheme(saved);
    }
  } catch {}

  // Show cookie notice once per browser.
  try { initCookieBanner(); } catch {}

  // Handle pages restored from back/forward cache (bfcache).
  window.addEventListener("pageshow", (e) => {
    if (e && e.persisted) {
      // If the session cookie was cleared on logout, force redirect away from protected pages.
      try { revalidateSession(); } catch {}
    }
  });
})();
