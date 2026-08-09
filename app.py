import json, re, math
import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go
from scipy.optimize import curve_fit, minimize
from scipy import stats

st.set_page_config(page_title="Socioeconomic Phenomena Analyzer", layout="wide")

C_DATA, C_MODEL, C_FC, C_ALT = "#e74c3c", "#2980b9", "#27ae60", "#8e44ad"

# ==================== ИНФРАСТРУКТУРА ====================

def lay(fig, title="", x="", y="", h=380, hover="x unified"):
    fig.update_layout(title=title, xaxis_title=x, yaxis_title=y, height=h,
                      margin=dict(l=10, r=10, t=45, b=10), template="plotly_white",
                      legend=dict(orientation="h", y=-0.18), hovermode=hover)
    return fig

def M(label, value, help=None):
    return (label, value, help)

def _r2(y, yh):
    y = np.asarray(y, float); yh = np.asarray(yh, float)
    sst = np.sum((y - y.mean()) ** 2)
    return float(1 - np.sum((y - yh) ** 2) / sst) if sst > 0 else float("nan")

def _pts(d, key, xk, yk):
    s = d.get(key, [])
    return (np.array([p[xk] for p in s], float), np.array([p[yk] for p in s], float))

def resolve_url(u: str) -> str:
    u = u.strip()
    m = re.search(r"drive\.google\.com/file/d/([\w-]+)", u)
    if m: return f"https://drive.google.com/uc?export=download&id={m.group(1)}&confirm=t"
    m = re.search(r"drive\.google\.com/(?:open|uc)\?(?:export=download&)?id=([\w-]+)", u)
    if m: return f"https://drive.google.com/uc?export=download&id={m.group(1)}&confirm=t"
    if "github.com" in u and "/blob/" in u:
        return u.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
    return u

@st.cache_data(ttl=180, show_spinner=False)
def fetch(url: str) -> str:
    r = requests.get(resolve_url(url), timeout=25)
    r.raise_for_status()
    r.encoding = "utf-8"
    return r.text

def parse_json(txt: str) -> dict:
    txt = txt.strip()
    txt = re.sub(r"^```(?:json)?|```$", "", txt, flags=re.M).strip()
    try:
        return json.loads(txt)
    except Exception:
        i, j = txt.find("{"), txt.rfind("}")
        if i >= 0 and j > i:
            return json.loads(txt[i:j + 1])
        raise

# ==================== МОДЕЛИ ====================

def f_logistic(t, K, r, t0):
    return K / (1 + np.exp(np.clip(-r * (np.asarray(t, float) - t0), -500, 500)))

def r_logistic(d):
    figs, met, notes = [], [], []
    if d.get("series"):
        t, y = _pts(d, "series", "t", "y")
        p0 = [d.get("K_hint") or max(y) * 1.25, 0.5, float(np.median(t))]
        try:
            popt, _ = curve_fit(f_logistic, t, y, p0=p0, maxfev=40000)
        except Exception:
            popt = p0
        K, r, t0 = [float(v) for v in popt]
        met.append(M("R² подгонки", f"{_r2(y, f_logistic(t, K, r, t0)):.3f}"))
    else:
        p = d["params"]; K, r, t0 = float(p["K"]), float(p["r"]), float(p["t0"])
        t = np.linspace(t0 - 6 / max(r, .01), t0 + 6 / max(r, .01), 30); y = f_logistic(t, K, r, t0)
    hz = float(d.get("forecast_periods", max((t.max() - t.min()) * 0.6, 3)))
    grid = np.linspace(t.min(), t.max() + hz, 300)
    now = float(d.get("t_now", t.max())); y_now = float(f_logistic(now, K, r, t0)); frac = y_now / K

    fig = go.Figure()
    if d.get("series"):
        fig.add_trace(go.Scatter(x=t, y=y, mode="markers", name="Данные",
                                 marker=dict(color=C_DATA, size=9)))
    fig.add_trace(go.Scatter(x=grid[grid <= now], y=f_logistic(grid[grid <= now], K, r, t0),
                             mode="lines", name="Модель", line=dict(color=C_MODEL, width=3)))
    fig.add_trace(go.Scatter(x=grid[grid >= now], y=f_logistic(grid[grid >= now], K, r, t0),
                             mode="lines", name="Прогноз", line=dict(color=C_FC, width=3, dash="dash")))
    fig.add_hline(y=K, line=dict(color="gray", dash="dot"), annotation_text="Потолок K")
    fig.add_vline(x=t0, line=dict(color=C_ALT, dash="dot"), annotation_text="Перелом t₀")
    figs.append(lay(fig, "", d.get("t_label", "Время"), d.get("y_label", "Значение")))

    ln9 = math.log(9)
    t10, t90 = t0 - ln9 / r, t0 + ln9 / r
    phase = ("Зарождение (<10% потолка)" if frac < .1 else
             "УСКОРЕНИЕ (10–50%) — золотое окно" if frac < .5 else
             "Замедление (50–90%)" if frac < .9 else "Насыщение (>90%)")
    met += [M("Потолок K", f"{K:,.0f}"), M("Скорость r", f"{r:.3f}"),
            M("Пройдено пути", f"{frac*100:.1f}%"), M("Точка перелома t₀", f"{t0:.2f}")]
    notes += [f"**Фаза: {phase}.** Сейчас реализовано {frac*100:.1f}% потенциала.",
              f"Взрывной участок кривой: **{t10:.1f} → {t90:.1f}** (80% всего роста происходит здесь).",
              f"Максимальная скорость роста — {r*K/4:,.1f} ед./период — достигается в t₀={t0:.1f}.",
              "Ошибка №1 в таких системах — линейная экстраполяция текущего темпа: после t₀ ускорение падает."]
    return dict(figs=figs, metrics=met, notes=notes)

def r_exponential(d):
    met, notes = [], []
    f = lambda t, x0, r: x0 * np.exp(np.clip(r * np.asarray(t, float), -500, 500))
    if d.get("series"):
        t, y = _pts(d, "series", "t", "y")
        try: popt, _ = curve_fit(f, t - t.min(), y, p0=[y[0] if y[0] > 0 else 1, .1], maxfev=40000)
        except Exception: popt = [y[0], .1]
        x0, r = [float(v) for v in popt]; t0 = t.min()
        met.append(M("R²", f"{_r2(y, f(t - t0, x0, r)):.3f}"))
    else:
        p = d["params"]; x0, r = float(p["x0"]), float(p["r"]); t0 = 0.0
        t = np.linspace(0, 20, 50); y = f(t, x0, r)
    hz = float(d.get("forecast_periods", 5))
    g = np.linspace(0, (t.max() - t0) + hz, 200)
    fig = go.Figure()
    if d.get("series"):
        fig.add_trace(go.Scatter(x=t, y=y, mode="markers", name="Данные", marker=dict(color=C_DATA, size=9)))
    fig.add_trace(go.Scatter(x=g + t0, y=f(g, x0, r), mode="lines", name="Модель",
                             line=dict(color=C_MODEL, width=3)))
    dbl = math.log(2) / r if r > 0 else float("inf")
    met += [M("Темп r", f"{r:.4f}"), M("CAGR", f"{(math.exp(r)-1)*100:.1f}%"),
            M("Удвоение за", f"{dbl:.2f} периода" if np.isfinite(dbl) else "—")]
    notes += [f"Каждые **{dbl:.1f}** периода величина удваивается — это {(math.exp(r)-1)*100:.1f}% в период.",
              "Чистая экспонента в социальных системах не живёт долго: ищите потолок K и переходите на S-кривую.",
              f"Через {hz:.0f} периодов при неизменном r: **{f((t.max()-t0)+hz, x0, r):,.0f}**."]
    return dict(figs=[lay(fig, "", d.get("t_label", "Время"), d.get("y_label", "Значение"))],
                metrics=met, notes=notes)

def r_gompertz(d):
    f = lambda t, K, b, c: K * np.exp(-b * np.exp(np.clip(-c * np.asarray(t, float), -500, 500)))
    met, notes = [], []
    if d.get("series"):
        t, y = _pts(d, "series", "t", "y")
        try: popt, _ = curve_fit(f, t, y, p0=[max(y) * 1.2, 1.0, .3], maxfev=40000)
        except Exception: popt = [max(y) * 1.2, 1.0, .3]
        K, b, c = [float(v) for v in popt]
        met.append(M("R²", f"{_r2(y, f(t, K, b, c)):.3f}"))
    else:
        p = d["params"]; K, b, c = float(p["K"]), float(p["b"]), float(p["c"])
        t = np.linspace(0, 20, 60); y = f(t, K, b, c)
    g = np.linspace(t.min(), t.max() + float(d.get("forecast_periods", 5)), 300)
    infl = math.log(b) / c if b > 0 and c > 0 else float("nan")
    fig = go.Figure()
    if d.get("series"):
        fig.add_trace(go.Scatter(x=t, y=y, mode="markers", name="Данные", marker=dict(color=C_DATA, size=9)))
    fig.add_trace(go.Scatter(x=g, y=f(g, K, b, c), mode="lines", name="Гомпертц", line=dict(color=C_MODEL, width=3)))
    if np.isfinite(infl): fig.add_vline(x=infl, line=dict(color=C_ALT, dash="dot"), annotation_text="Перелом")
    met += [M("Асимптота K", f"{K:,.0f}"), M("Перелом в t", f"{infl:.2f}"), M("Скорость c", f"{c:.3f}")]
    notes += ["Гомпертц асимметричен: медленный старт, ранний перелом (на 37% потолка), длинный хвост насыщения.",
              f"Перелом уже {'пройден' if t.max() > infl else 'впереди'} — окно агрессивных инвестиций {'закрывается' if t.max() > infl else 'открыто'}."]
    return dict(figs=[lay(fig, "", d.get("t_label", "Время"), d.get("y_label", "Значение"))], metrics=met, notes=notes)

def r_power_law(d):
    met, notes, figs = [], [], []
    if d.get("observations"):
        rk, v = _pts(d, "observations", "rank", "value")
        m = (rk > 0) & (v > 0); rk, v = rk[m], v[m]
        s, i, rr, _, _ = stats.linregress(np.log(rk), np.log(v))
        alpha, C = -float(s), float(np.exp(i))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=rk, y=v, mode="markers", name="Данные", marker=dict(color=C_DATA, size=9)))
        fig.add_trace(go.Scatter(x=rk, y=C * rk ** (-alpha), mode="lines", name=f"C·rank^-{alpha:.2f}",
                                 line=dict(color=C_MODEL, width=3)))
        fig.update_xaxes(type="log"); fig.update_yaxes(type="log")
        figs.append(lay(fig, "Ранг–размер (log-log)", "Ранг", "Величина", hover="closest"))
        n = len(rk); full = C * np.arange(1, max(n, 50) + 1) ** (-alpha)
        top20 = full[:max(1, int(len(full) * .2))].sum() / full.sum() * 100
        met += [M("Экспонента α", f"{alpha:.2f}"), M("R² (log-log)", f"{rr**2:.3f}"),
                M("Доля топ-20%", f"{top20:.1f}%")]
        notes.append(f"Топ-20% объектов держат **{top20:.1f}%** всей массы — правило Парето {'подтверждается' if top20>60 else 'слабое'}.")
    else:
        x = np.array(d["samples"], float); x = x[x > 0]; xmin = float(d.get("xmin", np.percentile(x, 50)))
        tail = x[x >= xmin]
        alpha = 1 + len(tail) / np.sum(np.log(tail / xmin))
        xs = np.sort(x); ccdf = 1 - np.arange(len(xs)) / len(xs)
        fig = go.Figure(go.Scatter(x=xs, y=ccdf, mode="markers", marker=dict(color=C_DATA, size=6), name="P(X>x)"))
        fig.update_xaxes(type="log"); fig.update_yaxes(type="log")
        figs.append(lay(fig, "Хвост распределения", "x", "P(X > x)", hover="closest"))
        met += [M("α (Hill)", f"{alpha:.2f}"), M("x_min", f"{xmin:,.1f}"), M("Наблюдений в хвосте", f"{len(tail)}")]
    a = met[0][1]
    notes += [f"α = {a}: {'бесконечная дисперсия — средним пользоваться НЕЛЬЗЯ' if float(a)<3 else 'дисперсия конечна'}; "
              f"{'среднее не определено — только медианы и квантили' if float(a)<2 else ''}",
              "В степенном законе редкое событие определяет итог: планируйте портфель, а не средний случай."]
    return dict(figs=figs, metrics=met, notes=notes)

def r_lognormal(d):
    x = np.array(d.get("samples", []), float); x = x[x > 0]
    if len(x):
        mu, sg = float(np.mean(np.log(x))), float(np.std(np.log(x), ddof=1))
    else:
        mu, sg = float(d["params"]["mu"]), float(d["params"]["sigma"])
    med, mean = math.exp(mu), math.exp(mu + sg ** 2 / 2)
    grid = np.linspace(max(1e-9, math.exp(mu - 3.5 * sg)), math.exp(mu + 3.5 * sg), 400)
    pdf = stats.lognorm.pdf(grid, s=sg, scale=math.exp(mu))
    fig = go.Figure()
    if len(x): fig.add_trace(go.Histogram(x=x, histnorm="probability density", name="Данные",
                                          marker=dict(color=C_DATA), opacity=.55))
    fig.add_trace(go.Scatter(x=grid, y=pdf, mode="lines", name="Логнормальная", line=dict(color=C_MODEL, width=3)))
    fig.add_vline(x=med, line=dict(color=C_ALT, dash="dot"), annotation_text="медиана")
    fig.add_vline(x=mean, line=dict(color=C_FC, dash="dot"), annotation_text="среднее")
    p10, p90 = stats.lognorm.ppf([.1, .9], s=sg, scale=math.exp(mu))
    met = [M("Медиана", f"{med:,.1f}"), M("Среднее", f"{mean:,.1f}"),
           M("σ (лог)", f"{sg:.2f}"), M("P90/P10", f"{p90/p10:.1f}×")]
    thr = d.get("threshold")
    if thr is not None:
        met.append(M(f"P(X > {thr})", f"{(1-stats.lognorm.cdf(thr, s=sg, scale=math.exp(mu)))*100:.1f}%"))
    notes = [f"Среднее выше медианы на **{(mean/med-1)*100:.0f}%** — «средний» объект встречается реже, чем кажется.",
             f"Разброс P90/P10 = {p90/p10:.1f}× : планировать по среднему = систематически ошибаться.",
             "Логнормальность = результат множества мультипликативных факторов (успех рождает успех)."]
    return dict(figs=[lay(fig, "", "Значение", "Плотность", hover="closest")], metrics=met, notes=notes)

def r_weibull(d):
    if d.get("durations"):
        x = np.sort(np.array(d["durations"], float)); x = x[x > 0]
        F = (np.arange(1, len(x) + 1) - .3) / (len(x) + .4)
        s, i, _, _, _ = stats.linregress(np.log(x), np.log(-np.log(1 - F)))
        k, lam = float(s), float(math.exp(-i / s))
    else:
        k, lam = float(d["params"]["k"]), float(d["params"]["lam"]); x = None
    t = np.linspace(.01, lam * 2.5, 300)
    S = np.exp(-(t / lam) ** k); h = (k / lam) * (t / lam) ** (k - 1)
    f1 = go.Figure([go.Scatter(x=t, y=S, mode="lines", name="Выживание S(t)", line=dict(color=C_MODEL, width=3))])
    f1.add_hline(y=.5, line=dict(color="gray", dash="dot"))
    f2 = go.Figure([go.Scatter(x=t, y=h, mode="lines", name="Интенсивность отказа", line=dict(color=C_DATA, width=3))])
    med = lam * (math.log(2)) ** (1 / k)
    reg = "детская смертность (k<1): риск падает со временем" if k < .95 else \
          "случайные отказы (k≈1): риск постоянен" if k < 1.15 else "износ (k>1): риск растёт со временем"
    met = [M("Форма k", f"{k:.2f}"), M("Характ. жизнь λ", f"{lam:.2f}"), M("Медианная жизнь", f"{med:.2f}")]
    hz = d.get("horizon")
    if hz: met.append(M(f"Дожитие до {hz}", f"{math.exp(-(float(hz)/lam)**k)*100:.1f}%"))
    return dict(figs=[lay(f1, "Кривая выживания", "Время", "P(жив)"), lay(f2, "Риск отказа", "Время", "h(t)")],
                metrics=met, notes=[f"Режим: **{reg}**.",
                                    "Если k<1 — переживший ранний этап становится надёжнее: вкладывайтесь в прохождение старта, а не в страховку зрелой фазы."])

def r_utility(d):
    p = d.get("params", {}); a = float(p.get("alpha", .88)); b = float(p.get("beta", .88)); lam = float(p.get("lam", 2.25))
    x = np.linspace(-100, 100, 400)
    u = np.where(x >= 0, np.power(np.abs(x), a), -lam * np.power(np.abs(x), b))
    fig = go.Figure([go.Scatter(x=x, y=u, mode="lines", line=dict(color=C_MODEL, width=3), name="v(x)")])
    fig.add_hline(y=0, line=dict(color="gray")); fig.add_vline(x=0, line=dict(color="gray"))
    fig.add_trace(go.Scatter(x=x, y=x, mode="lines", name="Линейная (рацион.)", line=dict(color="gray", dash="dot")))
    met = [M("λ (неприятие потерь)", f"{lam:.2f}"), M("α выигрышей", f"{a:.2f}"), M("β потерь", f"{b:.2f}")]
    notes = [f"Потеря ощущается в **{lam:.1f}×** сильнее равного выигрыша — это цена любого «переубеждения».",
             "Вогнутость в выигрышах = осторожность при прибыли; выпуклость в потерях = риск ради отыгрыша."]
    g = d.get("gamble")
    if g:
        pr, gain, loss = float(g["p"]), float(g["gain"]), float(g["loss"])
        v = pr * gain ** a - (1 - pr) * lam * abs(loss) ** b
        ev = pr * gain - (1 - pr) * abs(loss)
        met += [M("EV сделки", f"{ev:,.1f}"), M("Субъективная ценность", f"{v:,.1f}")]
        notes.append(f"Матожидание {ev:,.1f}, но воспринимается как **{v:,.1f}** → "
                     f"{'сделку будут отвергать вопреки выгоде' if v<0<ev else 'восприятие согласуется с расчётом'}.")
    return dict(figs=[lay(fig, "Функция ценности (Канеман–Тверски)", "Исход", "Полезность", hover="closest")],
                metrics=met, notes=notes)

def r_hyper(d):
    p = d.get("params", {}); k = float(p.get("k", .5)); rho = float(p.get("rho", .08)); V = float(d.get("value", 100))
    T = float(d.get("horizon", 20)); t = np.linspace(0, T, 300)
    hyp, ex = V / (1 + k * t), V * np.exp(-rho * t)
    fig = go.Figure([go.Scatter(x=t, y=hyp, mode="lines", name="Гиперболическое (люди)", line=dict(color=C_DATA, width=3)),
                     go.Scatter(x=t, y=ex, mode="lines", name="Экспоненциальное (финмодель)", line=dict(color=C_MODEL, width=3, dash="dash"))])
    gap = ex - hyp; imax = int(np.argmax(np.abs(gap)))
    met = [M("k", f"{k:.2f}"), M("ρ", f"{rho:.2%}"),
           M("Ценность через 10 лет (люди)", f"{V/(1+k*10):,.1f}"),
           M("Ценность через 10 лет (модель)", f"{V*math.exp(-rho*10):,.1f}")]
    notes = [f"Максимальный разрыв восприятия — на горизонте **{t[imax]:.1f}**: там проекты недофинансируются сильнее всего.",
             f"Выгода через 10 лет в глазах ЛПР весит как {V/(1+k*10):,.0f} вместо {V*math.exp(-rho*10):,.0f} — "
             "поэтому длинные климатические/инфраструктурные проекты продаются плохо.",
             "Приём: сократите видимый горизонт — разбейте на 18–24-месячные вехи с измеримым результатом."]
    return dict(figs=[lay(fig, "Как обесценивается будущее", "Годы", "Воспринимаемая ценность")], metrics=met, notes=notes)

def r_logit(d):
    if d.get("observations"):
        s = d["observations"]; X = np.array([o["x"] for o in s], float); Y = np.array([o["y"] for o in s], float)
        def nll(b):
            z = np.clip(b[0] + b[1] * X, -500, 500); p = 1 / (1 + np.exp(-z))
            p = np.clip(p, 1e-9, 1 - 1e-9)
            return -np.sum(Y * np.log(p) + (1 - Y) * np.log(1 - p))
        res = minimize(nll, [0., .1], method="Nelder-Mead", options=dict(maxiter=8000))
        b0, b1 = [float(v) for v in res.x]
    else:
        b0, b1 = float(d["params"]["beta0"]), float(d["params"]["beta1"]); X = None
    lo, hi = (X.min(), X.max()) if X is not None else (-10, 10)
    span = (hi - lo) or 1
    g = np.linspace(lo - .3 * span, hi + .3 * span, 300)
    p = 1 / (1 + np.exp(-np.clip(b0 + b1 * g, -500, 500)))
    fig = go.Figure([go.Scatter(x=g, y=p, mode="lines", name="P(событие)", line=dict(color=C_MODEL, width=3))])
    if X is not None:
        fig.add_trace(go.Scatter(x=X, y=Y, mode="markers", name="Наблюдения", marker=dict(color=C_DATA, size=9)))
    x50 = -b0 / b1 if b1 != 0 else float("nan")
    fig.add_vline(x=x50, line=dict(color=C_ALT, dash="dot"), annotation_text="порог 50%")
    fig.add_hline(y=.5, line=dict(color="gray", dash="dot"))
    met = [M("Порог 50%", f"{x50:,.2f}"), M("β₁ (крутизна)", f"{b1:.3f}"),
           M("Макс. чувствительность", f"{b1/4:.3f} /ед.")]
    xq = d.get("query_x")
    if xq is not None:
        met.append(M(f"P при x={xq}", f"{1/(1+math.exp(-(b0+b1*float(xq))))*100:.1f}%"))
    notes = [f"Критическая точка — **{x50:,.2f}**: ниже неё усилия почти не конвертируются, выше — растут скачком.",
             f"Вблизи порога +1 ед. фактора даёт до **{b1/4*100:.1f} п.п.** вероятности — максимальный рычаг именно здесь.",
             "Стратегия: не размазывать ресурс, а дотолкать объект через порог."]
    return dict(figs=[lay(fig, "", d.get("x_label", "Фактор"), "Вероятность", hover="closest")], metrics=met, notes=notes)

def r_cobb(d):
    p = d.get("params", {}); A = float(p.get("A", 1)); al = float(p.get("alpha", .3)); be = float(p.get("beta", 1 - al))
    K = float(d.get("K", 100)); L = float(d.get("L", 100))
    Y = A * K ** al * L ** be
    ks = np.linspace(max(K * .2, 1), K * 2.2, 60); ls = np.linspace(max(L * .2, 1), L * 2.2, 60)
    KK, LL = np.meshgrid(ks, ls); ZZ = A * KK ** al * LL ** be
    fig = go.Figure(go.Contour(x=ks, y=ls, z=ZZ, contours_coloring="lines", line_width=2, colorscale="Blues"))
    fig.add_trace(go.Scatter(x=[K], y=[L], mode="markers+text", text=["текущая точка"], textposition="top center",
                             marker=dict(color=C_DATA, size=13)))
    mpk, mpl = al * Y / K, be * Y / L
    rts = al + be
    met = [M("Выпуск Y", f"{Y:,.1f}"), M("Отдача на масштаб", f"{rts:.2f}"),
           M("MPK (на ед. капитала)", f"{mpk:,.2f}"), M("MPL (на ед. труда)", f"{mpl:,.2f}")]
    notes = [f"Отдача на масштаб {rts:.2f}: "
             + ("**возрастающая** — рост объёма сам по себе повышает эффективность" if rts > 1.03 else
                "**убывающая** — механическое масштабирование не окупится" if rts < .97 else "постоянная"),
             f"Сейчас 1 ед. капитала даёт {mpk:,.2f}, 1 ед. труда — {mpl:,.2f}: "
             f"**вкладывать выгоднее в {'капитал/технологию' if mpk>mpl else 'людей/компетенции'}**."]
    w = d.get("wage"); rr = d.get("rental")
    if w and rr:
        opt = (al / be) * (float(w) / float(rr))
        met.append(M("Оптимум K/L", f"{opt:.2f}"))
        notes.append(f"Оптимальное соотношение K/L = {opt:.2f} против текущего {K/L:.2f} → "
                     f"{'перекос в капитал' if K/L>opt else 'перекос в труд'}.")
    return dict(figs=[lay(fig, "Изокванты выпуска", "Капитал K", "Труд L", hover="closest")], metrics=met, notes=notes)

def r_metcalfe(d):
    if d.get("observations"):
        n, v = _pts(d, "observations", "n", "v")
        m = (n > 0) & (v > 0); n, v = n[m], v[m]
        s, i, rr, _, _ = stats.linregress(np.log(n), np.log(v))
        alpha, k = float(s), float(math.exp(i)); r2 = rr ** 2
    else:
        alpha, k, r2 = float(d["params"]["alpha"]), float(d["params"]["k"]), float("nan"); n = None
    lo = n.min() if n is not None else 100
    hi = float(d.get("target_n", (n.max() * 3) if n is not None else 10000))
    g = np.geomspace(max(lo, 1), hi, 200)
    fig = go.Figure([go.Scatter(x=g, y=k * g ** alpha, mode="lines", name=f"V = k·n^{alpha:.2f}",
                                line=dict(color=C_MODEL, width=3))])
    if n is not None:
        fig.add_trace(go.Scatter(x=n, y=v, mode="markers", name="Данные", marker=dict(color=C_DATA, size=9)))
    fig.update_xaxes(type="log"); fig.update_yaxes(type="log")
    met = [M("Экспонента α", f"{alpha:.2f}"), M("R²", f"{r2:.3f}" if np.isfinite(r2) else "—"),
           M("Рост при удвоении n", f"×{2**alpha:.2f}"), M(f"V при n={hi:,.0f}", f"{k*hi**alpha:,.0f}")]
    reg = ("сетевого эффекта практически нет (α≈1) — это не платформа, а линейный сервис" if alpha < 1.15 else
           "умеренный сетевой эффект (α≈1.2–1.5) — типично для сообществ" if alpha < 1.6 else
           "сильный сетевой эффект (α>1.6) — защищаемая позиция, догнать почти невозможно")
    return dict(figs=[lay(fig, "Стоимость сети (log-log)", "Участники n", "Ценность V", hover="closest")],
                metrics=met, notes=[f"Диагноз: **{reg}**.",
                                    f"Удвоение аудитории повышает ценность в **{2**alpha:.2f} раза** — "
                                    "именно это оправдывает вложения в рост, а не в монетизацию на раннем этапе."])

def r_elasticity(d):
    p, q = _pts(d, "observations", "p", "q")
    m = (p > 0) & (q > 0); p, q = p[m], q[m]
    s, i, rr, _, _ = stats.linregress(np.log(p), np.log(q))
    eps = float(s)
    fig = go.Figure([go.Scatter(x=p, y=q, mode="markers", name="Данные", marker=dict(color=C_DATA, size=10))])
    g = np.linspace(p.min(), p.max(), 100)
    fig.add_trace(go.Scatter(x=g, y=np.exp(i) * g ** eps, mode="lines", name="Модель", line=dict(color=C_MODEL, width=3)))
    lerner = eps / (eps + 1) if eps < -1 else float("nan")
    met = [M("Эластичность ε", f"{eps:.2f}"), M("R²", f"{rr**2:.3f}"),
           M("Опт. наценка (Лернер)", f"{lerner:.2f}×MC" if np.isfinite(lerner) else "н/д")]
    notes = [("Спрос **эластичен** (|ε|>1): снижение цены увеличивает выручку; конкуренция ценовая." if eps < -1 else
              "Спрос **неэластичен** (|ε|<1): можно поднимать цену — выручка вырастет; есть рыночная власть."),
             f"±10% к цене → {abs(eps)*10:.1f}% изменения объёма в обратную сторону."]
    return dict(figs=[lay(fig, "Кривая спроса", "Цена", "Объём", hover="closest")], metrics=met, notes=notes)

def r_phillips(d):
    s = d["observations"]
    u = np.array([o["u"] for o in s], float); pi = np.array([o["pi"] for o in s], float)
    pe = np.array([o.get("pi_e", 0) for o in s], float)
    y = pi - pe
    sl, ic, rr, _, _ = stats.linregress(u, y)
    alpha = -float(sl); u_n = float(ic / alpha) if alpha != 0 else float("nan")
    fig = go.Figure([go.Scatter(x=u, y=pi, mode="markers+text", text=[str(o.get("label", "")) for o in s],
                                textposition="top center", marker=dict(color=C_DATA, size=11), name="Наблюдения")])
    g = np.linspace(u.min() * .9, u.max() * 1.1, 100)
    fig.add_trace(go.Scatter(x=g, y=pe.mean() + ic + sl * g, mode="lines", name="Кривая Филлипса",
                             line=dict(color=C_MODEL, width=3)))
    fig.add_vline(x=u_n, line=dict(color=C_ALT, dash="dot"), annotation_text="NAIRU")
    met = [M("NAIRU (естеств. безработица)", f"{u_n:.2f}%"), M("Чувствительность α", f"{alpha:.2f}"),
           M("R²", f"{rr**2:.3f}"), M("Жертвенный коэф.", f"{1/alpha:.1f}" if alpha > 0 else "—")]
    gap = float(u[-1] - u_n)
    notes = [f"Разрыв безработицы сейчас {gap:+.2f} п.п. → давление на инфляцию "
             f"**{'вниз' if gap>0 else 'вверх'}** примерно на {abs(alpha*gap):.2f} п.п. в год.",
             f"Снижение инфляции на 1 п.п. стоит около **{1/alpha:.1f} п.п.·года** безработицы.",
             "Если R² низкий — ожидания «отвязались», кривая плоская и монетарная политика работает слабо."]
    return dict(figs=[lay(fig, "", "Безработица u, %", "Инфляция π, %", hover="closest")], metrics=met, notes=notes)

def r_lv(d):
    p = d["params"]; a, b, dl, gm = [float(p[k]) for k in ("alpha", "beta", "delta", "gamma")]
    x, y = float(d.get("x0", 10)), float(d.get("y0", 5)); T = float(d.get("T", 60)); n = 4000
    dt = T / n; xs, ys, ts = [x], [y], [0.]
    for i in range(n):
        dx = a * x - b * x * y; dy = dl * x * y - gm * y
        x = max(x + dx * dt, 1e-9); y = max(y + dy * dt, 1e-9)
        xs.append(x); ys.append(y); ts.append((i + 1) * dt)
    lb = d.get("labels", {}); lx, ly = lb.get("x", "Ресурс/жертва"), lb.get("y", "Потребитель/хищник")
    f1 = go.Figure([go.Scatter(x=ts, y=xs, mode="lines", name=lx, line=dict(color=C_FC, width=2.5)),
                    go.Scatter(x=ts, y=ys, mode="lines", name=ly, line=dict(color=C_DATA, width=2.5))])
    f2 = go.Figure([go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=C_MODEL, width=2), name="Траектория")])
    xe, ye = gm / dl, a / b
    f2.add_trace(go.Scatter(x=[xe], y=[ye], mode="markers", marker=dict(color=C_ALT, size=13), name="Равновесие"))
    per = 2 * math.pi / math.sqrt(a * gm)
    return dict(figs=[lay(f1, "Динамика во времени", "Время", "Численность"),
                      lay(f2, "Фазовый портрет", lx, ly, hover="closest")],
                metrics=[M("Равновесие x*", f"{xe:,.2f}"), M("Равновесие y*", f"{ye:,.2f}"),
                         M("Период цикла", f"{per:.1f}")],
                notes=[f"Система колеблется с периодом ≈ **{per:.1f}** единиц времени — «кризис» здесь не аномалия, а фаза.",
                       "Пик потребителя отстаёт от пика ресурса — вход на пике ресурса означает вход перед сжатием.",
                       "Равновесие нейтрально: любой шок задаёт новую амплитуду навсегда (нет возврата к старому циклу)."])

def r_game(d):
    P = np.array(d["payoffs"], float)
    s1 = d.get("strategies", {}).get("A") or [f"A{i+1}" for i in range(P.shape[0])]
    s2 = d.get("strategies", {}).get("B") or [f"B{j+1}" for j in range(P.shape[1])]
    A, B = P[:, :, 0], P[:, :, 1]
    b1 = A == A.max(axis=0, keepdims=True); b2 = B == B.max(axis=1, keepdims=True)
    nash = np.argwhere(b1 & b2)
    txt = [[f"{A[i,j]:g} , {B[i,j]:g}" + ("  ★" if any((i == r and j == c) for r, c in nash) else "")
            for j in range(P.shape[1])] for i in range(P.shape[0])]
    fig = go.Figure(go.Heatmap(z=A + B, x=s2, y=s1, text=txt, texttemplate="%{text}",
                               colorscale="Blues", showscale=False))
    tot = A + B
    so = np.unravel_index(np.argmax(tot), tot.shape)
    met = [M("Равновесий Нэша", f"{len(nash)}"),
           M("Соц. оптимум", f"{s1[so[0]]} / {s2[so[1]]} = {tot[so]:g}")]
    notes = []
    for r, c in nash:
        eff = tot[r, c] / tot[so] * 100
        notes.append(f"Равновесие: **{s1[r]} / {s2[c]}** — выигрыши ({A[r,c]:g}, {B[r,c]:g}), "
                     f"{eff:.0f}% от общественного оптимума.")
    if len(nash) == 1 and tuple(nash[0]) != so:
        notes.append("⚠️ Классическая ловушка: рациональное поведение каждого приводит обе стороны к худшему исходу. "
                     "Выход — не убеждение, а изменение выплат (контракт, залог, репутация, повторяемость).")
    if len(nash) == 0 and P.shape == (2, 2):
        den1 = (B[0,0]-B[0,1]-B[1,0]+B[1,1]); den2 = (A[0,0]-A[1,0]-A[0,1]+A[1,1])
        if den1 and den2:
            pm = (B[1,1]-B[1,0])/den1; qm = (A[1,1]-A[0,1])/den2
            met.append(M("Смешанное (p, q)", f"{pm:.2f}, {qm:.2f}"))
            notes.append(f"Чистого равновесия нет — устойчива **смешанная стратегия**: {s1[0]} с вер. {pm:.2f}, "
                         f"{s2[0]} с вер. {qm:.2f}. Практически: непредсказуемость сама по себе оптимальна.")
    if len(nash) == 2:
        notes.append("Две точки равновесия = игра на координацию: выигрывает не самый сильный, а тот, "
                     "кто первым создаст фокусную точку (стандарт, публичное обязательство).")
    return dict(figs=[lay(fig, "Матрица выплат (★ = равновесие Нэша)", "Игрок B", "Игрок A", h=340, hover="closest")],
                metrics=met, notes=notes)

def r_evo(d):
    P = np.array(d["payoffs"], float); A = P[:, :, 0]
    a, b, c, dd = A[0,0], A[0,1], A[1,0], A[1,1]
    G = int(d.get("generations", 60))
    fig = go.Figure()
    for p0 in d.get("p0_list", [.05, .2, .5, .8, .95]):
        p = float(p0); path = [p]
        for _ in range(G):
            u1 = p * a + (1 - p) * b; u2 = p * c + (1 - p) * dd; ub = p * u1 + (1 - p) * u2
            p = min(max(p * u1 / ub, 0), 1) if ub > 0 else p
            path.append(p)
        fig.add_trace(go.Scatter(x=list(range(G + 1)), y=path, mode="lines", name=f"p₀={p0}"))
    names = d.get("strategies", ["Стратегия 1", "Стратегия 2"])
    if a > c and dd > b: verdict = f"Бистабильность: побеждает та, что стартует выше порога p* = {(dd-b)/((a-c)+(dd-b)):.2f}"
    elif a > c and dd < b: verdict = f"«{names[0]}» доминирует — вытеснит другую при любом старте"
    elif a < c and dd > b: verdict = f"«{names[1]}» доминирует — вытеснит другую при любом старте"
    else: verdict = f"Сосуществование: устойчивая доля «{names[0]}» = {(dd-b)/((c-a)+(dd-b)):.2f}"
    return dict(figs=[lay(fig, "Репликаторная динамика", "Поколения", f"Доля «{names[0]}»")],
                metrics=[M("Исход", verdict.split(":")[0])],
                notes=[f"**{verdict}**.",
                       "В эволюционных играх выигрывает не «лучшая» стратегия, а та, что успела набрать критическую массу."])

def r_cascade(d):
    nodes = d.get("nodes") or list(range(d.get("n", 20)))
    N = len(nodes); idx = {v: i for i, v in enumerate(nodes)}
    Adj = [[] for _ in range(N)]
    for e in d.get("edges", []):
        i, j = idx.get(e[0]), idx.get(e[1])
        if i is None or j is None: continue
        Adj[i].append(j); Adj[j].append(i)
    thr = float(d.get("threshold", .3))
    act = np.zeros(N, bool)
    for s in d.get("seeds", []):
        if s in idx: act[idx[s]] = True
    hist = [act.sum()]
    for _ in range(30):
        new = act.copy()
        for i in range(N):
            if not act[i] and Adj[i]:
                if sum(act[j] for j in Adj[i]) / len(Adj[i]) >= thr: new[i] = True
        if (new == act).all(): break
        act = new; hist.append(act.sum())
    ang = np.linspace(0, 2 * np.pi, N, endpoint=False)
    X, Y = np.cos(ang), np.sin(ang)
    ex, ey = [], []
    for i in range(N):
        for j in Adj[i]:
            if j > i: ex += [X[i], X[j], None]; ey += [Y[i], Y[j], None]
    f1 = go.Figure([go.Scatter(x=ex, y=ey, mode="lines", line=dict(color="#ccc", width=1), hoverinfo="skip", showlegend=False),
                    go.Scatter(x=X, y=Y, mode="markers+text", text=[str(v) for v in nodes], textposition="top center",
                               marker=dict(size=16, color=[C_DATA if a else "#bdc3c7" for a in act]), name="Узлы")])
    f1.update_xaxes(visible=False); f1.update_yaxes(visible=False)
    f2 = go.Figure([go.Scatter(x=list(range(len(hist))), y=hist, mode="lines+markers", line=dict(color=C_MODEL, width=3))])
    deg = [len(a) for a in Adj]
    top = [nodes[i] for i in np.argsort(deg)[::-1][:3]]
    return dict(figs=[lay(f1, "Сеть после каскада (красное = приняли)", "", "", hover="closest"),
                      lay(f2, "Охват по раундам", "Раунд", "Принявших")],
                metrics=[M("Итоговый охват", f"{act.sum()}/{N} ({act.sum()/N*100:.0f}%)"),
                         M("Раундов до остановки", f"{len(hist)-1}"), M("Порог принятия", f"{thr:.0%}")],
                notes=[f"Каскад остановился на {act.sum()/N*100:.0f}% — "
                       + ("сеть проводит идею целиком, достаточно малого посева." if act.sum()/N > .8 else
                          "распространение затухает: нужен либо другой посев, либо снижение порога (проще продукт, соцдоказательство)."),
                       f"Наибольший рычаг у узлов: **{', '.join(map(str, top))}** — начинать нужно с них."])

def r_auction(d):
    typ = d.get("type", "first_price"); n = int(d.get("n_bidders", 5))
    lo, hi = float(d.get("value_low", 0)), float(d.get("value_high", 100)); v = float(d.get("my_value", hi * .7))
    bid = v * (n - 1) / n if typ == "first_price" else v
    exp_rev = lo + (hi - lo) * (n - 1) / (n + 1)
    vs = np.linspace(lo, hi, 100)
    fig = go.Figure([go.Scatter(x=vs, y=vs * (n - 1) / n if typ == "first_price" else vs, mode="lines",
                                name="Оптимальная ставка", line=dict(color=C_MODEL, width=3)),
                     go.Scatter(x=vs, y=vs, mode="lines", name="Ставка = оценка", line=dict(color="gray", dash="dot")),
                     go.Scatter(x=[v], y=[bid], mode="markers", marker=dict(color=C_DATA, size=14), name="Вы")])
    pwin = ((v - lo) / (hi - lo)) ** (n - 1)
    return dict(figs=[lay(fig, f"Аукцион: {typ}", "Ваша оценка", "Ставка", hover="closest")],
                metrics=[M("Оптимальная ставка", f"{bid:,.1f}"), M("P(выигрыша)", f"{pwin*100:.1f}%"),
                         M("Ожид. цена продажи", f"{exp_rev:,.1f}"), M("Ожид. излишек", f"{(v-bid)*pwin:,.1f}")],
                notes=[f"Шейдинг: ставьте **{bid:,.1f}** вместо {v:,.1f} — недоставка {(1-bid/v)*100:.0f}% это цена риска переплаты.",
                       f"Каждый дополнительный участник поднимает ожидаемую цену: при n={n} продавец забирает "
                       f"{exp_rev/hi*100:.0f}% максимума. Ваш рычаг — сокращать число претендентов, а не повышать ставку."])

def r_threshold(d):
    x, y = _pts(d, "observations", "x", "y")
    o = np.argsort(x); x, y = x[o], y[o]
    best = None
    for c in x[2:-2]:
        A = np.column_stack([np.ones_like(x), np.clip(x - c, 0, None)])
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        sse = float(np.sum((y - A @ beta) ** 2))
        if best is None or sse < best[0]: best = (sse, float(c), beta)
    sse, c, beta = best
    sl, ic, rr, _, _ = stats.linregress(x, y)
    lin_sse = float(np.sum((y - (ic + sl * x)) ** 2))
    fig = go.Figure([go.Scatter(x=x, y=y, mode="markers", name="Данные", marker=dict(color=C_DATA, size=10)),
                     go.Scatter(x=x, y=beta[0] + beta[1] * np.clip(x - c, 0, None), mode="lines",
                                name="Пороговая модель", line=dict(color=C_MODEL, width=3)),
                     go.Scatter(x=x, y=ic + sl * x, mode="lines", name="Линейная", line=dict(color="gray", dash="dot"))])
    fig.add_vline(x=c, line=dict(color=C_ALT, dash="dash"), annotation_text="порог")
    gain = (1 - sse / lin_sse) * 100 if lin_sse > 0 else 0
    return dict(figs=[lay(fig, "", d.get("x_label", "Фактор"), d.get("y_label", "Результат"), hover="closest")],
                metrics=[M("Критический порог", f"{c:,.2f}"), M("Наклон после порога", f"{beta[1]:.3f}"),
                         M("Точнее линейной на", f"{gain:.0f}%")],
                notes=[f"До **{c:,.2f}** отдача почти нулевая, после — {beta[1]:.3f} на единицу. "
                       "Ресурс, вложенный ниже порога, сгорает без результата.",
                       f"Правильная тактика: концентрировать усилия, чтобы перевалить {c:,.2f}, а не распределять равномерно."])

def r_bifurcation(d):
    rlo, rhi = d.get("r_range", [2.5, 4.0])
    rs = np.linspace(float(rlo), float(rhi), 700); X, Yv = [], []
    for r in rs:
        x = .5
        for _ in range(300): x = r * x * (1 - x)
        for _ in range(60):
            x = r * x * (1 - x); X.append(r); Yv.append(x)
    fig = go.Figure(go.Scattergl(x=X, y=Yv, mode="markers", marker=dict(size=1.6, color=C_MODEL), name="Аттрактор"))
    for rv, t in [(3.0, "первое удвоение"), (3.449, "4-цикл"), (3.5699, "хаос")]:
        if rlo <= rv <= rhi: fig.add_vline(x=rv, line=dict(color=C_ALT, dash="dot"), annotation_text=t)
    cur = d.get("current_r")
    if cur: fig.add_vline(x=float(cur), line=dict(color=C_DATA, width=3), annotation_text="сейчас")
    notes = ["Пока параметр мал — исход один и предсказуем. За порогом система начинает раздваиваться: "
             "одинаковые действия дают разные результаты.",
             "Практический смысл: рядом с точкой бифуркации прогнозы бесполезны, а малое действие даёт непропорциональный эффект. "
             "Это и место максимального риска, и место максимального рычага."]
    if cur:
        v = float(cur)
        notes.append(f"Ваш параметр {v:.2f} — " + ("зона стабильности" if v < 3 else
                     "зона циклов (система колеблется между состояниями)" if v < 3.57 else
                     "зона хаоса: планировать можно только диапазонами и сценариями"))
    return dict(figs=[lay(fig, "Диаграмма бифуркаций", "Управляющий параметр", "Состояние", h=440, hover="closest")],
                metrics=[M("Порог удвоения", "3.00"), M("Начало хаоса", "3.57")], notes=notes)

def r_hyst(d):
    fx, fy = _pts(d, "forward", "x", "y"); bx, by = _pts(d, "backward", "x", "y")
    fig = go.Figure([go.Scatter(x=fx, y=fy, mode="lines+markers", name="Вход (ухудшение)",
                                line=dict(color=C_DATA, width=3)),
                     go.Scatter(x=bx, y=by, mode="lines+markers", name="Выход (восстановление)",
                                line=dict(color=C_FC, width=3, dash="dash"))])
    px = np.concatenate([fx, bx[::-1]]); py = np.concatenate([fy, by[::-1]])
    area = float(abs(np.sum(px * np.roll(py, -1) - np.roll(px, -1) * py)) / 2)
    fi = np.interp(np.sort(fx), np.sort(fx), fy[np.argsort(fx)])
    bi = np.interp(np.sort(fx), np.sort(bx), by[np.argsort(bx)])
    gap = float(np.max(np.abs(bi - fi)))
    return dict(figs=[lay(fig, "Петля гистерезиса", d.get("x_label", "Управляющий фактор"),
                          d.get("y_label", "Состояние системы"), hover="closest")],
                metrics=[M("Площадь петли", f"{area:,.1f}"), M("Макс. разрыв", f"{gap:,.2f}")],
                notes=[f"Площадь петли **{area:,.1f}** — это невозвратная цена разворота: вернуть систему "
                       "в прежнее состояние дороже, чем было её туда завести.",
                       "Отсюда правило: не допускать входа в плохое состояние дешевле, чем потом выходить. "
                       "Профилактика бьёт лечение не морально, а арифметически."])

def r_crit(d):
    s = np.array([e["size"] for e in d["events"]], float); s = np.sort(s[s > 0])
    ccdf = 1 - np.arange(len(s)) / len(s)
    xmin = float(d.get("xmin", np.percentile(s, 60))); tail = s[s >= xmin]
    alpha = 1 + len(tail) / np.sum(np.log(tail / xmin))
    fig = go.Figure([go.Scatter(x=s, y=ccdf, mode="markers", marker=dict(color=C_DATA, size=6), name="Эмпирика"),
                     go.Scatter(x=tail, y=(tail / xmin) ** (-(alpha - 1)) * (len(tail) / len(s)),
                                mode="lines", line=dict(color=C_MODEL, width=3), name=f"Хвост α={alpha:.2f}")])
    fig.update_xaxes(type="log"); fig.update_yaxes(type="log")
    q = d.get("query_size")
    met = [M("α хвоста", f"{alpha:.2f}"), M("Событий", f"{len(s)}"), M("x_min", f"{xmin:,.1f}")]
    notes = ["Распределение по степенному закону = система самоорганизовалась в критическое состояние. "
             "Катастрофа здесь не «сбой», а нормальный режим работы.",
             "Ловушка: за спокойный период система накапливает напряжение, а не становится безопаснее."]
    if q:
        q = float(q); p = (q / xmin) ** (-(alpha - 1)) * (len(tail) / len(s))
        met.append(M(f"P(размер ≥ {q:,.0f})", f"{p*100:.2f}%"))
        notes.append(f"Событие масштаба {q:,.0f} случается с вероятностью {p*100:.2f}% на наблюдение — "
                     f"то есть примерно раз в {1/max(p,1e-9):,.0f} событий.")
    return dict(figs=[lay(fig, "Распределение размеров событий", "Размер", "P(≥ размер)", hover="closest")],
                metrics=met, notes=notes)

def r_fractal(d):
    r, N = _pts(d, "scales", "r", "N")
    m = (r > 0) & (N > 0); r, N = r[m], N[m]
    sl, ic, rr, _, _ = stats.linregress(np.log(1 / r), np.log(N))
    D = float(sl)
    fig = go.Figure([go.Scatter(x=1 / r, y=N, mode="markers", marker=dict(color=C_DATA, size=11), name="Замеры"),
                     go.Scatter(x=1 / r, y=np.exp(ic) * (1 / r) ** D, mode="lines",
                                line=dict(color=C_MODEL, width=3), name=f"D = {D:.2f}")])
    fig.update_xaxes(type="log"); fig.update_yaxes(type="log")
    verdict = ("структура почти линейна (D≈1) — масштабирование даёт мало нового" if D < 1.3 else
               "выраженное самоподобие — одна методика работает на всех уровнях, копируйте шаблон" if D < 2.2 else
               "плотное заполнение пространства (D>2) — высокая связность и высокая сложность управления")
    return dict(figs=[lay(fig, "Скейлинг структуры", "1/масштаб", "Число элементов", hover="closest")],
                metrics=[M("Фрактальная размерность D", f"{D:.2f}"), M("R²", f"{rr**2:.3f}")],
                notes=[f"D = {D:.2f}: {verdict}.",
                       "Если ваша методология фрактальна — она переносится с проекта за $1M на проект за $500M без переписывания. Это и есть основание для платформы."])

def r_gbm(d):
    if d.get("prices"):
        p = np.array(d["prices"], float); lr = np.diff(np.log(p))
        mu = float(np.mean(lr) * float(d.get("periods_per_year", 1)) + .5 * np.var(lr) * float(d.get("periods_per_year", 1)))
        sg = float(np.std(lr, ddof=1) * math.sqrt(float(d.get("periods_per_year", 1)))); S0 = float(p[-1])
    else:
        pr = d["params"]; S0, mu, sg = float(pr["S0"]), float(pr["mu"]), float(pr["sigma"])
    T = float(d.get("T", 5)); n = 120; paths = int(d.get("paths", 400))
    dt = T / n; rng = np.random.default_rng(7)
    Z = rng.standard_normal((paths, n))
    S = S0 * np.exp(np.cumsum((mu - .5 * sg ** 2) * dt + sg * math.sqrt(dt) * Z, axis=1))
    S = np.hstack([np.full((paths, 1), S0), S]); t = np.linspace(0, T, n + 1)
    q = np.percentile(S, [5, 25, 50, 75, 95], axis=0)
    fig = go.Figure()
    for i, (lo, hi, nm) in enumerate([(0, 4, "5–95%"), (1, 3, "25–75%")]):
        fig.add_trace(go.Scatter(x=np.concatenate([t, t[::-1]]),
                                 y=np.concatenate([q[hi], q[lo][::-1]]), fill="toself",
                                 fillcolor=f"rgba(41,128,185,{0.15 if i==0 else 0.3})",
                                 line=dict(width=0), name=nm))
    fig.add_trace(go.Scatter(x=t, y=q[2], mode="lines", line=dict(color=C_MODEL, width=3), name="Медиана"))
    fin = S[:, -1]
    return dict(figs=[lay(fig, "Веер сценариев", "Время", "Значение")],
                metrics=[M("Дрейф μ", f"{mu:.2%}"), M("Волатильность σ", f"{sg:.2%}"),
                         M("Медиана через T", f"{np.median(fin):,.1f}"),
                         M("P(ниже старта)", f"{(fin<S0).mean()*100:.1f}%")],
                notes=[f"Медианный исход {np.median(fin):,.0f}, но диапазон 5–95%: **{np.percentile(fin,5):,.0f} … {np.percentile(fin,95):,.0f}**. "
                       "Точечный прогноз здесь бессмыслен.",
                       f"Вероятность оказаться ниже текущего уровня — {(fin<S0).mean()*100:.0f}%: "
                       "при σ такого размера терпение важнее точности входа."])

def r_jump(d):
    pr = d["params"]; S0, mu, sg = float(pr["S0"]), float(pr["mu"]), float(pr["sigma"])
    lam, jm, js = float(pr.get("lam", .2)), float(pr.get("jump_mean", -.15)), float(pr.get("jump_std", .1))
    T = float(d.get("T", 5)); n = 240; paths = int(d.get("paths", 600)); dt = T / n
    rng = np.random.default_rng(11)
    Z = rng.standard_normal((paths, n)); Njump = rng.poisson(lam * dt, (paths, n))
    J = Njump * (jm + js * rng.standard_normal((paths, n)))
    S = S0 * np.exp(np.cumsum((mu - .5 * sg ** 2) * dt + sg * math.sqrt(dt) * Z + J, axis=1))
    Sg = S0 * np.exp(np.cumsum((mu - .5 * sg ** 2) * dt + sg * math.sqrt(dt) * Z, axis=1))
    fig = go.Figure([go.Histogram(x=S[:, -1], name="Со скачками", opacity=.65, marker=dict(color=C_DATA)),
                     go.Histogram(x=Sg[:, -1], name="Чистый GBM", opacity=.55, marker=dict(color=C_MODEL))])
    fig.update_layout(barmode="overlay")
    dd = (S.min(axis=1) / S0 - 1)
    return dict(figs=[lay(fig, "Распределение исходов", "Значение", "Частота", hover="closest")],
                metrics=[M("Частота шоков", f"{lam:.2f}/год"), M("Средний шок", f"{jm:.0%}"),
                         M("P(просадка >30%)", f"{(dd<-.3).mean()*100:.1f}%"),
                         M("5-й перцентиль", f"{np.percentile(S[:,-1],5):,.1f}")],
                notes=[f"Скачки утяжеляют левый хвост: P(просадка >30%) = **{(dd<-.3).mean()*100:.1f}%**, "
                       "тогда как гладкая модель такое почти исключает.",
                       f"Ожидаемо {lam*float(d.get('T',5)):.1f} шоков за горизонт. Запас прочности считайте по хвосту, а не по среднему."])

def r_mc(d):
    rng = np.random.default_rng(int(d.get("seed", 42))); n = int(d.get("n", 20000)); env = {}
    for v in d["variables"]:
        nm, dist, p = v["name"], v["dist"], v.get("params", {})
        if dist == "normal": env[nm] = rng.normal(p["mean"], p["sd"], n)
        elif dist == "lognormal": env[nm] = rng.lognormal(p["mu"], p["sigma"], n)
        elif dist == "uniform": env[nm] = rng.uniform(p["low"], p["high"], n)
        elif dist == "triangular": env[nm] = rng.triangular(p["low"], p["mode"], p["high"], n)
        elif dist == "bernoulli": env[nm] = (rng.random(n) < p["p"]).astype(float)
        else: env[nm] = np.full(n, float(p.get("value", 0)))
    env["np"] = np
    res = np.asarray(eval(d["expression"], {"__builtins__": {}}, env), float)
    p5, p50, p95 = np.percentile(res, [5, 50, 95])
    fig = go.Figure(go.Histogram(x=res, nbinsx=70, marker=dict(color=C_MODEL)))
    for v, c, t in [(p5, C_DATA, "P5"), (p50, C_ALT, "медиана"), (p95, C_FC, "P95")]:
        fig.add_vline(x=v, line=dict(color=c, dash="dot"), annotation_text=t)
    thr = float(d.get("success_threshold", 0))
    return dict(figs=[lay(fig, "Распределение результата", d.get("unit", "Результат"), "Частота", hover="closest")],
                metrics=[M("Медиана", f"{p50:,.1f}"), M("Среднее", f"{res.mean():,.1f}"),
                         M("P5 … P95", f"{p5:,.0f} … {p95:,.0f}"),
                         M(f"P(> {thr:g})", f"{(res>thr).mean()*100:.1f}%")],
                notes=[f"Вероятность успеха (> {thr:g}) = **{(res>thr).mean()*100:.1f}%**.",
                       f"Разброс P5–P95 составляет {abs(p95-p5)/max(abs(p50),1e-9)*100:.0f}% от медианы — "
                       "решение принимайте по диапазону, а не по «базовому сценарию».",
                       f"Худшие 5% сценариев дают ниже {p5:,.0f} — проверьте, переживёте ли вы это."])

def r_var(d):
    r = np.array(d.get("returns") or d.get("values"), float)
    v95, v99 = np.percentile(r, [5, 1])
    es95 = r[r <= v95].mean() if (r <= v95).any() else v95
    fig = go.Figure(go.Histogram(x=r, nbinsx=60, marker=dict(color=C_MODEL)))
    fig.add_vline(x=v95, line=dict(color=C_DATA, dash="dot"), annotation_text="VaR 95%")
    fig.add_vline(x=v99, line=dict(color="#c0392b", dash="dot"), annotation_text="VaR 99%")
    return dict(figs=[lay(fig, "Распределение исходов", "Доходность / результат", "Частота", hover="closest")],
                metrics=[M("VaR 95%", f"{v95:,.2f}"), M("VaR 99%", f"{v99:,.2f}"),
                         M("ES (ожид. потери в хвосте)", f"{es95:,.2f}"), M("Хуже нуля", f"{(r<0).mean()*100:.1f}%")],
                notes=[f"В 5% худших случаев потери превысят **{abs(v95):,.2f}**, а в среднем по этому хвосту — **{abs(es95):,.2f}**.",
                       "VaR говорит «где начинается беда», ES — «насколько она глубока». Лимиты ставьте по ES."])

def r_inter(d):
    factors = d["factors"]; cells = d["cells"]
    labels = [" + ".join(c["combo"]) for c in cells]; vals = [float(c["value"]) for c in cells]
    o = np.argsort(vals)
    fig = go.Figure(go.Bar(x=[vals[i] for i in o], y=[labels[i] for i in o], orientation="h",
                           marker=dict(color=[C_FC if v == max(vals) else C_MODEL for v in [vals[i] for i in o]])))
    base = min(vals); best = max(vals)
    singles = [c for c in cells if sum(1 for s in c["combo"] if s.lower() not in ("low", "низкий", "нет", "0")) == 1]
    add = sum(float(c["value"]) - base for c in singles)
    syn = (best - base) - add
    met = [M("Лучшая комбинация", labels[int(np.argmax(vals))]), M("Максимум", f"{best:,.1f}"),
           M("Синергия сверх суммы", f"{syn:,.1f}")]
    return dict(figs=[lay(fig, "Эффект комбинаций факторов", "Значение", "", h=max(300, 45 * len(cells)), hover="closest")],
                metrics=met,
                notes=[f"Сумма одиночных эффектов даёт {base+add:,.1f}, а вместе они дают {best:,.1f} → "
                       f"**сверхаддитивность {syn:,.1f}** ({syn/max(best-base,1e-9)*100:.0f}% всего эффекта создаётся самой связкой).",
                       "Именно этот избыток не видят те, кто смотрит на факторы по отдельности — это ваш арбитраж.",
                       f"Факторы: {', '.join(factors)}. Ценность падает, если выпадает хотя бы один — связка неделима."])

def r_reg(d):
    obs = d["observations"]; xs = d["x"]; inter = d.get("interactions", [])
    Y = np.array([o["y"] for o in obs], float)
    cols, names = [np.ones(len(obs))], ["const"]
    for k in xs: cols.append(np.array([o[k] for o in obs], float)); names.append(k)
    for pair in inter:
        cols.append(np.prod([[o[k] for o in obs] for k in pair], axis=0).astype(float)); names.append("×".join(pair))
    X = np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    resid = Y - X @ beta; dof = max(len(Y) - X.shape[1], 1)
    s2 = float(resid @ resid / dof)
    try: se = np.sqrt(np.diag(s2 * np.linalg.pinv(X.T @ X)))
    except Exception: se = np.full(len(beta), np.nan)
    tv = beta / np.where(se == 0, np.nan, se)
    df = pd.DataFrame({"Переменная": names, "Коэффициент": np.round(beta, 4),
                       "t-стат": np.round(tv, 2),
                       "Значимость": ["***" if abs(t) > 2.6 else "**" if abs(t) > 2 else "" for t in tv]})
    fig = go.Figure(go.Bar(x=names[1:], y=beta[1:], marker=dict(color=[C_ALT if "×" in n else C_MODEL for n in names[1:]])))
    sig_int = [n for n, t in zip(names, tv) if "×" in n and abs(t) > 2]
    return dict(figs=[lay(fig, "Вклад факторов", "", "Коэффициент", hover="closest")],
                metrics=[M("R²", f"{_r2(Y, X@beta):.3f}"), M("Наблюдений", f"{len(Y)}")],
                tables=[("Оценки коэффициентов", df)],
                notes=([f"Значимые взаимодействия: **{', '.join(sig_int)}** — эффект одного фактора зависит от уровня другого. "
                        "Раздельная оптимизация здесь даёт неверный ответ."] if sig_int else
                       ["Значимых взаимодействий не найдено: факторы работают независимо, можно оптимизировать по отдельности."]))

def r_pca(d):
    X = np.array(d["matrix"], float); cols = d["columns"]
    Xs = (X - X.mean(0)) / np.where(X.std(0) == 0, 1, X.std(0))
    U, S, Vt = np.linalg.svd(Xs, full_matrices=False)
    ev = S ** 2 / np.sum(S ** 2)
    f1 = go.Figure([go.Bar(x=[f"PC{i+1}" for i in range(len(ev))], y=ev * 100, marker=dict(color=C_MODEL)),
                    go.Scatter(x=[f"PC{i+1}" for i in range(len(ev))], y=np.cumsum(ev) * 100,
                               mode="lines+markers", name="Накопл.", line=dict(color=C_DATA, width=3))])
    f2 = go.Figure(go.Heatmap(z=Vt[:2], x=cols, y=["PC1", "PC2"], colorscale="RdBu", zmid=0,
                              text=np.round(Vt[:2], 2), texttemplate="%{text}"))
    top1 = cols[int(np.argmax(np.abs(Vt[0])))]; top2 = cols[int(np.argmax(np.abs(Vt[1])))]
    return dict(figs=[lay(f1, "Объяснённая дисперсия, %", "", "%", hover="closest"),
                      lay(f2, "Нагрузки факторов", "", "", h=260, hover="closest")],
                metrics=[M("PC1", f"{ev[0]*100:.1f}%"), M("PC1+PC2", f"{(ev[0]+ev[1])*100:.1f}%"),
                         M("Факторов", f"{len(cols)}")],
                notes=[f"Две компоненты объясняют **{(ev[0]+ev[1])*100:.0f}%** всей вариации — "
                       f"вместо {len(cols)} показателей достаточно следить за двумя.",
                       f"PC1 определяется прежде всего «{top1}», PC2 — «{top2}». Это скрытые оси, по которым реально движется система."])

# ==================== РЕЕСТР ====================

REG = {
 "logistic": ("Динамика и рост", "S-кривая (логистическая)", "Рост с насыщением: адопция, проникновение, охват.", r_logistic),
 "exponential": ("Динамика и рост", "Экспоненциальный рост", "Компаундирование без ограничений — ранняя фаза.", r_exponential),
 "gompertz": ("Динамика и рост", "Гомпертц", "Асимметричный жизненный цикл: ранний перелом, долгий хвост.", r_gompertz),
 "power_law": ("Динамика и рост", "Степенной закон (Парето)", "Концентрация и редкие крупные события.", r_power_law),
 "lognormal": ("Динамика и рост", "Логнормальное распределение", "Размеры, доходы, сроки — правый хвост.", r_lognormal),
 "weibull": ("Динамика и рост", "Вейбулл (надёжность)", "Время до отказа и режим риска.", r_weibull),
 "utility": ("Поведение и выбор", "Функция полезности", "Как воспринимаются выигрыши и потери.", r_utility),
 "hyperbolic_discounting": ("Поведение и выбор", "Гиперболическое дисконтирование", "Почему будущее недооценивают.", r_hyper),
 "logit": ("Поведение и выбор", "Логит (вероятность выбора)", "Порог, после которого решения переключаются.", r_logit),
 "cobb_douglas": ("Производство и ресурсы", "Кобба–Дугласа", "Комбинация капитала и труда, отдача на масштаб.", r_cobb),
 "metcalfe": ("Производство и ресурсы", "Закон Меткалфа", "Ценность сети от числа участников.", r_metcalfe),
 "elasticity": ("Производство и ресурсы", "Эластичность спроса", "Чувствительность объёма к цене.", r_elasticity),
 "phillips": ("Макроэкономика", "Кривая Филлипса", "Связка инфляции и безработицы, NAIRU.", r_phillips),
 "lotka_volterra": ("Макроэкономика", "Лотка–Вольтерра", "Циклы «ресурс — потребитель», бум и спад.", r_lv),
 "game_matrix": ("Теория игр", "Матрица выплат и равновесие Нэша", "Стратегическое взаимодействие сторон.", r_game),
 "evolutionary_game": ("Теория игр", "Эволюционная динамика", "Какая стратегия захватит популяцию.", r_evo),
 "network_cascade": ("Теория игр", "Каскад в сети", "Распространение решения через связи.", r_cascade),
 "auction": ("Теория игр", "Аукцион", "Оптимальная ставка и ожидаемая цена.", r_auction),
 "threshold": ("Критические переходы", "Пороговая функция", "Точка, ниже которой усилия не работают.", r_threshold),
 "bifurcation": ("Критические переходы", "Бифуркация", "Где траектория раздваивается.", r_bifurcation),
 "hysteresis": ("Критические переходы", "Гистерезис", "Цена возврата выше цены входа.", r_hyst),
 "criticality": ("Критические переходы", "Самоорганизованная критичность", "Частота и масштаб катастроф.", r_crit),
 "fractal_scaling": ("Критические переходы", "Фрактальный скейлинг", "Переносится ли решение между масштабами.", r_fractal),
 "gbm": ("Риск и неопределённость", "Броуновское движение (GBM)", "Веер сценариев при заданной волатильности.", r_gbm),
 "jump_diffusion": ("Риск и неопределённость", "Процесс со скачками", "Шоки и тяжёлые хвосты.", r_jump),
 "monte_carlo": ("Риск и неопределённость", "Монте-Карло", "Распределение итога сложной формулы.", r_mc),
 "var_es": ("Риск и неопределённость", "VaR / Expected Shortfall", "Глубина хвостовых потерь.", r_var),
 "interaction_matrix": ("Синергия и взаимодействие", "Матрица взаимодействия", "Сверхаддитивность комбинаций.", r_inter),
 "regression_interactions": ("Синергия и взаимодействие", "Регрессия с взаимодействиями", "Какие факторы усиливают друг друга.", r_reg),
 "pca": ("Синергия и взаимодействие", "Метод главных компонент", "Скрытые оси, по которым движется система.", r_pca),
}

CONF = {"high": ("🟢", "высокая"), "medium": ("🟡", "средняя"), "low": ("🔴", "низкая")}

GEN_PROMPT = """Ты — исследователь-аналитик. Собери данные по теме «<ТЕМА>» и верни ОДИН JSON
по стандарту SPA-DATA 1.0 (файл DATA_STANDARD.md, приложен). Требования:
1) Только валидный JSON, без markdown-обёрток и комментариев.
2) Каждый блок — одна модель из реестра; поля model, title, data, insight, conclusion, actions, confidence, source_ids.
3) Все числовые ряды — реальные или явно помеченные confidence:"low" с пояснением в notes.
4) Минимум 6 блоков из разных контекстов; обязательно 1 блок про риск и 1 про синергию/взаимодействие.
5) insight — не пересказ графика, а неочевидное следствие. conclusion — что делать. actions — 2–4 конкретных шага.
6) sources — реальные ссылки с годом."""

# ==================== ИНТЕРФЕЙС ====================

st.markdown("<h2 style='margin-bottom:0'>📊 Socioeconomic Phenomena Analyzer</h2>"
            "<p style='color:#666;margin-top:4px'>Дашборд анализа явлений через математические модели</p>",
            unsafe_allow_html=True)

if "data" not in st.session_state: st.session_state.data = None

with st.sidebar:
    st.subheader("Источник данных")
    qp = st.query_params.get("data", "")
    url = st.text_input("Ссылка на JSON (Google Drive / raw GitHub)", value=qp,
                        placeholder="https://drive.google.com/file/d/.../view")
    up = st.file_uploader("или файл .json", type=["json"])
    c1, c2 = st.columns(2)
    if c1.button("Загрузить", use_container_width=True, type="primary"):
        try:
            raw = up.read().decode("utf-8") if up else fetch(url)
            st.session_state.data = parse_json(raw)
            if url and not up: st.query_params["data"] = url
            st.success("Данные загружены")
        except Exception as e:
            st.error(f"Не удалось загрузить: {e}")
    if c2.button("Обновить", use_container_width=True):
        fetch.clear()
        try:
            st.session_state.data = parse_json(fetch(url)); st.success("Обновлено")
        except Exception as e: st.error(str(e))

D = st.session_state.data

if not D:
    st.info("Загрузите файл данных, чтобы построить дашборд. Ниже — как его подготовить.")
    t1, t2 = st.tabs(["🚀 Как работать", "🤖 Промпт для ИИ"])
    with t1:
        st.markdown("""
1. Откройте `DATA_STANDARD.md` из репозитория и приложите его в чат с ИИ.
2. Дайте промпт из соседней вкладки, подставив свою тему.
3. Сохраните ответ как `data.json` → загрузите на Google Drive.
4. Настройте доступ **«Доступно всем, у кого есть ссылка»** и вставьте ссылку слева.
5. Дашборд соберётся автоматически под содержимое файла.
        """)
    with t2:
        st.code(GEN_PROMPT, language="text")
    st.stop()

meta = D.get("meta", {}); blocks = D.get("blocks", [])
srcs = {s["id"]: s for s in D.get("sources", [])}
ok, bad = [], []
for b in blocks:
    (ok if b.get("model") in REG else bad).append(b)
ctxs = []
for b in ok:
    c = REG[b["model"]][0]
    if c not in ctxs: ctxs.append(c)

with st.sidebar:
    st.divider(); st.subheader("Навигация")
    page = st.radio("Раздел", ["🗺 Обзор"] + ctxs + ["🧩 Синтез", "🩺 Валидатор"], label_visibility="collapsed")
    st.divider()
    conf_f = st.multiselect("Уровень уверенности", ["high", "medium", "low"], default=["high", "medium", "low"])
    st.caption(f"Блоков: {len(ok)} · Контекстов: {len(ctxs)}")

def render_block(b):
    key = b["model"]; ctx, name, desc, fn = REG[key]
    ic, cf = CONF.get(b.get("confidence", "medium"), ("⚪", "—"))
    st.markdown(f"### {b.get('title', name)}")
    st.caption(f"**Модель:** {name} · {desc} · Уверенность: {ic} {cf}")
    if b.get("question"): st.markdown(f"> ❓ *{b['question']}*")
    try:
        out = fn(b.get("data", {}))
    except Exception as e:
        st.error(f"Ошибка расчёта блока «{b.get('title','')}»: {e}"); st.divider(); return
    L, R = st.columns([2.1, 1])
    with L:
        for f in out.get("figs", []): st.plotly_chart(f, use_container_width=True)
        for cap, df in out.get("tables", []):
            st.caption(cap); st.dataframe(df, use_container_width=True, hide_index=True)
    with R:
        mets = out.get("metrics", [])
        for i in range(0, len(mets), 2):
            cc = st.columns(2)
            for j, m in enumerate(mets[i:i + 2]):
                cc[j].metric(m[0], m[1], help=m[2])
        if out.get("notes"):
            st.markdown("**Что показывает модель**")
            for n in out["notes"]: st.markdown(f"- {n}")
    if b.get("insight"): st.info(f"💡 **Инсайт.** {b['insight']}")
    if b.get("conclusion"): st.success(f"🎯 **Вывод.** {b['conclusion']}")
    if b.get("actions"):
        with st.expander("Что делать — конкретные шаги"):
            for a in b["actions"]: st.markdown(f"- {a}")
    ids = b.get("source_ids", [])
    if ids:
        st.caption("Источники: " + " · ".join(
            f"[{srcs[i].get('title', i)}]({srcs[i].get('url','#')})" if i in srcs else i for i in ids))
    st.divider()

if page == "🗺 Обзор":
    st.markdown(f"## {meta.get('title', 'Без названия')}")
    c = st.columns(4)
    c[0].metric("Тема", meta.get("subject", "—"))
    c[1].metric("География", meta.get("geography", "—"))
    c[2].metric("Моделей", len(ok))
    c[3].metric("Источников", len(srcs))
    if meta.get("summary"): st.info(meta["summary"])
    cnt = pd.Series([REG[b["model"]][0] for b in ok]).value_counts()
    cc = pd.Series([b.get("confidence", "medium") for b in ok]).value_counts()
    a, bb = st.columns(2)
    a.plotly_chart(lay(go.Figure(go.Bar(x=cnt.values, y=cnt.index, orientation="h",
                                        marker=dict(color=C_MODEL))), "Покрытие контекстов", "Блоков", "",
                       h=300, hover="closest"), use_container_width=True)
    bb.plotly_chart(lay(go.Figure(go.Pie(labels=[CONF.get(k, ('', k))[1] for k in cc.index],
                                         values=cc.values, hole=.55)), "Уверенность данных", h=300, hover="closest"),
                    use_container_width=True)
    st.markdown("### Ключевые выводы")
    for b in ok:
        if b.get("conclusion"):
            ic = CONF.get(b.get("confidence", "medium"), ("⚪",))[0]
            st.markdown(f"- {ic} **{b.get('title','')}** — {b['conclusion']}")
    if srcs:
        with st.expander("Источники"):
            st.dataframe(pd.DataFrame(list(srcs.values())), use_container_width=True, hide_index=True)

elif page == "🧩 Синтез":
    st.markdown("## Синтез: инсайты, выводы, действия")
    rows = [{"Контекст": REG[b["model"]][0], "Блок": b.get("title", ""),
             "Инсайт": b.get("insight", ""), "Вывод": b.get("conclusion", ""),
             "Уверенность": CONF.get(b.get("confidence", "medium"), ("", ""))[1]} for b in ok]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.markdown("### Сводный план действий")
    for b in ok:
        for a in b.get("actions", []):
            st.markdown(f"- [{REG[b['model']][0]}] {a}")

elif page == "🩺 Валидатор":
    st.markdown("## Проверка файла данных")
    errs, warns = [], []
    if not meta.get("title"): warns.append("meta.title не задан")
    if not srcs: warns.append("Нет ни одного источника")
    for i, b in enumerate(bad):
        errs.append(f"Блок #{i+1}: неизвестная модель «{b.get('model')}»")
    for b in ok:
        t = b.get("title", b.get("model"))
        for f in ("insight", "conclusion"):
            if not b.get(f): warns.append(f"«{t}»: пустое поле {f}")
        if not b.get("source_ids"): warns.append(f"«{t}»: не указаны источники")
        for sid in b.get("source_ids", []):
            if sid not in srcs: errs.append(f"«{t}»: источник {sid} отсутствует в sources")
    if errs: st.error("Ошибки:\n" + "\n".join(f"- {e}" for e in errs))
    else: st.success("Критических ошибок нет")
    if warns: st.warning("Замечания:\n" + "\n".join(f"- {w}" for w in warns))
    st.markdown("### Доступные модели")
    st.dataframe(pd.DataFrame([{"key": k, "Контекст": v[0], "Модель": v[1], "Назначение": v[2]}
                               for k, v in REG.items()]), use_container_width=True, hide_index=True)
    with st.expander("Промпт для генерации данных"): st.code(GEN_PROMPT, language="text")

else:
    st.markdown(f"## {page}")
    sel = [b for b in ok if REG[b["model"]][0] == page and b.get("confidence", "medium") in conf_f]
    if not sel: st.info("В этом разделе нет блоков с выбранным уровнем уверенности.")
    for b in sel: render_block(b)
