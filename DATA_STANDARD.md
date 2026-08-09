# SPA-DATA 1.0 — стандарт файла данных для Socioeconomic Phenomena Analyzer

Файл — один валидный JSON в UTF-8. Никаких markdown-обёрток, комментариев, висящих запятых.

## 1. Каркас

{
  "meta": {
    "schema_version": "1.0",
    "title": "Название анализа",
    "subject": "Предмет: рынок / проект / сообщество / страна",
    "geography": "Саудовская Аравия, MENA",
    "time_unit": "year",
    "currency": "USD",
    "created": "2026-08-09",
    "summary": "3–5 предложений: что за явление и главный вывод",
    "confidence": "medium"
  },
  "sources": [
    {"id":"s1","title":"IEA Renewables 2025","url":"https://...","year":2025,"reliability":"high"}
  ],
  "blocks": [ { ...блок... } ]
}

## 2. Блок

{
  "id": "b1",
  "model": "logistic",
  "title": "Проникновение раздельного сбора в городах",
  "question": "На каком этапе адопции мы находимся?",
  "data": { ... зависит от модели, см. §3 ... },
  "insight": "Неочевидное следствие, а не пересказ графика",
  "conclusion": "Что это означает для решения",
  "actions": ["Шаг 1", "Шаг 2"],
  "confidence": "high|medium|low",
  "source_ids": ["s1"],
  "notes": "оговорки, допущения, как получены оценки"
}

Правила: `model` — строго ключ из §3. Если данных мало — ставьте `confidence:"low"` и опишите допущение в `notes`, но блок всё равно давайте. Минимум 6 блоков из разных контекстов, обязательно один про риск и один про синергию.

## 3. Реестр моделей и контракт `data`

### Контекст: Динамика и рост
| key | data |
|---|---|
| `logistic` | `{"series":[{"t":2015,"y":12}, ...], "t_label":"Год","y_label":"% городов","forecast_periods":6,"K_hint":100}` либо `{"params":{"K":100,"r":0.4,"t0":2024}}` |
| `exponential` | `{"series":[{"t":..,"y":..}], "forecast_periods":5}` либо `{"params":{"x0":10,"r":0.12}}` |
| `gompertz` | как logistic, либо `{"params":{"K":100,"b":3,"c":0.3}}` |
| `power_law` | `{"observations":[{"rank":1,"value":540},...]}` либо `{"samples":[...],"xmin":50}` |
| `lognormal` | `{"samples":[...]}` либо `{"params":{"mu":3.1,"sigma":0.8},"threshold":100}` |
| `weibull` | `{"durations":[2.1,5.4,...],"horizon":10}` либо `{"params":{"k":1.4,"lam":6}}` |

### Контекст: Поведение и выбор
| key | data |
|---|---|
| `utility` | `{"params":{"alpha":0.88,"beta":0.88,"lam":2.25},"gamble":{"p":0.5,"gain":100,"loss":80}}` |
| `hyperbolic_discounting` | `{"params":{"k":0.6,"rho":0.08},"value":100,"horizon":20}` |
| `logit` | `{"observations":[{"x":12,"y":1},...],"x_label":"Бюджет, $млн","query_x":8}` либо `{"params":{"beta0":-3,"beta1":0.4}}` |

### Контекст: Производство и ресурсы
| key | data |
|---|---|
| `cobb_douglas` | `{"params":{"A":1.2,"alpha":0.35,"beta":0.65},"K":120,"L":80,"wage":50,"rental":10}` |
| `metcalfe` | `{"observations":[{"n":100,"v":5},...],"target_n":10000}` либо `{"params":{"k":0.01,"alpha":1.6}}` |
| `elasticity` | `{"observations":[{"p":10,"q":900},...]}` |

### Контекст: Макроэкономика
| key | data |
|---|---|
| `phillips` | `{"observations":[{"u":5.2,"pi":7.1,"pi_e":6.0,"label":"2023"},...]}` |
| `lotka_volterra` | `{"params":{"alpha":0.8,"beta":0.02,"delta":0.01,"gamma":0.6},"x0":50,"y0":20,"T":60,"labels":{"x":"Спрос","y":"Мощности"}}` |

### Контекст: Теория игр
| key | data |
|---|---|
| `game_matrix` | `{"strategies":{"A":["Сотрудничать","Уклоняться"],"B":["Сотрудничать","Уклоняться"]},"payoffs":[[[3,3],[0,5]],[[5,0],[1,1]]]}` — payoffs[i][j] = [выигрыш A, выигрыш B] |
| `evolutionary_game` | `{"payoffs":[[[3,3],[0,5]],[[5,0],[1,1]]],"strategies":["Кооперация","Оппортунизм"],"generations":60,"p0_list":[0.1,0.4,0.6]}` |
| `network_cascade` | `{"nodes":["A","B","C"],"edges":[["A","B"],["B","C"]],"threshold":0.3,"seeds":["A"]}` |
| `auction` | `{"type":"first_price","n_bidders":5,"value_low":0,"value_high":100,"my_value":70}` |

### Контекст: Критические переходы
| key | data |
|---|---|
| `threshold` | `{"observations":[{"x":..,"y":..}],"x_label":"Бюджет","y_label":"Охват"}` |
| `bifurcation` | `{"r_range":[2.5,4.0],"current_r":3.2}` |
| `hysteresis` | `{"forward":[{"x":..,"y":..}],"backward":[{"x":..,"y":..}],"x_label":"Давление","y_label":"Состояние"}` |
| `criticality` | `{"events":[{"size":12},{"size":340},...],"xmin":10,"query_size":500}` |
| `fractal_scaling` | `{"scales":[{"r":1,"N":8},{"r":0.5,"N":30},...]}` |

### Контекст: Риск и неопределённость
| key | data |
|---|---|
| `gbm` | `{"params":{"S0":100,"mu":0.08,"sigma":0.25},"T":5}` либо `{"prices":[...],"periods_per_year":12,"T":3}` |
| `jump_diffusion` | `{"params":{"S0":100,"mu":0.08,"sigma":0.2,"lam":0.3,"jump_mean":-0.2,"jump_std":0.1},"T":5}` |
| `monte_carlo` | `{"variables":[{"name":"rev","dist":"triangular","params":{"low":8,"mode":12,"high":20}},{"name":"capex","dist":"normal","params":{"mean":50,"sd":8}}],"expression":"rev*5 - capex","success_threshold":0,"unit":"$ млн"}` |
| `var_es` | `{"returns":[0.02,-0.11,...]}` |

Доступные `dist`: `normal{mean,sd}`, `lognormal{mu,sigma}`, `uniform{low,high}`, `triangular{low,mode,high}`, `bernoulli{p}`, `const{value}`. В `expression` можно использовать имена переменных, арифметику и `np.`.

### Контекст: Синергия и взаимодействие
| key | data |
|---|---|
| `interaction_matrix` | `{"factors":["Waste","Energy","Carbon"],"cells":[{"combo":["Low","Low","Low"],"value":0},{"combo":["High","Low","Low"],"value":25},{"combo":["High","High","High"],"value":100}]}` — обязательно включите базовую комбинацию (все Low) и одиночные |
| `regression_interactions` | `{"observations":[{"y":..,"x1":..,"x2":..}],"x":["x1","x2"],"interactions":[["x1","x2"]]}` |
| `pca` | `{"columns":["цена","спрос","субсидия"],"matrix":[[..],[..],[..]]}` |

## 4. Правила качества
1. Числа — числами, без строк и знаков «%», «$», пробелов-разделителей.
2. Временные ряды — по возрастанию `t`; для календаря используйте реальные годы (2015, 2016…).
3. Каждый блок с фактическими данными обязан иметь `source_ids`.
4. `insight` ≤ 300 знаков, `conclusion` ≤ 200, 2–4 `actions` в повелительном наклонении.
5. Оценочные/смоделированные числа помечать `confidence:"low"` + пояснение в `notes`.
6. Не выдумывать источники. Нет источника — `confidence:"low"` и пометка «экспертная оценка».

## 5. Промпт для ИИ

> Ты — исследователь-аналитик. Тема: «<ТЕМА>». Горизонт: <ГОДЫ>. География: <РЕГИОН>.
> Собери данные и верни ОДИН валидный JSON строго по стандарту SPA-DATA 1.0 (выше).
> Минимум 6 блоков из разных контекстов; обязательно `monte_carlo` или `var_es` и обязательно `interaction_matrix`.
> Для каждого блока: реальные числа со ссылками; `insight` — неочевидное следствие; `conclusion` — что делать;
> `actions` — 2–4 конкретных шага. Никакого текста вне JSON.
