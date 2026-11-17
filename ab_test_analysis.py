"""
Статистический анализ A/B-теста.

Сценарий: A/B-тест нового алгоритма рекомендаций на лендинге интернет-магазина.
- Группа A (контроль): старый лендинг
- Группа B (тест): новый алгоритм рекомендаций
- Метрика: бинарная конверсия в покупку (0/1)
- Размер: 5 000 + 5 000 пользователей

Данные: синтетические, сгенерированные через numpy (seed=42) для воспроизводимости.
Реальные данные A/B-тестов компаний не публикуются — это честный подход для
демонстрации статистических методов. Если у вас есть собственный датасет
A/B-теста (CSV с колонками group и converted) — замените блок генерации на
чтение CSV, остальной код останется без изменений.

Статистические методы (порядок — от канонического к проверочному):
  1. Z-test для пропорций — основной, канонический метод для бинарной конверсии.
     Формула: z = (p_B - p_A) / sqrt(p_pool*(1-p_pool)*(1/n_A + 1/n_B))
  2. Welch's t-test (scipy.stats.ttest_ind, equal_var=False) — перекрёстная
     проверка. Для бинарных 0/1 исходов асимптотически эквивалентен z-test при
     больших n; приводится для полноты, но не является «более корректным», чем
     z-test для долей.
  3. 95% доверительный интервал для разницы конверсий (Wald).
  4. Effect size — Cohen's h (для пропорций).
  5. Вывод о значимости при α = 0.05 (двухсторонний тест как основной).

Запуск:
    python ab_test_analysis.py

Результаты сохраняются в ./results/ab-test-real-results.json (рядом со скриптом).
Все пути в скрипте относительные — код запускается из любого каталога.
"""

import json
from pathlib import Path

import numpy as np
from scipy import stats

# ---------------------------------------------------------------------------
# 0. Пути (относительно location скрипта — переносимо)
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Генерация синтетических данных A/B-теста
# ---------------------------------------------------------------------------
RNG = np.random.default_rng(seed=42)

N_CONTROL = 5_000  # группа A
N_VARIANT = 5_000  # группа B

# Истинные конверсии (B лучше — реалистичный значимый эффект)
TRUE_CONV_A = 0.042  # 4.2% — старый лендинг
TRUE_CONV_B = 0.052  # 5.2% — новый алгоритм рекомендаций

control_conversions = RNG.binomial(1, TRUE_CONV_A, size=N_CONTROL)
variant_conversions = RNG.binomial(1, TRUE_CONV_B, size=N_VARIANT)

n_conv_a = int(control_conversions.sum())
n_conv_b = int(variant_conversions.sum())
conv_rate_a = n_conv_a / N_CONTROL
conv_rate_b = n_conv_b / N_VARIANT

print("=== Сгенерированы данные A/B-теста ===")
print(f"Группа A (контроль): {N_CONTROL:,} пользователей, конверсий {n_conv_a}, "
      f"rate = {conv_rate_a:.4f} ({conv_rate_a*100:.2f}%)")
print(f"Группа B (тест):     {N_VARIANT:,} пользователей, конверсий {n_conv_b}, "
      f"rate = {conv_rate_b:.4f} ({conv_rate_b*100:.2f}%)")
print(f"Разница конверсий (B - A): {(conv_rate_b - conv_rate_a)*100:.2f} п.п.")

# ---------------------------------------------------------------------------
# 2. Z-test для пропорций — ОСНОВНОЙ метод для бинарной конверсии
# ---------------------------------------------------------------------------
p_pool = (n_conv_a + n_conv_b) / (N_CONTROL + N_VARIANT)
se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / N_CONTROL + 1 / N_VARIANT))
z_stat = (conv_rate_b - conv_rate_a) / se_pool
z_p_one = 1 - stats.norm.cdf(z_stat)        # односторонний: H1: B > A
z_p_two = 2 * z_p_one                        # двухсторонний

print("\n=== Z-test для пропорций (основной метод) ===")
print(f"p_pool: {p_pool:.4f}")
print(f"z-статистика: {z_stat:.4f}")
print(f"p-value (one-sided): {z_p_one:.6f}")
print(f"p-value (two-sided): {z_p_two:.6f}")

# ---------------------------------------------------------------------------
# 3. Welch's t-test — перекрёстная проверка (не основной метод для долей)
# ---------------------------------------------------------------------------
t_stat, t_p_one = stats.ttest_ind(
    variant_conversions, control_conversions,
    equal_var=False,         # Welch's — не предполагает равенства дисперсий
    alternative="greater",   # односторонний: H1: B > A
)
t_stat_two, t_p_two = stats.ttest_ind(
    variant_conversions, control_conversions, equal_var=False
)
print("\n=== Welch's t-test (перекрёстная проверка) ===")
print(f"t-статистика: {t_stat:.4f}")
print(f"p-value (one-sided): {t_p_one:.6f}")
print(f"p-value (two-sided): {t_p_two:.6f}")
print("Примечание: для бинарных 0/1 исходов z-test для долей каноничен; "
      "t-test асимптотически эквивалентен ему при больших n.")

# ---------------------------------------------------------------------------
# 4. 95% доверительный интервал для разницы конверсий (Wald)
# ---------------------------------------------------------------------------
se_diff = np.sqrt(
    conv_rate_a * (1 - conv_rate_a) / N_CONTROL +
    conv_rate_b * (1 - conv_rate_b) / N_VARIANT
)
alpha = 0.05
z_crit = stats.norm.ppf(1 - alpha / 2)  # 1.96 для 95%
diff = conv_rate_b - conv_rate_a
ci_low = diff - z_crit * se_diff
ci_high = diff + z_crit * se_diff

print(f"\n=== 95% доверительный интервал для разницы (B - A) ===")
print(f"Точечная оценка: {diff*100:.4f} п.п.")
print(f"95% CI: [{ci_low*100:.4f}%, {ci_high*100:.4f}%]")

# ---------------------------------------------------------------------------
# 5. Effect size — Cohen's h (для пропорций)
# ---------------------------------------------------------------------------
h = 2 * np.arcsin(np.sqrt(conv_rate_b)) - 2 * np.arcsin(np.sqrt(conv_rate_a))
if h < 0.2:
    interpretation = "малый эффект (< 0.2)"
elif h < 0.5:
    interpretation = "средний эффект (0.2 — 0.5)"
elif h < 0.8:
    interpretation = "большой эффект (0.5 — 0.8)"
else:
    interpretation = "очень большой эффект (≥ 0.8)"
print(f"\n=== Effect size ===")
print(f"Cohen's h: {h:.4f}")
print(f"Интерпретация: {interpretation}")

# ---------------------------------------------------------------------------
# 6. Статистическая значимость (двухсторонний z-test как основной)
# ---------------------------------------------------------------------------
is_significant = z_p_two < alpha
print(f"\n=== Вывод (основной — двухсторонний z-test для пропорций) ===")
print(f"Уровень значимости α = {alpha}")
print(f"p-value (двухсторонний, z-test) = {z_p_two:.6f}")
print(f"p-value < α? {is_significant}")

if is_significant:
    conclusion = (
        f"Результат статистически значим (двухсторонний z-test, p = {z_p_two:.4f} < α = {alpha}). "
        f"Отвергаем нулевую гипотезу H0: конверсии групп A и B равны. "
        f"Группа B (новый алгоритм) показала значимо более высокую конверсию: "
        f"{conv_rate_b*100:.2f}% vs {conv_rate_a*100:.2f}%, "
        f"абсолютный прирост +{(conv_rate_b - conv_rate_a)*100:.2f} п.п., "
        f"относительный прирост +{(conv_rate_b - conv_rate_a)/conv_rate_a*100:.1f}%. "
        f"95% доверительный интервал для разницы не содержит 0: "
        f"[{ci_low*100:.2f}%, {ci_high*100:.2f}%]. "
        f"Effect size (Cohen's h = {h:.3f}) — {interpretation}. "
        f"Важно: при размере групп 5 000+5 000 статистическая значимость достигается "
        f"даже при малом effect size, поэтому решение о раскатке нужно принимать с "
        f"учётом практической значимости (бизнес-выгода от +1.22 п.п. vs стоимость "
        f"разработки и поддержки нового алгоритма)."
    )
else:
    conclusion = (
        f"Результат не статистически значим (двухсторонний z-test, p = {z_p_two:.4f} ≥ α = {alpha}). "
        f"Не можем отвергнуть H0. Различия могут быть случайны. "
        f"95% CI содержит 0: [{ci_low*100:.2f}%, {ci_high*100:.2f}%]. "
        f"Рекомендация: увеличить выборку или принять H0."
    )
print(conclusion)

# ---------------------------------------------------------------------------
# 7. Подготовка данных для графика
# ---------------------------------------------------------------------------
chart_data = [
    {"name": "Группа A (контроль)", "value": float(conv_rate_a * 100)},
    {"name": "Группа B (тест)", "value": float(conv_rate_b * 100)},
    {"name": "Lift (B - A)", "value": float((conv_rate_b - conv_rate_a) * 100)},
]
chart_config = [
    {"dataKey": "value", "color": "var(--chart-2)", "label": "Конверсия, %"},
]
real_results = [
    {"metric": "P-value (двухсторонний, z-test)", "value": f"{z_p_two:.4f}"},
    {"metric": "Конверсия A (контроль)", "value": f"{conv_rate_a*100:.2f}%"},
    {"metric": "Конверсия B (тест)", "value": f"{conv_rate_b*100:.2f}%"},
    {"metric": "Статистическая значимость", "value":
        "Значимо (p < 0.05)" if is_significant else "Не значимо (p ≥ 0.05)"},
]

# ---------------------------------------------------------------------------
# 8. Сохранение в JSON (в results/ рядом со скриптом)
# ---------------------------------------------------------------------------
output = {
    "chartType": "bar",
    "chartData": chart_data,
    "chartConfig": chart_config,
    "realMetrics": {
        "test_name": "A/B-тест: влияние нового алгоритма рекомендаций на конверсию",
        "metric": "Конверсия в покупку (бинарная, 0/1)",
        "data_source": "Синтетические данные, сгенерированные numpy (seed=42)",
        "is_synthetic": True,
        "synthetic_note": (
            "Данные сгенерированы через numpy.random.binomial с seed=42 для воспроизводимости. "
            "Размеры групп 5000+5000, реалистичные конверсии e-commerce (4-5%). "
            "Реальные данные A/B-тестов компаний не публикуются."
        ),
        "n_control": N_CONTROL,
        "n_variant": N_VARIANT,
        "n_conv_control": n_conv_a,
        "n_conv_variant": n_conv_b,
        "conv_rate_control": float(conv_rate_a),
        "conv_rate_variant": float(conv_rate_b),
        "absolute_diff": float(diff),
        "relative_lift_pct": float((conv_rate_b - conv_rate_a) / conv_rate_a * 100),
        # z-test (основной)
        "z_statistic": float(z_stat),
        "z_p_value_one_sided": float(z_p_one),
        "z_p_value_two_sided": float(z_p_two),
        # t-test (перекрёстная проверка)
        "t_statistic": float(t_stat),
        "t_p_value_one_sided": float(t_p_one),
        "t_p_value_two_sided": float(t_p_two),
        # CI
        "ci_95_low": float(ci_low),
        "ci_95_high": float(ci_high),
        "ci_95_level": 0.95,
        # effect size
        "effect_size_cohens_h": float(h),
        "effect_size_interpretation": interpretation,
        # вывод
        "alpha": alpha,
        "is_significant": bool(is_significant),
        "primary_method": "Z-test для пропорций (двухсторонний)",
        "conclusion": conclusion,
        "method": "Z-test для пропорций (основной) + Welch's t-test (перекрёстная проверка) "
                  "+ Wald 95% CI + Cohen's h",
    },
    "realResults": real_results,
}

results_path = RESULTS_DIR / "ab-test-real-results.json"
with open(results_path, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"\n✓ Saved → {results_path.relative_to(SCRIPT_DIR)}")
print(f"\nФинальные метрики:")
print(f"  P-value (z-test, two-sided): {z_p_two:.4f}")
print(f"  P-value (t-test, two-sided): {t_p_two:.4f}")
print(f"  Конверсия A → B: {conv_rate_a*100:.2f}% → {conv_rate_b*100:.2f}%")
print(f"  Cohen's h: {h:.4f} ({interpretation})")
print(f"  Значимо: {is_significant}")
