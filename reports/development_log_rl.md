# Development Log: RL for Sequential RecSys

## 1. Подготовка данных и архитектура
Для задачи next-item prediction мы использовали архитектуру GPT-2. Первоначально словарь состоял из 9.4 миллионов уникальных `item_id` из Yambda, что приводило к **OOM (Out of Memory)** на выходном слое (требовалось 65 ГБ видеопамяти). 
**Решение:** Мы переиндексировали все `item_id` в компактный диапазон `[0, 499381]` и сохранили маппинг в `item2idx.json`. Это уменьшило выходной слой в 18 раз и позволило обучать модель батчами по 16 на RTX 4070 Ti.

## 2. Обучение базовой модели (GPT-2)
Модель обучалась 5 эпох. Loss упал с 11.78 до 8.59. 
*Проблема:* Модель ругалась на отсутствие `attention_mask` при генерации.
*Решение:* В RL-скрипте мы вручную реализовали left-padding (дополнение нулями слева), чтобы GPT-2 корректно понимала, где заканчивается история пользователя.

## 3. Reward Model
Мы использовали историю пользователя для предсказания `played_ratio_pct` следующего трека.
*Проблема:* Модель быстро переобучается (Train MSE падает, Valid MSE растет после 1-й эпохи). Это связано с огромным словарем и малым количеством данных.
*Решение:* Применили раннюю остановку (Early Stopping) на 1-й эпохе. Лучший Valid MSE составил 0.167. Мы добавили Dropout в конфигурацию GPT-2 (`resid_pdrop=0.2`), чтобы стабилизировать обучение.

## 4. RL Дообучение (REINFORCE)
*Проблема 1:* `CUDA error: device-side assert triggered`. Модель генерировала токены, выходящие за пределы словаря, либо длина последовательности превышала `MAX_LEN` (30), что ломало позиционные эмбеддинги.
*Решение:* Добавили `clamp(0, vocab_size-1)` для сгенерированных токенов и принудительно обрезали тензоры до `[-MAX_LEN:]` перед подачей в Reward Model.
*Проблема 2:* Модель "сходила с ума" и начинала генерировать один и тот же популярный трек.
*Решение:* Внедрили **KL-штраф** (`BETA=0.1`) к базовой политике (frozen GPT-2). Это удержало модель от полного коллапса, хотя Diversity все равно упал (это ожидаемое поведение при оптимизации reward).

## 5. Анализ результатов
В итоге RL-модель увеличила средний Reward на 1.07%, но потеряла 36% разнообразия (Mode Collapse). Это классический пример того, как Reward Model может быть "взломана" (Reward Hacking) политикой, которая ищет локальные максимумы (популярные треки).

Эксперимент воспроизводит базовую идею REINFORCE (Williams, 1992) в парадигме RLHF (Ouyang et al., 2022), адаптируя её для последовательной рекомендательной системы на датасете Yambda. Наблюдаемый компромисс между Reward и Diversity согласуется с выводами обзора Lin et al. (2021) о том, что для sequential recommendation критически важен дизайн функции награды, так как оптимизация краткосрочных сигналов (например, played_ratio_pct) может приводить к коллапсу разнообразия и нежелательному поведению политики.

## 6. Ссылки на источники (References)

1. [**Williams, R. J. (1992).**]((https://scispace.com/pdf/simple-statistical-gradient-following-algorithms-for-39sidawxep.pdf)) Simple statistical gradient-following algorithms for connectionist reinforcement learning. (Оригинальный алгоритм REINFORCE).
2. [**Christiano, P. F., et al. (2017).**](https://arxiv.org/pdf/1706.03741) Deep Reinforcement Learning from Human Preferences. (Основы обучения Reward Model на предпочтениях).
3. [**Ouyang, L., et al. (2022).**](https://arxiv.org/pdf/2203.02155) Training language models to follow instructions with human feedback. (Канонический пайплайн RLHF: SFT → RM → PPO).
4. [**Schulman, J., et al. (2017).**](https://arxiv.org/pdf/1707.06347) Proximal Policy Optimization Algorithms. (Более стабильная альтернатива REINFORCE, обсуждается в Future Work).
5. [**Lin, Y., et al. (2021).**](https://arxiv.org/html/2109.10665v4) A Survey on Reinforcement Learning for Recommender Systems. (принята в *IEEE TNNLS*, 2023). (Фундаментальный обзор RL в RecSys, включая sequential recommendation и анализ проблем дизайна награды).
6. [**Matveev, A., et al. (2026).**](https://arxiv.org/pdf/2607.02818) Session-Level Optimization for Large-Scale Retrieval using REINFORCE with Multi-Step Off-Policy Correction. (Применение REINFORCE на датасете Yambda-5B).