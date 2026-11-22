# Модельный backend

Официальный источник: [Albertzry/SpineSegPlus](https://github.com/Albertzry/SpineSegPlus), ветка `ldh-alternative`, закреплённый commit `66fd963b4e7e4f9530e6cf7b983b23348712dcab`.

Публикуемый Python CLI использует NumPy, pydicom и Pillow. `setup` создаёт отдельное окружение с `torch==2.5.1`, `nnunetv2==2.4.2`, `dynamic_network_architectures==0.3.1` и закреплённым исходником SpineSegPlus. Требования остальных моделей берутся из его `pyproject.toml`. Проверенное модельное окружение — Python 3.12 / NumPy 2.2.6. Первичная установка требует сети; inference — локальный.

Веса берутся из официального [release weights](https://github.com/Albertzry/SpineSegPlus/releases/tag/weights). Проверяемые опубликованные SHA-256:

| Архив | SHA-256 |
|---|---|
| Dataset101_TotalSpineSeg_step1.tar.gz | `826084a1837f53ea05936190e058012369e843d5726dad71ed160a41362a4264` |
| Dataset102_TotalSpineSeg_step2.tar.gz | `10a2e709d7984d4191f865255064815ea715675dd0b4d2104667711a23ebc666` |
| Dataset105_TotalSpineSeg_LDH.tar.gz | `e9e3c53d64f31f1ecd22634f783f15c7b522656b68971acd51a5203ccf5cba3a` |
| Dataset107_LDH_ROI.tar.gz | `334b236ce7b5243901ab791a911b44619e0db497b1b095eb0206510eeba319c0` |

Архивы, source checkout, исходные и производные checkpoint хранятся только в model-home. Они не включены в wheel, Git или HTML-отчёт. Перед использованием проверяются pinned commit, отсутствие tracked-правок и SHA производных весов/`plans.json`/`dataset.json`.

Преобразование checkpoint не обучает модель и не меняет тензоры: сохраняются поля inference, проверяются ключи, shape/dtype и `torch.equal` каждого тензора. Только первоначальное чтение NumPy metadata использует небольшой явный allowlist конструктора scalar/dtype; `weights_only=False` не используется. Все последующие загрузки строго weights-only. CPU threads ограничены выбранным числом, seed=42, Stage B top-k=1; пороги и clip-to-disc взяты из закреплённого upstream.

Повторное использование подготовленных локальных зависимостей:

```sh
disc-morphology setup --model-home ./models-local \
  --python /path/to/model-venv/bin/python \
  --source /path/to/pinned-spinesegplus-checkout \
  --assets /path/to/published-weight-archives
```

`--source` клонируется в model-home и проверяется по commit. `--assets` экономит скачивание, SHA остаются обязательными. Эти локальные пути не зашиты в исходный код.

## Ограничения модели

Stage A даёт максимум patch sigmoid, не калиброванную вероятность диагноза. Stage B выдаёт бинарный LDH-кандидат; классов protrusion/extrusion/sequestration в нейросети нет. Английский README описывает цель как IVD degeneration, китайская техническая документация — как LDH; экспертное определение обучающих границ не подтверждено.

В предыдущем исследовательском эксперименте расширение top-1 до top-3 автоматических ROI добавило большой передний компонент: IoU полных масок 0,215, заднего компонента около 0,819. Это техническая устойчивость, не точность относительно врача. Новое приложение не объявляет контур клинически подтверждённым. Upstream clipping к расширенному диску может скрывать удалённый мигрировавший фрагмент.

Локализация уровня по ближайшему центру автоматического диска не устанавливает происхождение мигрировавшего фрагмента. Приближённая плоскость задних точек позвонков не является сегментацией апофизарного кольца. Возврат неопределённости сохраняет эти ограничения.

Исходный код SpineSegPlus/TotalSpineSeg распространяется отдельно под LGPL-3.0. Лицензия этого CLI не заменяет условия исходников, весов и исходных наборов данных. Код моделей и веса не перераспространяются этим репозиторием.
