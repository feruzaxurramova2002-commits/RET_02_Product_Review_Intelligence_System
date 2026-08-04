"""Trilingual UI string catalog for the RET-02 Streamlit application.

The model itself remains English-centric (review text is never translated);
only the interface — titles, tabs, buttons, placeholders, status and
empty-state messages, and sidebar notes — is fully localized.

Usage:
    from src.i18n import t, LANGUAGES
    label = t("analyze_btn", lang="en")
"""

from __future__ import annotations

from typing import Dict

LANGUAGES: Dict[str, str] = {
    "en": "English 🇬🇧",
    "uz": "Oʻzbekcha 🇺🇿",
    "ru": "Русский 🇷🇺",
}

DEFAULT_LANGUAGE = "en"

_TRANSLATIONS: Dict[str, Dict[str, str]] = {
    "en": {
        "app_title": "📦 Product Review Intelligence System",
        "app_subtitle": "AI-powered review triage for product, CX, and e-commerce teams",
        "language_label": "🌐 Language",
        "profile_label": "👤 I am a...",
        "profile_pm_cx": "Product Manager / CX Lead",
        "profile_data_ml": "Data Analyst / ML Engineer",
        "profile_exec_biz": "Executive / Business",
        "profile_genz_student": "Gen-Z / Student",
        "tab_single": "📝 Single Review Analysis",
        "tab_batch": "📊 Batch CSV / Excel Analysis",
        "tab_summary": "📈 Executive Summary",
        "tab_model": "⚙️ Model Info",
        "single_intro_title": "Analyze one review at a time",
        "single_intro_body": (
            "Paste or type a single product review below. The system cleans the "
            "text, predicts its sentiment, flags relevant issue categories "
            "(delivery, defect, packaging, etc.), scores how actionable it is "
            "for a CX team, and tells you whether the result is confident enough "
            "to auto-process or should go to a human reviewer."
        ),
        "review_text_label": "Review text",
        "review_text_placeholder": "e.g. The package arrived crushed and the item was defective...",
        "rating_label": "Star rating (optional)",
        "review_id_label": "Review ID (optional)",
        "analyze_btn": "🔍 Analyze Review",
        "no_review_warning": "⚠️ No review entered. Please type a review or choose an upload method.",
        "result_header": "Analysis Result",
        "status_auto": "✅ AUTO-PROCESSED",
        "status_human": "🧑‍💼 HUMAN REVIEW REQUIRED",
        "confidence_label": "Overall confidence",
        "sentiment_label": "Predicted sentiment",
        "issues_label": "Detected issue categories",
        "no_issues_detected": "No specific issue detected",
        "actionability_label": "Actionability score",
        "edge_case_label": "Data-quality flag",
        "why_prediction_label": "🔎 Why this prediction?",
        "why_prediction_empty": "Not enough signal in the text to extract keyword drivers.",
        "why_prediction_intro": "Top keywords that drove this result:",
        "backbone_mode_label": "Model backbone",
        "batch_intro_title": "Analyze a whole file at once",
        "batch_intro_body": (
            "Upload a CSV or Excel file with a column named `review_text` "
            "(a `review_id` and `rating` column are optional but recommended). "
            "Every row is cleaned, scored, and routed the same way as a single "
            "review, and you can download the full results as CSV."
        ),
        "upload_label": "Upload CSV or Excel file",
        "file_preview_label": "📋 Uploaded file preview",
        "process_batch_btn": "🚀 Run Batch Analysis",
        "no_file_info": "ℹ️ No file uploaded yet. Upload a CSV or Excel file to run batch analysis.",
        "missing_column_error": "⚠️ The uploaded file must contain a `review_text` column.",
        "batch_done_msg": "Batch analysis complete.",
        "download_results_btn": "⬇️ Download results as CSV",
        "results_table_label": "Results",
        "summary_source_label": "Summary source",
        "summary_source_batch": "Use my last batch upload",
        "summary_source_demo": "Use demo dataset",
        "summary_no_data": "No batch data yet — run a batch analysis first, or switch to the demo dataset.",
        "metric_total_reviews": "Total reviews",
        "metric_auto_processed": "Auto-processed",
        "metric_human_review": "Human review required",
        "metric_avg_actionability": "Avg. actionability",
        "metric_urgent": "Urgent items",
        "metric_nss": "Net Sentiment Score",
        "top_issues_label": "Top recurring issues",
        "sentiment_dist_label": "Sentiment distribution",
        "issue_freq_label": "Issue frequency",
        "model_info_title": "Current model configuration",
        "model_mode_label": "Active backbone mode",
        "model_mode_baseline_desc": (
            "TF-IDF features + Logistic Regression / Ridge multi-task heads "
            "(issue tagging, sentiment, actionability). Trained on synthetic "
            "data at startup; fast and lightweight, ideal for free-tier hosting."
        ),
        "model_mode_offline_desc": (
            "Dependency-free keyword/rule-based fallback, active because "
            "scikit-learn training was unavailable. Reduced accuracy — treat "
            "predictions as indicative only."
        ),
        "model_threshold_label": "Confidence threshold (auto-process cutoff)",
        "model_training_rows_label": "Training rows used",
        "model_issue_classes_label": "Issue categories",
        "model_sentiment_classes_label": "Sentiment classes",
        "sidebar_note": "Language and visual profile apply across the whole app.",
        "footer_note": "RET-02 · Product Review Intelligence System · Model text stays English-only; interface is trilingual.",
    },
    "uz": {
        "app_title": "📦 Mahsulot sharhlarini tahlil qilish tizimi",
        "app_subtitle": "Mahsulot, CX va e-commerce jamoalari uchun AI asosidagi sharh saralash tizimi",
        "language_label": "🌐 Til",
        "profile_label": "👤 Men...",
        "profile_pm_cx": "Product Menejer / CX Rahbari",
        "profile_data_ml": "Data Analitik / ML Muhandis",
        "profile_exec_biz": "Rahbar / Biznes",
        "profile_genz_student": "Gen-Z / Talaba",
        "tab_single": "📝 Bitta sharhni tahlil qilish",
        "tab_batch": "📊 CSV / Excel ommaviy tahlil",
        "tab_summary": "📈 Yakuniy hisobot",
        "tab_model": "⚙️ Model haqida",
        "single_intro_title": "Bitta sharhni tahlil qiling",
        "single_intro_body": (
            "Quyiga bitta mahsulot sharhini kiriting. Tizim matnni tozalaydi, "
            "kayfiyatini (sentiment) aniqlaydi, tegishli muammo turlarini "
            "(yetkazib berish, nuqson, qadoqlash va h.k.) belgilaydi, CX jamoasi "
            "uchun qanchalik muhimligini baholaydi va natija avtomatik qayta "
            "ishlash uchun yetarlicha ishonchli yoki inson tekshiruvi kerakligini bildiradi."
        ),
        "review_text_label": "Sharh matni",
        "review_text_placeholder": "Masalan: Buyum yorilgan holda keldi va ishlamayapti...",
        "rating_label": "Yulduzcha bahosi (ixtiyoriy)",
        "review_id_label": "Sharh ID (ixtiyoriy)",
        "analyze_btn": "🔍 Sharhni tahlil qilish",
        "no_review_warning": "⚠️ Sharh kiritilmagan. Iltimos, sharh matnini kiriting yoki fayl yuklash usulini tanlang.",
        "result_header": "Tahlil natijasi",
        "status_auto": "✅ AVTOMATIK QAYTA ISHLANDI",
        "status_human": "🧑‍💼 INSON TEKSHIRUVI TALAB QILINADI",
        "confidence_label": "Umumiy ishonch darajasi",
        "sentiment_label": "Bashorat qilingan kayfiyat",
        "issues_label": "Aniqlangan muammo turlari",
        "no_issues_detected": "Aniq muammo topilmadi",
        "actionability_label": "Muhimlik (actionability) darajasi",
        "edge_case_label": "Ma'lumot sifati belgisi",
        "why_prediction_label": "🔎 Nima uchun bu natija?",
        "why_prediction_empty": "Kalit so'zlarni ajratib olish uchun matnda yetarli signal yo'q.",
        "why_prediction_intro": "Natijaga ta'sir qilgan asosiy kalit so'zlar:",
        "backbone_mode_label": "Model asosi",
        "batch_intro_title": "Butun faylni bir vaqtda tahlil qiling",
        "batch_intro_body": (
            "`review_text` nomli ustunga ega CSV yoki Excel faylni yuklang "
            "(`review_id` va `rating` ustunlari ixtiyoriy, lekin tavsiya etiladi). "
            "Har bir qator bitta sharh kabi tozalanadi, baholanadi va yo'naltiriladi, "
            "natijalarni to'liq CSV shaklida yuklab olishingiz mumkin."
        ),
        "upload_label": "CSV yoki Excel faylni yuklang",
        "file_preview_label": "📋 Yuklangan fayl ko'rinishi",
        "process_batch_btn": "🚀 Ommaviy tahlilni boshlash",
        "no_file_info": "ℹ️ Hali fayl yuklanmagan. Ommaviy tahlil uchun CSV yoki Excel faylini yuklang.",
        "missing_column_error": "⚠️ Yuklangan faylda `review_text` ustuni bo'lishi shart.",
        "batch_done_msg": "Ommaviy tahlil yakunlandi.",
        "download_results_btn": "⬇️ Natijalarni CSV sifatida yuklab olish",
        "results_table_label": "Natijalar",
        "summary_source_label": "Hisobot manbasi",
        "summary_source_batch": "Oxirgi ommaviy yuklashimni ishlatish",
        "summary_source_demo": "Demo ma'lumotlar to'plamidan foydalanish",
        "summary_no_data": "Hozircha ma'lumot yo'q — avval ommaviy tahlil qiling yoki demo ma'lumotlarga o'ting.",
        "metric_total_reviews": "Jami sharhlar",
        "metric_auto_processed": "Avtomatik qayta ishlangan",
        "metric_human_review": "Inson tekshiruvi kerak",
        "metric_avg_actionability": "O'rtacha muhimlik darajasi",
        "metric_urgent": "Shoshilinch elementlar",
        "metric_nss": "Sof kayfiyat balli (NSS)",
        "top_issues_label": "Eng ko'p uchraydigan muammolar",
        "sentiment_dist_label": "Kayfiyat taqsimoti",
        "issue_freq_label": "Muammolar chastotasi",
        "model_info_title": "Joriy model konfiguratsiyasi",
        "model_mode_label": "Faol model asosi",
        "model_mode_baseline_desc": (
            "TF-IDF xususiyatlari + Logistic Regression / Ridge ko'p vazifali "
            "boshlar (muammo belgilash, kayfiyat, muhimlik). Ishga tushganda "
            "sintetik ma'lumotlarda o'qitiladi; tez va yengil, bepul hosting uchun ideal."
        ),
        "model_mode_offline_desc": (
            "scikit-learn o'qitish mavjud bo'lmagani sababli faollashtirilgan, "
            "kutubxonalarsiz kalit so'z/qoida asosidagi zaxira model. Aniqlik "
            "pastroq — natijalarni faqat taxminiy deb hisoblang."
        ),
        "model_threshold_label": "Ishonch chegarasi (avtomatik qayta ishlash uchun)",
        "model_training_rows_label": "O'qitishda ishlatilgan qatorlar",
        "model_issue_classes_label": "Muammo toifalari",
        "model_sentiment_classes_label": "Kayfiyat sinflari",
        "sidebar_note": "Til va vizual profil butun ilova bo'ylab qo'llaniladi.",
        "footer_note": "RET-02 · Mahsulot sharhlarini tahlil qilish tizimi · Model matni faqat inglizcha; interfeys uch tilli.",
    },
    "ru": {
        "app_title": "📦 Система интеллектуального анализа отзывов",
        "app_subtitle": "AI-платформа сортировки отзывов для продуктовых, CX и e-commerce команд",
        "language_label": "🌐 Язык",
        "profile_label": "👤 Я...",
        "profile_pm_cx": "Продакт-менеджер / Руководитель CX",
        "profile_data_ml": "Аналитик данных / ML-инженер",
        "profile_exec_biz": "Руководитель / Бизнес",
        "profile_genz_student": "Gen-Z / Студент",
        "tab_single": "📝 Анализ одного отзыва",
        "tab_batch": "📊 Пакетный анализ CSV / Excel",
        "tab_summary": "📈 Итоговый отчёт",
        "tab_model": "⚙️ О модели",
        "single_intro_title": "Проанализируйте один отзыв",
        "single_intro_body": (
            "Введите ниже текст одного отзыва о товаре. Система очищает текст, "
            "определяет тональность, отмечает категории проблем (доставка, брак, "
            "упаковка и т.д.), оценивает важность отзыва для CX-команды и "
            "сообщает, достаточно ли уверен результат для автообработки или "
            "нужна проверка человеком."
        ),
        "review_text_label": "Текст отзыва",
        "review_text_placeholder": "Например: Посылка пришла раздавленной, товар оказался неисправен...",
        "rating_label": "Оценка в звёздах (необязательно)",
        "review_id_label": "ID отзыва (необязательно)",
        "analyze_btn": "🔍 Анализировать отзыв",
        "no_review_warning": "⚠️ Отзыв не введён. Пожалуйста, введите текст отзыва или выберите способ загрузки.",
        "result_header": "Результат анализа",
        "status_auto": "✅ ОБРАБОТАНО АВТОМАТИЧЕСКИ",
        "status_human": "🧑‍💼 ТРЕБУЕТСЯ ПРОВЕРКА ЧЕЛОВЕКОМ",
        "confidence_label": "Общая уверенность",
        "sentiment_label": "Прогнозируемая тональность",
        "issues_label": "Обнаруженные категории проблем",
        "no_issues_detected": "Конкретных проблем не обнаружено",
        "actionability_label": "Оценка важности (actionability)",
        "edge_case_label": "Флаг качества данных",
        "why_prediction_label": "🔎 Почему такой результат?",
        "why_prediction_empty": "Недостаточно сигнала в тексте для извлечения ключевых слов.",
        "why_prediction_intro": "Ключевые слова, повлиявшие на результат:",
        "backbone_mode_label": "Основа модели",
        "batch_intro_title": "Анализ целого файла за раз",
        "batch_intro_body": (
            "Загрузите CSV или Excel файл со столбцом `review_text` (столбцы "
            "`review_id` и `rating` необязательны, но рекомендуются). Каждая "
            "строка обрабатывается так же, как и один отзыв, а результаты можно "
            "скачать в виде CSV."
        ),
        "upload_label": "Загрузить CSV или Excel файл",
        "file_preview_label": "📋 Предпросмотр загруженного файла",
        "process_batch_btn": "🚀 Запустить пакетный анализ",
        "no_file_info": "ℹ️ Файл ещё не загружен. Загрузите CSV или Excel файл для пакетного анализа.",
        "missing_column_error": "⚠️ Загруженный файл должен содержать столбец `review_text`.",
        "batch_done_msg": "Пакетный анализ завершён.",
        "download_results_btn": "⬇️ Скачать результаты в CSV",
        "results_table_label": "Результаты",
        "summary_source_label": "Источник отчёта",
        "summary_source_batch": "Использовать последнюю загрузку",
        "summary_source_demo": "Использовать демо-данные",
        "summary_no_data": "Данных пока нет — сначала выполните пакетный анализ или переключитесь на демо-данные.",
        "metric_total_reviews": "Всего отзывов",
        "metric_auto_processed": "Обработано автоматически",
        "metric_human_review": "Требуется проверка",
        "metric_avg_actionability": "Средняя важность",
        "metric_urgent": "Срочные позиции",
        "metric_nss": "Индекс чистой тональности (NSS)",
        "top_issues_label": "Наиболее частые проблемы",
        "sentiment_dist_label": "Распределение тональности",
        "issue_freq_label": "Частота проблем",
        "model_info_title": "Текущая конфигурация модели",
        "model_mode_label": "Активный режим модели",
        "model_mode_baseline_desc": (
            "Признаки TF-IDF + многозадачные головы Logistic Regression / Ridge "
            "(категории проблем, тональность, важность). Обучается на "
            "синтетических данных при запуске; быстро и легковесно — подходит "
            "для бесплатного хостинга."
        ),
        "model_mode_offline_desc": (
            "Резервная модель на основе ключевых слов/правил без внешних "
            "зависимостей, активна из-за недоступности обучения scikit-learn. "
            "Точность снижена — считайте результаты приблизительными."
        ),
        "model_threshold_label": "Порог уверенности (для автообработки)",
        "model_training_rows_label": "Использовано строк для обучения",
        "model_issue_classes_label": "Категории проблем",
        "model_sentiment_classes_label": "Классы тональности",
        "sidebar_note": "Язык и визуальный профиль применяются ко всему приложению.",
        "footer_note": "RET-02 · Система анализа отзывов о товарах · Текст модели только на английском; интерфейс трёхъязычный.",
    },
}


def t(key: str, lang: str = DEFAULT_LANGUAGE) -> str:
    """Looks up a UI string by key for the given language.

    Args:
        key: Translation key.
        lang: Language code ('en', 'uz', 'ru').

    Returns:
        The translated string. Falls back to the English string, then to the
        key itself, if a translation is missing — so the UI never crashes or
        shows a blank label for an untranslated string.
    """
    lang_map = _TRANSLATIONS.get(lang, _TRANSLATIONS[DEFAULT_LANGUAGE])
    if key in lang_map:
        return lang_map[key]
    return _TRANSLATIONS[DEFAULT_LANGUAGE].get(key, key)
