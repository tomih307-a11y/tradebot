# تشغيل وحدة توصيات القناة على جهاز ويندوز (وضع ورقي — بلا أوامر حقيقية)

## 1) التثبيت (مرة واحدة)
1. نزّل Python من https://www.python.org/downloads/ وأثناء التثبيت **علّم على «Add python.exe to PATH»**.
2. نزّل المشروع: https://github.com/tomih307-a11y/tradebot/archive/refs/heads/claude/pmo-zero-crossing-strategy-tzs5l3.zip ثم فك الضغط.
3. افتح مجلد المشروع، واكتب في شريط العنوان `cmd` ثم Enter. في النافذة السوداء:
   ```
   pip install -r requirements.txt
   ```

## 2) الإعداد
افتح `config.toml` بالمفكرة (Notepad) وابحث عن `[telegram_signals]`، ثم:
```
channel = "اسم_القناة"        # الجزء بعد t.me/  (القناة العامة فقط)
```
الباقي مضبوط: هامش 10% من الرصيد، صفقتان كحد أقصى، رصيد ورقي 25$، وقف بقاعدة القناة (إغلاق شمعة 4 ساعات).

## 3) التشغيل (ورقي)
```
python -m tradebot --strategy telegram_signals live
```
- لا يفتح صفقات من الرسائل القديمة؛ يبدأ من آخر رسالة.
- كل دخول/هدف/إغلاق يظهر في النافذة (ويصلك على Telegram إن ضبطت `TELEGRAM_BOT_TOKEN` و`TELEGRAM_CHAT_ID`).
- سجل الصفقات المغلقة: `reports\telegram_signals_trades.csv`.
- يحتاج أن يبقى الجهاز والإنترنت شغالين. أوقف الشاشة فقط، وعطّل السكون.

## 4) اختبار تاريخ القناة على أسعار MEXC (الأهم قبل أي مال)
1. على **Telegram Desktop**: افتح القناة ← ⋮ ← **Export chat history** ← صيغة **JSON** ← نزّل بدون وسائط.
2. أرسل لي الملف `result.json` (أو شغّل بنفسك):
   ```
   python -m tradebot --strategy telegram_signals backtest --history result.json --delay 60
   ```
   `--delay` = ثواني بين نشر التوصية ودخولك (جرّب 30 و60 و180).
3. النتيجة: `reports\telegram\signals_report.md` و`signals_backtest.csv`.
