import json
from datetime import datetime, timedelta, timezone
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.formatting.rule import ColorScaleRule, CellIsRule
from openpyxl.utils import get_column_letter

SC = "radar_peaks.txt"   # output of alert_peaks.py (scan-results branch)
txt = open(SC, encoding="utf-8").read()
data = json.loads([l for l in txt.splitlines() if l.startswith("JSON:")][0][5:])
now_line = txt.splitlines()[0].split("حتى")[1].strip()
UTC4 = timedelta(hours=4)
dt4 = lambda ts: (datetime.fromtimestamp(ts, timezone.utc) + UTC4).replace(tzinfo=None)
data.sort(key=lambda r: (r["sym"], r["a0"]))
coins = sorted({r["sym"] for r in data})

F = "Arial"
hdr_fill = PatternFill("solid", start_color="1F3864")
hdr_font = Font(name=F, bold=True, color="FFFFFF", size=11)
thin = Side(style="thin", color="BFBFBF")
border = Border(left=thin, right=thin, top=thin, bottom=thin)
center = Alignment(horizontal="center", vertical="center", wrap_text=True)
inp_font = Font(name=F, color="0000FF", size=11)       # hard-coded inputs (blue)
calc_font = Font(name=F, color="000000", size=11)      # formulas (black)

wb = Workbook()

# ---------------- Sheet 1: guide ----------------
g = wb.active
g.title = "اقرأني أولاً"
g.sheet_view.rightToLeft = True
g.column_dimensions["A"].width = 34
g.column_dimensions["B"].width = 95
g["A1"] = "نتائج تنبيهات رادار الانفجار — كيف أقرأ هذا الملف؟"
g["A1"].font = Font(name=F, bold=True, size=15)
g.merge_cells("A1:B1")
rows = [
    ("الفكرة", "كل تنبيه وصلك من الرادار له سعر مكتوب (سعر التنبيه). هذا الملف يقيس: ما أعلى سعر وصلته العملة بعد ذلك؟ وكم ارتفعت بالنسبة المئوية؟ وكم استغرقت للوصول إليه؟"),
    ("الورقة 1: ملخص العملات", "صف واحد لكل عملة. يعتمد على **أول** تنبيه وصلك لهذه العملة. هذه أسهل ورقة للقراءة."),
    ("الورقة 2: كل التنبيهات", "صف لكل تنبيه (قد يكون للعملة الواحدة عدة تنبيهات)، بالتفصيل الكامل."),
    ("نسبة أعلى ارتفاع", "المعادلة: (أعلى سعر ÷ سعر التنبيه) − 1. مثال: +100% تعني أن العملة تضاعفت. 0% أو أقل تعني أنها لم ترتفع فوق سعر التنبيه أبداً."),
    ("المدة حتى أعلى سعر", "الوقت من لحظة التنبيه إلى لحظة وصول العملة لأعلى سعر لها. مكتوب بثلاث طرق: أيام بالكسر (مثلاً 15.3)، وأيام كاملة + ساعات (مثلاً 15 يوماً + 8 ساعات)، وتصنيف بالكلمات (أقل من ساعة / خلال يوم / 1–3 أيام / 3–7 أيام / أكثر من أسبوع)."),
    ("التغير الآن", "المعادلة: (السعر الآن ÷ سعر التنبيه) − 1. سالب يعني أن العملة الآن أرخص من سعر التنبيه، أي أن الارتفاع لم يستمر."),
    ("الألوان", "الأخضر: ارتفاع كبير. الأحمر: لا ارتفاع أو نزول. الخط الأزرق: أرقام مُدخلة من السوق. الخط الأسود: أرقام تُحسب بمعادلات."),
    ("التوقيت", "كل الأوقات بتوقيت UTC+4 (رادارك يعمل بتوقيت UTC+3، فكل وقت هنا أكثر من وقت الرادار بساعة)."),
    ("مصدر الأسعار", "شموع 15 دقيقة لعقود MEXC الآجلة (USDT). أعلى سعر = أعلى قمة شمعة بعد التنبيه حتى " + now_line + "."),
    ("كيف حُدّدت التواريخ", "رسائل الرادار بلا تاريخ، فطابقت كل تنبيه مع شمعة ساعة على MEXC بنفس سعر الإغلاق وحركة الساعة والأربع ساعات."),
    ("لم تُقَس (8 تنبيهات)", "HOOKED (5) وEPIC (2) وDARK (1): لم أجد شمعة تطابق أسعارها على MEXC، فلم تدخل الجداول."),
    ("ملاحظة على CLAUDIA", "أعلى سعر لها كان اليوم، أي أنها ما زالت ترتفع، فقد تزيد المدة ونسبة الارتفاع عند التحديث التالي."),
    ("تنبيه مهم", "أعلى سعر يُعرف بعد وقوعه فقط. لا أحد يعرف وقت التنبيه أي العملات ستصل لقمة عالية. هذا الملف وصف لما حدث، وليس توصية."),
]
for i, (a, b) in enumerate(rows, start=3):
    g.cell(row=i, column=1, value=a).font = Font(name=F, bold=True, size=11)
    c = g.cell(row=i, column=2, value=b.replace("**", ""))
    c.font = Font(name=F, size=11)
    c.alignment = Alignment(wrap_text=True, vertical="top")
    g.cell(row=i, column=1).alignment = Alignment(wrap_text=True, vertical="top")
    g.row_dimensions[i].height = 48

# ---------------- Sheet 3: all alerts ----------------
d = wb.create_sheet("كل التنبيهات")
d.sheet_view.rightToLeft = True
heads = ["العملة", "وقت التنبيه (UTC+4)", "سعر التنبيه", "أعلى سعر وصلته بعد التنبيه", "نسبة أعلى ارتفاع", "وقت الوصول لأعلى سعر (UTC+4)",
         "المدة حتى أعلى سعر (أيام بالكسر)", "المدة حتى أعلى سعر (ساعات)", "أيام كاملة", "+ ساعات", "تصنيف المدة", "السعر الآن", "التغير الآن عن سعر التنبيه"]
for j, h in enumerate(heads, start=1):
    c = d.cell(row=1, column=j, value=h)
    c.font, c.fill, c.alignment, c.border = hdr_font, hdr_fill, center, border
d.row_dimensions[1].height = 48
for i, r in enumerate(data, start=2):
    d.cell(row=i, column=1, value=r["sym"].replace("_USDT", "")).font = Font(name=F, bold=True, size=11)
    d.cell(row=i, column=2, value=dt4(r["a0"])).number_format = "dd-mmm-yyyy hh:mm"
    d.cell(row=i, column=3, value=r["price"]).number_format = "0.000000"
    d.cell(row=i, column=4, value=r["peak"]).number_format = "0.000000"
    d.cell(row=i, column=5, value=f"=D{i}/C{i}-1").number_format = '+0.0%;-0.0%;0.0%'
    d.cell(row=i, column=6, value=dt4(r["peak_t"])).number_format = "dd-mmm-yyyy hh:mm"
    d.cell(row=i, column=7, value=f"=MAX(0,F{i}-B{i})").number_format = "0.0"
    d.cell(row=i, column=8, value=f"=G{i}*24").number_format = "0.0"
    d.cell(row=i, column=9, value=f"=INT(G{i})").number_format = "0"
    d.cell(row=i, column=10, value=f"=ROUND(MOD(G{i},1)*24,0)").number_format = "0"
    d.cell(row=i, column=11, value=f'=IF(G{i}<1/24,"أقل من ساعة",IF(G{i}<1,"خلال يوم",IF(G{i}<3,"من 1 إلى 3 أيام",IF(G{i}<7,"من 3 إلى 7 أيام","أكثر من أسبوع"))))')
    d.cell(row=i, column=12, value=r["now"]).number_format = "0.000000"
    d.cell(row=i, column=13, value=f"=L{i}/C{i}-1").number_format = '+0.0%;-0.0%;0.0%'
    for j in range(1, 14):
        c = d.cell(row=i, column=j)
        c.border = border
        c.alignment = Alignment(horizontal="center", vertical="center")
        if j in (2, 3, 4, 6, 12):
            c.font = inp_font
        elif j > 1:
            c.font = calc_font
last = len(data) + 1
for j, w in zip(range(1, 14), (12, 22, 14, 18, 16, 24, 18, 16, 12, 12, 18, 14, 18)):
    d.column_dimensions[get_column_letter(j)].width = w
d.freeze_panes = "B2"
d.auto_filter.ref = f"A1:M{last}"
d.conditional_formatting.add(f"E2:E{last}", ColorScaleRule(start_type="num", start_value=0, start_color="FFFFFF", mid_type="num", mid_value=1, mid_color="C6E0B4", end_type="num", end_value=3, end_color="63BE7B"))
d.conditional_formatting.add(f"E2:E{last}", CellIsRule(operator="lessThanOrEqual", formula=["0"], fill=PatternFill("solid", start_color="F8CBAD")))
d.conditional_formatting.add(f"M2:M{last}", CellIsRule(operator="lessThan", formula=["0"], font=Font(name=F, color="C00000")))
d.conditional_formatting.add(f"M2:M{last}", CellIsRule(operator="greaterThan", formula=["0"], font=Font(name=F, color="00803C", bold=True)))

# ---------------- Sheet 2: coin summary ----------------
s = wb.create_sheet("ملخص العملات", 1)
s.sheet_view.rightToLeft = True
sh = ["العملة", "عدد التنبيهات", "سعر أول تنبيه", "أعلى سعر وصلته بعده", "نسبة أعلى ارتفاع", "وقت الوصول لأعلى سعر (UTC+4)", "المدة حتى أعلى سعر (أيام بالكسر)", "أيام كاملة", "+ ساعات", "تصنيف المدة", "السعر الآن", "التغير الآن عن سعر التنبيه"]
for j, h in enumerate(sh, start=1):
    c = s.cell(row=1, column=j, value=h)
    c.font, c.fill, c.alignment, c.border = hdr_font, hdr_fill, center, border
s.row_dimensions[1].height = 48
D = "'كل التنبيهات'"
rng = lambda col: f"{D}!${col}$2:${col}${last}"
for i, coin in enumerate(coins, start=2):
    name = coin.replace("_USDT", "")
    s.cell(row=i, column=1, value=name).font = Font(name=F, bold=True, size=11)
    s.cell(row=i, column=2, value=f"=COUNTIF({rng('A')},A{i})")
    for col, src, fmt in ((3, "C", "0.000000"), (4, "D", "0.000000"), (6, "F", "dd-mmm-yyyy hh:mm"), (7, "G", "0.0"), (8, "I", "0"), (9, "J", "0"), (10, "K", None), (11, "L", "0.000000")):
        c = s.cell(row=i, column=col, value=f"=INDEX({rng(src)},MATCH($A{i},{rng('A')},0))")
        if fmt:
            c.number_format = fmt
    s.cell(row=i, column=5, value=f"=D{i}/C{i}-1").number_format = '+0.0%;-0.0%;0.0%'
    s.cell(row=i, column=12, value=f"=K{i}/C{i}-1").number_format = '+0.0%;-0.0%;0.0%'
    for j in range(1, 13):
        c = s.cell(row=i, column=j)
        c.border = border
        c.alignment = Alignment(horizontal="center", vertical="center")
        if j > 1:
            c.font = calc_font
slast = len(coins) + 1
for j, w in zip(range(1, 13), (12, 12, 16, 18, 16, 24, 18, 12, 12, 18, 14, 18)):
    s.column_dimensions[get_column_letter(j)].width = w
s.freeze_panes = "B2"
s.auto_filter.ref = f"A1:L{slast}"
s.conditional_formatting.add(f"E2:E{slast}", ColorScaleRule(start_type="num", start_value=0, start_color="FFFFFF", mid_type="num", mid_value=1, mid_color="C6E0B4", end_type="num", end_value=3, end_color="63BE7B"))
s.conditional_formatting.add(f"E2:E{slast}", CellIsRule(operator="lessThanOrEqual", formula=["0"], fill=PatternFill("solid", start_color="F8CBAD")))
s.conditional_formatting.add(f"L2:L{slast}", CellIsRule(operator="lessThan", formula=["0"], font=Font(name=F, color="C00000")))
s.conditional_formatting.add(f"L2:L{slast}", CellIsRule(operator="greaterThan", formula=["0"], font=Font(name=F, color="00803C", bold=True)))
n = slast + 2
s.cell(row=n, column=1, value="ملاحظة: هذه الورقة تعتمد على أول تنبيه لكل عملة. المعادلات تسحب القيم من ورقة «كل التنبيهات». الأوقات بتوقيت UTC+4. الأسعار من شموع 15 دقيقة على MEXC حتى " + now_line + ".").font = Font(name=F, italic=True, size=10)
s.merge_cells(start_row=n, start_column=1, end_row=n, end_column=12)
wb.move_sheet("ملخص العملات", offset=-1) if wb.sheetnames.index("ملخص العملات") > 0 else None
wb._sheets = [wb["ملخص العملات"], wb["كل التنبيهات"], wb["اقرأني أولاً"]]
from openpyxl.worksheet.properties import PageSetupProperties
for ws in wb.worksheets:
    ws.page_setup.orientation = "landscape"
    ws.page_setup.fitToWidth = 1
    ws.page_setup.fitToHeight = 0
    ws.sheet_properties.pageSetUpPr = PageSetupProperties(fitToPage=True)
sx = wb["ملخص العملات"]
sx.cell(row=slast + 2, column=1).alignment = Alignment(wrap_text=True, vertical="top")
sx.row_dimensions[slast + 2].height = 45
wb.active = 0
wb.save("/home/user/tradebot/radar_results.xlsx")
print(wb.sheetnames, len(coins), "coins", len(data), "alerts")
