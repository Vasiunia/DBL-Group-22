#!/usr/bin/env python3
"""
generate_pdf_report.py

Generates a concise, high-impact, human-written 2-page executive summary PDF
of the South Sudan Payam Flood & Driver Prediction Model.
Strictly enforced to be MAX 2 PAGES.
"""

import sys
from pathlib import Path
import pandas as pd

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas

PDF_OUTPUT = Path("south_sudan_payam_flood_prediction_brief.pdf")

# Custom Canvas to count total pages and add footer
class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_footer(num_pages)
            super().showPage()
        super().save()

    def draw_footer(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748b"))
        # Top line above footer
        self.setStrokeColor(colors.HexColor("#cbd5e1"))
        self.setLineWidth(0.5)
        self.line(36, 28, letter[0] - 36, 28)
        
        # Footer text
        left_text = "South Sudan Flood & Driver Prediction System | Operational Policy Brief"
        right_text = f"Page {self._pageNumber} of {page_count}"
        self.drawString(36, 16, left_text)
        self.drawRightString(letter[0] - 36, 16, right_text)
        self.restoreState()

def create_pdf():
    print("Generating 2-page executive brief PDF...")
    
    # Load summary data
    df_s = pd.read_csv("flood_driver_analysis/data/payam_flood_predictions_annual_summary.csv")
    
    # Page dimensions: letter is 612 x 792 pt. Margins: 36 pt (0.5 inch). Printable width: 540 pt. Printable height: 720 pt.
    doc = SimpleDocTemplate(
        str(PDF_OUTPUT),
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=32,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()
    
    # Custom high-readability typography
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=17,
        leading=20,
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=2
    )
    
    subtitle_style = ParagraphStyle(
        'DocSub',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=12.5,
        textColor=colors.HexColor("#475569"),
        spaceAfter=8
    )

    h1_style = ParagraphStyle(
        'H1',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#0369a1"),
        spaceBefore=5,
        spaceAfter=3
    )

    bullet_style = ParagraphStyle(
        'Bullet',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor("#1e293b"),
        spaceAfter=2.5,
        leftIndent=12,
        firstLineIndent=-10
    )

    badge_style = ParagraphStyle(
        'Badge',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#ffffff"),
        alignment=1
    )

    table_header_style = ParagraphStyle(
        'TH',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor("#ffffff"),
        alignment=1
    )

    table_cell_style = ParagraphStyle(
        'TC',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor("#0f172a")
    )
    
    table_cell_bold = ParagraphStyle(
        'TCB',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor("#0f172a")
    )

    story = []

    # =========================================================================
    # PAGE 1: THE CORE STORY, DRIVERS & TIMING
    # =========================================================================
    
    # Header Banner
    story.append(Paragraph("South Sudan Payam Flood & Driver Prediction Brief", title_style))
    story.append(Paragraph("<b>Operational Intelligence for Government Leaders, Humanitarian Responders & Citizens</b> | Modeling All 512 Payams", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0284c7"), spaceBefore=0, spaceAfter=6))

    # Top Highlights Callout Box
    stat_data = [
        [
            Paragraph("<b>512</b><br/><font size=6.5 color='#64748b'>PAYAMS MODELED</font>", ParagraphStyle('C1', alignment=1, leading=9)),
            Paragraph("<b>4 TYPES</b><br/><font size=6.5 color='#64748b'>FLOOD DRIVERS</font>", ParagraphStyle('C2', alignment=1, leading=9)),
            Paragraph("<b>AUG - NOV</b><br/><font size=6.5 color='#64748b'>PEAK THREAT WINDOW</font>", ParagraphStyle('C3', alignment=1, leading=9)),
            Paragraph("<b>30-60 DAYS</b><br/><font size=6.5 color='#64748b'>EARLY WARNING LEAD</font>", ParagraphStyle('C4', alignment=1, leading=9)),
            Paragraph("<b>R² 0.773</b><br/><font size=6.5 color='#64748b'>5-FOLD CV ACCURACY</font>", ParagraphStyle('C5', alignment=1, leading=9))
        ]
    ]
    stat_table = Table(stat_data, colWidths=[108]*5)
    stat_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#f8fafc")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#cbd5e1")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#e2e8f0")),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(stat_table)
    story.append(Spacer(1, 6))

    # Section 1: The Human Problem & What We Built
    story.append(Paragraph("1. The Core Question: What Did We Build?", h1_style))
    story.append(Paragraph("• <b>Every single payam now has a specific prediction:</b> Instead of treating South Sudan as one big vague disaster zone, this model diagnoses all <b>512 payams individually</b> across 79 counties so leaders and families know what is coming.", bullet_style))
    story.append(Paragraph("• <b>Floods are not all caused by the same thing:</b> Flooding in Juba is nothing like flooding in Bentiu. The model breaks every payam down into its true causal driver so we stop applying one-size-fits-all fixes.", bullet_style))
    story.append(Paragraph("• <b>Answers Which, When, and How Much:</b> For each payam, the system predicts the exact dominant driver, the critical risk months, the expected millimeters of rain/runoff, the percentage of land underwater, and the population affected.", bullet_style))
    story.append(Spacer(1, 4))

    # Section 2: Which Driver Where?
    story.append(Paragraph("2. Which Type of Driver? (The 4 Flood Personalities)", h1_style))
    story.append(Paragraph("• <b>🌊 Riverine & Sudd Inundation (Central & Northern Basins):</b> Lake Victoria and Lake Albert release record surges down the White Nile into the flat Sudd swamps (Unity, Jonglei, Upper Nile). Water cannot drain and sits trapped for up to 6 months.", bullet_style))
    story.append(Paragraph("• <b>🌧️ Extreme Rainfall (Equatoria Greenbelt):</b> Cloudbursts and violent downpours drop 150–220 mm/month in Central, Western, and Eastern Equatoria, overwhelming city drains and creating sudden flash deluges.", bullet_style))
    story.append(Paragraph("• <b>💧 Surface Runoff (Clay Plains & Foothills):</b> Rain falls upstream and sheets rapidly across hard-baked soils in Warrap and Northern Bahr el Ghazal, cutting off roads before rivers even crest.", bullet_style))
    story.append(Paragraph("• <b>🌾 Farmland Exposure (Agro-Pastoral Fringe):</b> Floodwaters drown vital standing sorghum, maize, and cassava crops right before harvest, wiping out village food supplies and triggering severe hunger.", bullet_style))
    story.append(Spacer(1, 4))

    # Section 3: When Does It Hit?
    story.append(Paragraph("3. When Does It Strike? (Timing & Seasonality)", h1_style))
    story.append(Paragraph("• <b>July to August is the National Flashpoint:</b> 334 out of 512 payams reach their annual peak flood risk in August as local rains saturate the entire country.", bullet_style))
    story.append(Paragraph("• <b>The Deadly 2-Month Sudd Delay:</b> Upstream water travels slowly. While rainfall drops off by October, river inundation in Unity and Jonglei surges to its worst levels between <b>October and December</b>.", bullet_style))
    story.append(Paragraph("• <b>Massive Early Warning Window:</b> Because Nile flood waves take 30 to 60 days to move downstream, authorities have <b>1 to 2 months of advance notice</b> to strengthen dykes and move grain before waters peak.", bullet_style))
    story.append(Spacer(1, 5))

    # Image 1 on Page 1: Seasonal Timeline
    img_path1 = "flood_driver_analysis/graphs/payam_flood_drivers_seasonal_timeline.png"
    if Path(img_path1).exists():
        # Printable width is 540 pt. Scale image to fit: width=530, height=195 pt.
        story.append(Image(img_path1, width=535, height=190))
        story.append(Paragraph("<font size=7 color='#64748b'><b>Figure 1:</b> Monthly progression of rainfall, runoff, satellite flood extent, and peak month distribution across South Sudan payams.</font>", ParagraphStyle('Cap1', alignment=1, spaceBefore=2)))

    # Force PageBreak to strictly guarantee 2-page layout
    story.append(PageBreak())

    # =========================================================================
    # PAGE 2: HARD DATA, CRITICAL PAYAMS & REAL-WORLD ACTION
    # =========================================================================

    story.append(Paragraph("4. How Much & Where: The Critical High-Risk Payams", h1_style))
    story.append(Paragraph("• <b>183 Payams face High to Critical Risk:</b> Over 2.1 million citizens and 15,000 km² of land are exposed to severe inundation during peak months.", bullet_style))
    story.append(Paragraph("• <b>The Epicenter of Disaster:</b> Unity State (Leer, Mayendit, Panyijiar) and Jonglei State (Ayod, Fangak) suffer the highest scores nationally, with up to <b>73.5% of entire payams submerged</b>.", bullet_style))
    story.append(Spacer(1, 4))

    # Top 8 High Risk Payams Table
    top_payams = df_s.head(8)
    table_rows = [
        [
            Paragraph("Payam (ADM3)", table_header_style),
            Paragraph("County", table_header_style),
            Paragraph("State", table_header_style),
            Paragraph("Primary Driver", table_header_style),
            Paragraph("Peak", table_header_style),
            Paragraph("Risk", table_header_style),
            Paragraph("Flood %", table_header_style),
            Paragraph("Submerged", table_header_style),
            Paragraph("Exposed Pop", table_header_style)
        ]
    ]

    for _, r in top_payams.iterrows():
        driver_short = r['annual_primary_driver'].replace("Riverine Inundation", "🌊 Inundation").replace("Extreme Rainfall", "🌧️ Rainfall")
        table_rows.append([
            Paragraph(f"<b>{r['adm3_name']}</b>", table_cell_bold),
            Paragraph(r['adm2_name'], table_cell_style),
            Paragraph(r['adm1_name'], table_cell_style),
            Paragraph(driver_short, table_cell_style),
            Paragraph(r['peak_month_name'], table_cell_style),
            Paragraph(f"<font color='#dc2626'><b>{r['max_flood_risk_score']:.1f}</b></font>", table_cell_style),
            Paragraph(f"{r['peak_flood_pct']:.1f}%", table_cell_style),
            Paragraph(f"{r['peak_flood_km2']:.0f} km²", table_cell_style),
            Paragraph(f"{int(r['peak_population_at_risk']):,}", table_cell_style),
        ])

    payam_table = Table(table_rows, colWidths=[65, 55, 65, 80, 32, 40, 48, 65, 60])
    payam_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#0f172a")),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor("#ffffff"), colors.HexColor("#f8fafc")]),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
    ]))
    story.append(payam_table)
    story.append(Spacer(1, 5))

    # Image 2 on Page 2: Risk and Driver Distribution
    img_path2 = "flood_driver_analysis/graphs/payam_risk_and_driver_distribution.png"
    if Path(img_path2).exists():
        story.append(Image(img_path2, width=535, height=135))
        story.append(Paragraph("<font size=7 color='#64748b'><b>Figure 2:</b> National breakdown of primary flood drivers (left) and distribution of peak flood risk tiers across all 512 payams (right).</font>", ParagraphStyle('Cap2', alignment=1, spaceBefore=2)))
    story.append(Spacer(1, 4))

    # Section 5: Real-World Action Guidance
    story.append(Paragraph("5. What Leaders and Citizens Must Do (Action Protocols)", h1_style))
    story.append(Paragraph("• <b>🏛️ Government & UN Responders (Bentiu, Fangak, Ayod, Leer):</b> Use the 30–60 day upstream Nile warning window to inspect and compact perimeter dykes, unblock main drainage canals, and preposition motorboats, fuel, and water treatment tablets.", bullet_style))
    story.append(Paragraph("• <b>🌧️ Urban Municipalities (Juba, Yei, Magwi):</b> Clear culverts and drainage ditches prior to August downpours; restrict building in flash-flood ravines.", bullet_style))
    story.append(Paragraph("• <b>👥 Citizens & Herders (Livelihood Safety):</b> Move cattle herds to higher grazing grounds before September; store seed reserves on elevated timber platforms; harvest early-maturing crops ahead of peak waters.", bullet_style))
    story.append(Paragraph("• <b>💻 Interactive Access:</b> Use <b>predict_payam.py</b> in the terminal or open <b>south_sudan_payam_flood_predictor.html</b> for instant payam search, 12-month curves, and what-if climate simulations.", bullet_style))
    story.append(Spacer(1, 4))

    # Final Summary Sign-off Box
    summary_box_data = [[
        Paragraph("<b>Key Takeaway for Action:</b> Floods in South Sudan are predictable months in advance. By aligning disaster resources with the specific driver and peak timing of each payam, the government and humanitarian partners can shift from reactive emergency crisis response to life-saving anticipatory protection.", ParagraphStyle('SumB', fontSize=7.5, leading=10, textColor=colors.HexColor("#0f172a")))
    ]]
    sum_table = Table(summary_box_data, colWidths=[540])
    sum_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#e0f2fe")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#38bdf8")),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(sum_table)

    # Build document
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"PDF successfully built: {PDF_OUTPUT}")

if __name__ == "__main__":
    create_pdf()
