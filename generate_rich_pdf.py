import os
import sys

from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

def create_rich_presentation_pdf(output_filename="AirPulse_Presentation_Deck.pdf"):
    # 11 x 8.5 inches landscape presentation slides
    doc = SimpleDocTemplate(
        output_filename,
        pagesize=landscape(letter),
        rightMargin=28,
        leftMargin=28,
        topMargin=28,
        bottomMargin=28
    )

    styles = getSampleStyleSheet()

    # Rich Palette
    NAVY_DARK = colors.HexColor("#0f172a")
    CARD_BG = colors.HexColor("#1e293b")
    SKY_BLUE = colors.HexColor("#38bdf8")
    PURPLE_ACCENT = colors.HexColor("#818cf8")
    TEXT_LIGHT = colors.HexColor("#f8fafc")
    TEXT_MUTED = colors.HexColor("#94a3b8")
    ACCENT_GREEN = colors.HexColor("#22c55e")
    ACCENT_RED = colors.HexColor("#ef4444")
    ACCENT_AMBER = colors.HexColor("#f59e0b")

    # Custom Typography
    title_cover_style = ParagraphStyle(
        'CoverTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=26,
        leading=32,
        textColor=SKY_BLUE,
        alignment=TA_LEFT,
        spaceAfter=8
    )

    subtitle_cover_style = ParagraphStyle(
        'CoverSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=14,
        leading=18,
        textColor=TEXT_LIGHT,
        alignment=TA_LEFT,
        spaceAfter=15
    )

    slide_title_style = ParagraphStyle(
        'SlideTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=SKY_BLUE,
        spaceAfter=10
    )

    body_light_style = ParagraphStyle(
        'BodyLight',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10.5,
        leading=15,
        textColor=TEXT_LIGHT,
        spaceAfter=6
    )

    script_card_style = ParagraphStyle(
        'SpeakerScript',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor("#e2e8f0"),
        backColor=colors.HexColor("#1e293b"),
        borderColor=SKY_BLUE,
        borderWidth=1,
        borderPadding=10,
        spaceBefore=8,
        spaceAfter=5
    )

    tbl_header_style = ParagraphStyle(
        'TblHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9.5,
        leading=12,
        textColor=colors.white
    )

    tbl_cell_style = ParagraphStyle(
        'TblCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11.5,
        textColor=TEXT_LIGHT
    )

    story = []

    # ==================== SLIDE 1: COVER SLIDE ====================
    story.append(Spacer(1, 15))
    story.append(Paragraph("🌬️ AirPulse: Real-Time Air Quality Intelligence & MLOps System", title_cover_style))
    story.append(Paragraph("End-to-End Production ML Platform for India AQI Forecasting & Health Risk Management", subtitle_cover_style))
    story.append(HRFlowable(width="100%", thickness=2.5, color=SKY_BLUE, spaceAfter=15))

    cover_box_content = """
    <b>Presenter:</b> Riya Singh &nbsp;|&nbsp; <b>Corpus:</b> riyaaa04/AirPulse-MLOps<br/>
    <b>Live GitHub Repository:</b> https://github.com/riyaaa04/AirPulse-MLOps<br/>
    <b>Architecture Stack:</b> DVC + MLflow + FastAPI REST + Streamlit UI + Docker + GitHub Actions CI/CD<br/>
    <b>Promoted Model Performance:</b> GradientBoosting Regressor (R² Score: <b>0.9339</b> | RMSE: <b>34.80</b>)
    """
    story.append(Paragraph(cover_box_content, body_light_style))
    story.append(Spacer(1, 20))
    story.append(Paragraph("<b>🔊 Speaker Opening (30 sec):</b> <i>'Good morning/afternoon everyone. Today I present AirPulse — an enterprise-grade Air Quality Index (AQI) forecasting and public health intelligence platform. Instead of presenting a static notebook score, I am presenting a live production-ready system backed by real REST APIs, MLflow experiment tracking, DVC pipeline versioning, and actionable public health advisories.'</i>", script_card_style))
    story.append(PageBreak())

    # ==================== SLIDE 2: THE PROBLEM ====================
    story.append(Paragraph("📌 Slide 2: The Problem — Why Air Quality Forecasting Matters", slide_title_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=PURPLE_ACCENT, spaceAfter=12))

    prob_text = """
    • <b>Public Health Hazard:</b> Over 1.6 million premature deaths in India annually linked to severe PM2.5 and PM10 particulate exposure.<br/>
    • <b>The Operational Gap:</b> Citizens and city planners lack real-time predictive tools to know tomorrow's AQI before stepping outside or organizing public events.<br/>
    • <b>The Data Challenge:</b> Environmental sensor logs suffer from missing readings due to maintenance, extreme right-skewed pollutant spikes, and complex chemical interactions (NO2, SO2, CO, O3).
    """
    story.append(Paragraph(prob_text, body_light_style))
    story.append(Spacer(1, 15))
    story.append(Paragraph("<b>🔊 Speaker Script:</b> <i>'Air pollution in major Indian urban centers like Delhi and Mumbai is a severe health hazard. Current public apps only report yesterday's numbers. What citizens and city planners need is a predictive tool — one that warns sensitive groups, calculates health severity buckets, and gives actionable advice before AQI crosses severe levels.'</i>", script_card_style))
    story.append(PageBreak())

    # ==================== SLIDE 3: ARCHITECTURE DIAGRAM ====================
    story.append(Paragraph("📌 Slide 3: End-to-End MLOps System Architecture", slide_title_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=SKY_BLUE, spaceAfter=10))

    if os.path.exists("presentation_assets/architecture_diagram.png"):
        story.append(Image("presentation_assets/architecture_diagram.png", width=700, height=350))
    story.append(Spacer(1, 10))
    story.append(Paragraph("<b>🔊 Speaker Script:</b> <i>'Here is our system architecture: Raw data is versioned with DVC. Our feature engineering pipeline transforms pollutant inputs, which are evaluated across algorithms using MLflow. The winning model is served via a FastAPI REST API and visualized through a modern Streamlit dashboard containerized with Docker.'</i>", script_card_style))
    story.append(PageBreak())

    # ==================== SLIDE 4: FEATURE ENGINEERING TABLE ====================
    story.append(Paragraph("📌 Slide 4: Feature Engineering Rationale (Syllabus Phases 1–5)", slide_title_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=SKY_BLUE, spaceAfter=10))

    fe_table_data = [
        [Paragraph("<b>Syllabus Phase</b>", tbl_header_style), Paragraph("<b>Technique Applied</b>", tbl_header_style), Paragraph("<b>Domain Rationale & Evidence</b>", tbl_header_style)],
        [Paragraph("Phase 1: Foundations", tbl_cell_style), Paragraph("scikit-learn ColumnTransformer", tbl_cell_style), Paragraph("Prevents data leakage between training and validation splits.", tbl_cell_style)],
        [Paragraph("Phase 2: Cleaning & Prep", tbl_cell_style), Paragraph("Forward-fill + City Median, Log1p, RobustScaler", tbl_cell_style), Paragraph("Sensor downtime is MAR missingness; Log1p handles heavy right-skew; RobustScaler handles extreme winter smog spikes.", tbl_cell_style)],
        [Paragraph("Phase 3: Feature Creation", tbl_cell_style), Paragraph("Lags (1 & 3 days), 7-Day Rolling Stats, Sin/Cos Time", tbl_cell_style), Paragraph("PM2.5(t-1) & 7-day rolling averages capture pollutant momentum and multi-day weather accumulation.", tbl_cell_style)],
        [Paragraph("Phase 4: Feature Selection", tbl_cell_style), Paragraph("Variance Threshold & Mutual Information", tbl_cell_style), Paragraph("Filters zero-variance features and ranks primary pollutant drivers (PM2.5, PM10, NO2).", tbl_cell_style)],
        [Paragraph("Phase 5: Dimensionality", tbl_cell_style), Paragraph("PCA Scree Plot & Cumulative Variance", tbl_cell_style), Paragraph("Compresses multi-pollutant metrics into core air quality components while retaining ~95% variance.", tbl_cell_style)]
    ]
    t_fe_rich = Table(fe_table_data, colWidths=[130, 170, 430])
    t_fe_rich.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#0284c7")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#334155")),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [CARD_BG, NAVY_DARK])
    ]))
    story.append(t_fe_rich)
    story.append(Spacer(1, 10))
    story.append(Paragraph("<b>🔊 Speaker Script:</b> <i>'Every feature engineering choice in AirPulse was evidence-based: Lags and rolling averages captured time continuity, Log1p handled skewed concentrations, and RobustScaler ensured extreme smog spikes didn't distort model gradients.'</i>", script_card_style))
    story.append(PageBreak())

    # ==================== SLIDE 5: PCA SCREE PLOT ====================
    story.append(Paragraph("📌 Slide 5: Dimensionality Reduction & PCA Scree Plot", slide_title_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=PURPLE_ACCENT, spaceAfter=10))

    if os.path.exists("presentation_assets/pca_chart.png"):
        story.append(Image("presentation_assets/pca_chart.png", width=680, height=340))
    story.append(Spacer(1, 10))
    story.append(Paragraph("<b>🔊 Speaker Script:</b> <i>'To handle multi-collinearity among chemical pollutants, we used PCA to compress feature dimensions. As seen in our Scree Plot, 5 principal components capture 95% of the total cumulative variance, dramatically reducing computational complexity.'</i>", script_card_style))
    story.append(PageBreak())

    # ==================== SLIDE 6: MLFLOW BENCHMARK & RESIDUAL PLOT ====================
    story.append(Paragraph("📌 Slide 6: MLflow Experiment Tracking & Residual Plot", slide_title_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=SKY_BLUE, spaceAfter=10))

    img_col1 = "presentation_assets/model_benchmark.png"
    img_col2 = "models/residual_GradientBoosting.png"

    if os.path.exists(img_col1) and os.path.exists(img_col2):
        t_imgs = Table([
            [Image(img_col1, width=350, height=240), Image(img_col2, width=350, height=240)]
        ], colWidths=[360, 360])
        t_imgs.setStyle(TableStyle([
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('ALIGN', (0,0), (-1,-1), 'CENTER')
        ]))
        story.append(t_imgs)

    story.append(Spacer(1, 10))
    story.append(Paragraph("<b>🔊 Speaker Script:</b> <i>'Using MLflow, we tracked hyper-parameters, metrics, and residual plots across runs. Baseline Ridge scored 0.88 R². GradientBoosting achieved 0.9339 R² with an RMSE of 34.8, promoting GradientBoosting to production.'</i>", script_card_style))
    story.append(PageBreak())

    # ==================== SLIDE 7: LIVE PROTOTYPE DEMO ====================
    story.append(Paragraph("📌 Slide 7: Mandatory Live Working Prototype Demo", slide_title_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=SKY_BLUE, spaceAfter=12))

    demo_info = """
    <b>Live Prototype Demonstration Steps:</b><br/>
    1. <b>Launch App:</b> Open Streamlit dashboard UI (http://localhost:8501 or live cloud link).<br/>
    2. <b>Select City:</b> Choose <i>Delhi</i> from the sidebar dropdown.<br/>
    3. <b>Load Preset:</b> Click <i>'🌫️ Winter Delhi Smog'</i> preset button.<br/>
    4. <b>Inspect Gauge & Advisory:</b> View instant Plotly speedometer gauge predicting AQI (285 - Poor) and color-coded health warning.<br/>
    5. <b>Inspect Feature Engineering & MLflow Tabs:</b> Show Tab 2 (PCA Scree Plot) and Tab 3 (MLflow Leaderboard Table).
    """
    story.append(Paragraph(demo_info, body_light_style))
    story.append(Spacer(1, 15))
    story.append(Paragraph("<b>🔊 Speaker Script:</b> <i>'Now let's look at the live working product. Here is our Streamlit dashboard calling our FastAPI REST backend. Selecting Delhi and loading the Winter Smog preset instantly predicts an AQI of 285 with a red health advisory warning citizens to wear N95 masks.'</i>", script_card_style))
    story.append(PageBreak())

    # ==================== SLIDE 8: DOCKER & CI/CD ====================
    story.append(Paragraph("📌 Slide 8: Docker Containerization & GitHub Actions CI/CD", slide_title_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=ACCENT_GREEN, spaceAfter=12))

    cicd_info = """
    • <b>Multi-Stage Dockerfile:</b> Containerizes FastAPI REST backend (port 8000) and Streamlit UI (port 8501) into a single production image.<br/>
    • <b>GitHub Actions Workflow (.github/workflows/ci-cd.yml):</b><br/>
      1. Triggered automatically on git push to main.<br/>
      2. Runs pytest automated test suite (6 passing unit tests).<br/>
      3. Builds Docker container image.<br/>
      4. Pushes container automatically to Docker Hub (riyaaa04/airpulse-aqi:latest).<br/>
    • <b>Cloud Deployment:</b> Deployed live on public cloud servers for instant presentation access.
    """
    story.append(Paragraph(cicd_info, body_light_style))
    story.append(Spacer(1, 15))
    story.append(Paragraph("<b>🔊 Speaker Script:</b> <i>'Finally, AirPulse is containerized with Docker and automated via GitHub Actions CI/CD. On every git push, automated pytest suites run and updated images are pushed to Docker Hub, proving it doesn't just run on my machine — it runs in production.'</i>", script_card_style))
    story.append(PageBreak())

    # ==================== SLIDE 9: CONCLUSION ====================
    story.append(Paragraph("📌 Slide 9: Value Proposition & Conclusion", slide_title_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=SKY_BLUE, spaceAfter=20))

    conc_info = """
    • <b>End-to-End Excellence:</b> Integrated raw data, DVC versioning, feature engineering, MLflow tracking, FastAPI REST API, Streamlit UI, Docker, and CI/CD.<br/>
    • <b>High Predictive Precision:</b> Achieved 93.39% variance explained (R² = 0.9339).<br/>
    • <b>Real-World Product Impact:</b> Public health advisories empowering citizens & urban planners.<br/><br/>
    <b>Thank you! Ready for Questions.</b>
    """
    story.append(Paragraph(conc_info, body_light_style))
    story.append(Spacer(1, 20))
    story.append(Paragraph("<b>🔊 Speaker Closing (15 sec):</b> <i>'AirPulse brings together feature engineering, reproducible data versioning, experiment tracking, and real-time API delivery into one scalable product. Thank you, and I am happy to take any questions!'</i>", script_card_style))

    # Page Canvas Background Generator for Dark Slide Theme
    def draw_slide_background(canvas, doc):
        canvas.saveState()
        canvas.setFillColor(NAVY_DARK)
        canvas.rect(0, 0, doc.pagesize[0], doc.pagesize[1], fill=True, stroke=False)
        canvas.restoreState()

    doc.build(story, onFirstPage=draw_slide_background, onLaterPages=draw_slide_background)
    print(f"Rich PDF Presentation successfully generated at: {output_filename}")

if __name__ == "__main__":
    create_rich_presentation_pdf()
