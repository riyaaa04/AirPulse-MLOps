import os
import sys

from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY

def create_presentation_pdf(output_filename="AirPulse_Presentation_Deck.pdf"):
    doc = SimpleDocTemplate(
        output_filename,
        pagesize=landscape(letter),
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()

    # Custom Color Palette
    PRIMARY = colors.HexColor("#0f172a")     # Slate Dark
    SECONDARY = colors.HexColor("#0284c7")   # Sky Blue Accent
    PURPLE = colors.HexColor("#7e22ce")      # Purple Accent
    TEXT_DARK = colors.HexColor("#1e293b")   # Text Charcoal
    BG_LIGHT = colors.HexColor("#f8fafc")    # Light Card
    ACCENT_GREEN = colors.HexColor("#16a34a")

    # Custom Typography Styles
    cover_title_style = ParagraphStyle(
        'CoverTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=28,
        leading=34,
        textColor=PRIMARY,
        alignment=TA_LEFT,
        spaceAfter=10
    )
    
    cover_subtitle_style = ParagraphStyle(
        'CoverSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=15,
        leading=20,
        textColor=SECONDARY,
        alignment=TA_LEFT,
        spaceAfter=20
    )

    slide_title_style = ParagraphStyle(
        'SlideTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=PRIMARY,
        spaceAfter=15
    )

    body_style = ParagraphStyle(
        'SlideBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=11,
        leading=16,
        textColor=TEXT_DARK,
        spaceAfter=8
    )

    script_style = ParagraphStyle(
        'SpeakerScript',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#334155"),
        backColor=colors.HexColor("#f1f5f9"),
        borderColor=SECONDARY,
        borderWidth=1,
        borderPadding=10,
        spaceBefore=10,
        spaceAfter=10
    )

    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=12,
        textColor=colors.white,
        alignment=TA_LEFT
    )

    table_cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=12,
        textColor=TEXT_DARK
    )

    story = []

    # ==================== SLIDE 1: COVER ====================
    story.append(Spacer(1, 40))
    story.append(Paragraph("🌬️ AirPulse: Real-Time Air Quality Intelligence & MLOps System", cover_title_style))
    story.append(Paragraph("Production Machine Learning Platform for India AQI Forecasting & Public Health Advisories", cover_subtitle_style))
    story.append(HRFlowable(width="100%", thickness=3, color=SECONDARY, spaceAfter=20))
    
    meta_text = """
    <b>Presenter:</b> Riya Singh<br/>
    <b>Architecture Stack:</b> DVC + MLflow + FastAPI + Streamlit + Docker + GitHub Actions CI/CD<br/>
    <b>Live Repository:</b> github.com/riyaaa04/AirPulse-MLOps<br/>
    <b>Promoted Model Performance:</b> GradientBoosting Regressor (R² Score: 0.9339 | RMSE: 34.80)
    """
    story.append(Paragraph(meta_text, body_style))
    story.append(Spacer(1, 30))
    story.append(Paragraph("<b>🔊 Speaker Opening (30 sec):</b> <i>'Good morning/afternoon. Today I present AirPulse — an end-to-end Air Quality Index forecasting system. Rather than showing a static notebook score, I am presenting a production-grade system backed by real REST APIs, MLflow experiment tracking, DVC pipeline versioning, and live health advisories.'</i>", script_style))
    story.append(PageBreak())

    # ==================== SLIDE 2: THE PROBLEM ====================
    story.append(Paragraph("📌 Slide 2: The Problem — Why Air Quality Forecasting Matters", slide_title_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=PRIMARY, spaceAfter=15))
    
    p_text = """
    • <b>The Public Health Hazard:</b> Over 1.6 million premature deaths in India annually linked to severe PM2.5 and PM10 particulate exposure.<br/>
    • <b>The Operational Gap:</b> Citizens and city planners lack real-time predictive tools to forecast tomorrow's AQI before stepping outside.<br/>
    • <b>The Data Challenge:</b> Environmental sensor logs suffer from missing readings due to maintenance, extreme right-skewed pollutant spikes, and complex chemical interactions (NO2, SO2, CO, O3).
    """
    story.append(Paragraph(p_text, body_style))
    story.append(Spacer(1, 15))
    story.append(Paragraph("<b>🔊 Speaker Script:</b> <i>'Air pollution in major Indian urban centers is a severe health hazard. Current apps only report what happened yesterday. What citizens and city planners need is a predictive tool — one that warns sensitive groups and calculates health severity buckets before AQI crosses hazardous levels.'</i>", script_style))
    story.append(PageBreak())

    # ==================== SLIDE 3: SYSTEM ARCHITECTURE ====================
    story.append(Paragraph("📌 Slide 3: End-to-End MLOps Pipeline Architecture", slide_title_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=PRIMARY, spaceAfter=15))
    
    arch_data = [
        [Paragraph("<b>Component</b>", table_header_style), Paragraph("<b>Technology</b>", table_header_style), Paragraph("<b>Role & Purpose</b>", table_header_style)],
        [Paragraph("Data Versioning", table_cell_style), Paragraph("DVC (Data Version Control)", table_cell_style), Paragraph("Tracks raw dataset & pipeline stages (prepare -> features -> train)", table_cell_style)],
        [Paragraph("Experiment Tracking", table_cell_style), Paragraph("MLflow Tracking Server", table_cell_style), Paragraph("Logs hyper-parameters, metrics (RMSE, R²), residual plots, & model registry", table_cell_style)],
        [Paragraph("REST Backend API", table_cell_style), Paragraph("FastAPI (Python)", table_cell_style), Paragraph("Serves low-latency /predict, /health, and /model-info endpoints", table_cell_style)],
        [Paragraph("Frontend UI", table_cell_style), Paragraph("Streamlit Dashboard", table_cell_style), Paragraph("Interactive pollutant sliders, speedometers, PCA visualizers, & batch processing", table_cell_style)],
        [Paragraph("Containerization & CI/CD", table_cell_style), Paragraph("Docker & GitHub Actions", table_cell_style), Paragraph("Automated pytest execution & Docker Hub image build on git push", table_cell_style)]
    ]
    t_arch = Table(arch_data, colWidths=[140, 160, 420])
    t_arch.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), PRIMARY),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, BG_LIGHT])
    ]))
    story.append(t_arch)
    story.append(Spacer(1, 15))
    story.append(Paragraph("<b>🔊 Speaker Script:</b> <i>'Here is how AirPulse works under the hood. Raw data is versioned with DVC. Our feature engineering pipeline transforms pollutant inputs, evaluated across multiple algorithms in MLflow. The winning model is served via a FastAPI REST API and visualized through a modern Streamlit dashboard containerized with Docker.'</i>", script_style))
    story.append(PageBreak())

    # ==================== SLIDE 4: FEATURE ENGINEERING ====================
    story.append(Paragraph("📌 Slide 4: Feature Engineering Rationale (Syllabus Phases 1–5)", slide_title_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=PRIMARY, spaceAfter=15))

    fe_data = [
        [Paragraph("<b>Syllabus Phase</b>", table_header_style), Paragraph("<b>Technique Applied</b>", table_header_style), Paragraph("<b>Domain Rationale & Evidence</b>", table_header_style)],
        [Paragraph("Phase 1: Foundations", table_cell_style), Paragraph("scikit-learn ColumnTransformer", table_cell_style), Paragraph("Prevents data leakage between training and validation splits.", table_cell_style)],
        [Paragraph("Phase 2: Cleaning & Prep", table_cell_style), Paragraph("Forward-fill + City Median, Log1p, RobustScaler", table_cell_style), Paragraph("Sensor downtime is MAR missingness; Log1p handles heavy right-skew; RobustScaler handles extreme winter smog spikes.", table_cell_style)],
        [Paragraph("Phase 3: Feature Creation", table_cell_style), Paragraph("Lags (1 & 3 days), 7-Day Rolling Stats, Sin/Cos Time", table_cell_style), Paragraph("PM2.5(t-1) & 7-day rolling averages capture pollutant momentum and multi-day weather accumulation.", table_cell_style)],
        [Paragraph("Phase 4: Feature Selection", table_cell_style), Paragraph("Variance Threshold & Mutual Information", table_cell_style), Paragraph("Filters zero-variance features and ranks primary pollutant drivers (PM2.5, PM10, NO2).", table_cell_style)],
        [Paragraph("Phase 5: Dimensionality", table_cell_style), Paragraph("PCA Scree Plot & Cumulative Variance", table_cell_style), Paragraph("Compresses multi-pollutant metrics into core air quality components while retaining ~95% variance.", table_cell_style)]
    ]
    t_fe = Table(fe_data, colWidths=[150, 180, 390])
    t_fe.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), SECONDARY),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, BG_LIGHT])
    ]))
    story.append(t_fe)
    story.append(Spacer(1, 10))
    story.append(Paragraph("<b>🔊 Speaker Script:</b> <i>'Every feature engineering choice was justified: Lags and rolling averages captured time continuity, Log1p handled skewed concentrations, and RobustScaler ensured extreme smog spikes didn't distort model gradients.'</i>", script_style))
    story.append(PageBreak())

    # ==================== SLIDE 5: MODEL COMPARISON ====================
    story.append(Paragraph("📌 Slide 5: MLflow Model Benchmark Leaderboard", slide_title_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=PRIMARY, spaceAfter=15))

    model_data = [
        [Paragraph("<b>Model Candidate</b>", table_header_style), Paragraph("<b>R² Score</b>", table_header_style), Paragraph("<b>RMSE</b>", table_header_style), Paragraph("<b>MAE</b>", table_header_style), Paragraph("<b>MLflow Status</b>", table_header_style)],
        [Paragraph("<b>GradientBoosting Regressor</b>", table_cell_style), Paragraph("<b>0.9339 (93.4%)</b>", table_cell_style), Paragraph("<b>34.80</b>", table_cell_style), Paragraph("<b>17.51</b>", table_cell_style), Paragraph("🏆 <b>Promoted to Production</b>", table_cell_style)],
        [Paragraph("RandomForest Regressor", table_cell_style), Paragraph("0.9319 (93.2%)", table_cell_style), Paragraph("35.30", table_cell_style), Paragraph("17.35", table_cell_style), Paragraph("Staging Candidate", table_cell_style)],
        [Paragraph("Ridge Regression", table_cell_style), Paragraph("0.8825 (88.3%)", table_cell_style), Paragraph("46.38", table_cell_style), Paragraph("23.05", table_cell_style), Paragraph("Baseline Model", table_cell_style)]
    ]
    t_model = Table(model_data, colWidths=[180, 110, 110, 110, 210])
    t_model.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), PRIMARY),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, BG_LIGHT])
    ]))
    story.append(t_model)
    story.append(Spacer(1, 20))
    story.append(Paragraph("<b>Key Takeaway:</b> GradientBoosting outperformed linear baselines by capturing non-linear chemical transformations and seasonal interaction ratios.", body_style))
    story.append(Spacer(1, 10))
    story.append(Paragraph("<b>🔊 Speaker Script:</b> <i>'Using MLflow, we evaluated 3 models. Baseline Ridge scored 0.88 R². GradientBoosting achieved 0.9339 R² with an RMSE of 34.8, promoting GradientBoosting to production.'</i>", script_style))
    story.append(PageBreak())

    # ==================== SLIDE 6: LIVE DEMO ====================
    story.append(Paragraph("📌 Slide 6: Mandatory Live Working Prototype Demo", slide_title_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=PRIMARY, spaceAfter=15))

    demo_text = """
    <b>Live Demo Flow during presentation:</b><br/>
    1. <b>Open App:</b> Navigate to Streamlit UI (http://localhost:8501 or live cloud link).<br/>
    2. <b>Select City:</b> Choose <i>Delhi</i> from sidebar.<br/>
    3. <b>Load Preset:</b> Click <i>'🌫️ Winter Delhi Smog'</i> preset button.<br/>
    4. <b>Inspect Gauge & Advisory:</b> View instant Plotly speedometer gauge predicting AQI (285 - Poor) and color-coded health warning.<br/>
    5. <b>Inspect PCA & Leaderboard:</b> Show Tab 2 (PCA Scree Plot) and Tab 3 (MLflow Benchmark Table).
    """
    story.append(Paragraph(demo_text, body_style))
    story.append(Spacer(1, 15))
    story.append(Paragraph("<b>🔊 Speaker Script:</b> <i>'Now let's look at the live working product. Here is our Streamlit dashboard calling our FastAPI REST backend. Selecting Delhi and loading the Winter Smog preset instantly predicts an AQI of 285 with a red health advisory warning citizens to wear N95 masks.'</i>", script_style))
    story.append(PageBreak())

    # ==================== SLIDE 7: DOCKER & CI/CD ====================
    story.append(Paragraph("📌 Slide 7: Docker, GitHub Actions CI/CD & Deployment", slide_title_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=PRIMARY, spaceAfter=15))

    cicd_text = """
    • <b>Docker Containerization:</b> Multi-stage Dockerfile running FastAPI backend (port 8000) and Streamlit UI (port 8501) via startup entrypoint.<br/>
    • <b>GitHub Actions Workflow (.github/workflows/ci-cd.yml):</b><br/>
      1. Triggered automatically on git push to main.<br/>
      2. Executes pytest test suite (6 passing unit tests).<br/>
      3. Builds Docker container image.<br/>
      4. Pushes container automatically to Docker Hub (riyaaa04/airpulse-aqi:latest).<br/>
    • <b>Cloud Deployment:</b> Hosted live on cloud servers for instant global access.
    """
    story.append(Paragraph(cicd_text, body_style))
    story.append(Spacer(1, 15))
    story.append(Paragraph("<b>🔊 Speaker Script:</b> <i>'Finally, AirPulse is containerized with Docker and automated via GitHub Actions CI/CD. On every git push, automated pytest suites run and updated images are pushed to Docker Hub, proving it doesn't just run on my machine — it runs in production.'</i>", script_style))
    story.append(PageBreak())

    # ==================== SLIDE 8: CONCLUSION ====================
    story.append(Paragraph("📌 Slide 8: Value Proposition & Conclusion", slide_title_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=PRIMARY, spaceAfter=20))

    conc_text = """
    • <b>End-to-End Excellence:</b> Integrated raw data, DVC versioning, feature engineering, MLflow tracking, FastAPI REST API, Streamlit UI, Docker, and CI/CD.<br/>
    • <b>High Predictive Precision:</b> Achieved 93.39% variance explained (R² = 0.9339).<br/>
    • <b>Real-World Product Impact:</b> Public health advisories empowering citizens & urban planners.<br/><br/>
    <b>Thank you! Ready for Questions.</b>
    """
    story.append(Paragraph(conc_text, body_style))
    story.append(Spacer(1, 20))
    story.append(Paragraph("<b>🔊 Speaker Closing (15 sec):</b> <i>'AirPulse brings together feature engineering, reproducible data versioning, experiment tracking, and real-time API delivery into one scalable product. Thank you, and I am happy to take any questions!'</i>", script_style))

    doc.build(story)
    print(f"Presentation PDF successfully created at: {output_filename}")

if __name__ == "__main__":
    create_presentation_pdf()
