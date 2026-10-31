"""
Generate a professional PDF guide for deploying the Infosys Chatbot on Render.com
with the complete Key Vault / Environment Variable specifications.
"""
import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    KeepTogether,
    HRFlowable,
)

PDF_FILENAME = "render_deployment_guide.pdf"

def generate_pdf():
    doc = SimpleDocTemplate(
        PDF_FILENAME,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    primary_color = colors.HexColor("#0f2c59")
    accent_color = colors.HexColor("#0070f3")
    dark_gray = colors.HexColor("#333333")
    light_bg = colors.HexColor("#f8f9fa")

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=primary_color,
        spaceAfter=6,
    )

    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#555555"),
        spaceAfter=12,
    )

    heading2_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=17,
        textColor=primary_color,
        spaceBefore=12,
        spaceAfter=6,
    )

    body_style = ParagraphStyle(
        "BodyDark",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=dark_gray,
        spaceAfter=6,
    )

    bullet_style = ParagraphStyle(
        "BulletDark",
        parent=body_style,
        leftIndent=14,
        firstLineIndent=-10,
        spaceAfter=4,
    )

    code_style = ParagraphStyle(
        "CodeStyle",
        parent=styles["Normal"],
        fontName="Courier-Bold",
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#c7254e"),
    )

    table_header_style = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10,
        textColor=colors.white,
    )

    table_cell_style = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.5,
        leading=9.5,
        textColor=dark_gray,
    )

    table_code_style = ParagraphStyle(
        "TableCode",
        parent=styles["Normal"],
        fontName="Courier",
        fontSize=7,
        leading=9,
        textColor=colors.HexColor("#0b5394"),
    )

    story = []

    # Title & Metadata
    story.append(Paragraph("InfyOne AI Assistant — Render.com Deployment Guide", title_style))
    story.append(Paragraph("Production Cloud Deployment, Service Configuration & Key Vault / Environment Variables", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=primary_color, spaceAfter=10))

    # Section 1: Overview
    story.append(Paragraph("1. Deployment Architecture Overview", heading2_style))
    story.append(Paragraph(
        "The application is containerized and cloud-ready for <b>Render.com Web Services</b>. "
        "It features a <b>FastAPI</b> backend running on Python, orchestrating a <b>LangGraph</b> stateful agent "
        "checkpointer, backed by <b>OpenAI (gpt-4o-mini & text-embedding-3-large)</b>, <b>Chroma Cloud</b> for vector "
        "knowledge retrieval, and optional <b>Redis Cloud</b> for distributed session persistence. "
        "All configuration is dynamically injected through Render's Environment Key Vault at container startup.",
        body_style
    ))

    # Section 2: Render Service Setup
    story.append(Paragraph("2. Render.com Web Service Setup Instructions", heading2_style))
    story.append(Paragraph("Follow these exact steps in your <b>Render Dashboard (dashboard.render.com)</b>:", body_style))
    story.append(Paragraph("<b>1. Create Web Service:</b> Click <i>New +</i> &gt; <i>Web Service</i>, then connect your GitHub/GitLab repository.", bullet_style))
    story.append(Paragraph("<b>2. Runtime Environment:</b> Select <b>Python 3</b>.", bullet_style))
    story.append(Paragraph("<b>3. Build Command:</b> <font name='Courier'>pip install -r requirements.txt</font>", bullet_style))
    story.append(Paragraph("<b>4. Start Command:</b> <font name='Courier'>uvicorn app.main:app --host 0.0.0.0 --port $PORT</font>", bullet_style))
    story.append(Paragraph("<b>5. Health Check Path:</b> <font name='Courier'>/api/health</font> (ensures zero-downtime rolling deploys).", bullet_style))
    story.append(Paragraph("<b>6. Instance Plan:</b> Free or Starter (Standard 512MB RAM minimum).", bullet_style))

    story.append(Spacer(1, 8))

    # Section 3: Key Vault Table
    story.append(Paragraph("3. Render Key Vault: Complete Environment Variables List", heading2_style))
    story.append(Paragraph(
        "Enter the following Key-Value pairs in your Render Dashboard under <b>Environment &gt; Environment Variables</b>. "
        "Render securely encrypts these keys and injects them directly into the runtime container.",
        body_style
    ))

    raw_table_data = [
        [
            Paragraph("Category", table_header_style),
            Paragraph("Key Vault Name", table_header_style),
            Paragraph("Required?", table_header_style),
            Paragraph("Production Value / Setting", table_header_style),
            Paragraph("Purpose & Description", table_header_style),
        ],
        # Application
        [
            Paragraph("Application", table_cell_style),
            Paragraph("APP_NAME", table_code_style),
            Paragraph("Optional", table_cell_style),
            Paragraph("InfyOne", table_code_style),
            Paragraph("Public application name displayed in logs & health check.", table_cell_style),
        ],
        [
            Paragraph("Application", table_cell_style),
            Paragraph("APP_ENV", table_code_style),
            Paragraph("Required", table_cell_style),
            Paragraph("production", table_code_style),
            Paragraph("Disables debug overhead, swagger docs, and activates production handlers.", table_cell_style),
        ],
        [
            Paragraph("Application", table_cell_style),
            Paragraph("APP_DEBUG", table_code_style),
            Paragraph("Required", table_cell_style),
            Paragraph("false", table_code_style),
            Paragraph("Enforces strict production security and minimizes verbose tracebacks.", table_cell_style),
        ],
        [
            Paragraph("Application", table_cell_style),
            Paragraph("APP_HOST", table_code_style),
            Paragraph("Required", table_cell_style),
            Paragraph("0.0.0.0", table_code_style),
            Paragraph("Binds FastAPI to all network interfaces inside the Render container.", table_cell_style),
        ],
        [
            Paragraph("Application", table_cell_style),
            Paragraph("APP_ALLOWED_HOSTS", table_code_style),
            Paragraph("Required", table_cell_style),
            Paragraph('["*"]', table_code_style),
            Paragraph("Allows requests routed via *.onrender.com and custom domain names.", table_cell_style),
        ],
        [
            Paragraph("Application", table_cell_style),
            Paragraph("APP_CORS_ORIGINS", table_code_style),
            Paragraph("Required", table_cell_style),
            Paragraph('["*"]', table_code_style),
            Paragraph("Enables browser web chat client cross-origin API communication.", table_cell_style),
        ],
        [
            Paragraph("Application", table_cell_style),
            Paragraph("APP_SECRET_KEY", table_code_style),
            Paragraph("Required", table_cell_style),
            Paragraph("0fc53de77df447e9dd4622...", table_code_style),
            Paragraph("Cryptographic secret token for session signatures and security.", table_cell_style),
        ],
        # OpenAI Model
        [
            Paragraph("OpenAI LLM", table_cell_style),
            Paragraph("OPENAI_API_KEY", table_code_style),
            Paragraph("CRITICAL", table_cell_style),
            Paragraph("sk-proj-K6--23M... (Your Key)", table_code_style),
            Paragraph("OpenAI API authentication key for generation and embeddings.", table_cell_style),
        ],
        [
            Paragraph("OpenAI LLM", table_cell_style),
            Paragraph("OPENAI_MODEL", table_code_style),
            Paragraph("Required", table_cell_style),
            Paragraph("gpt-4o-mini", table_code_style),
            Paragraph("Fast, low-latency, highly accurate model configured for chat pipeline.", table_cell_style),
        ],
        [
            Paragraph("Embeddings", table_cell_style),
            Paragraph("EMBEDDING_PROVIDER", table_code_style),
            Paragraph("Required", table_cell_style),
            Paragraph("openai", table_code_style),
            Paragraph("Embedding generator engine (OpenAI text-embedding API).", table_cell_style),
        ],
        [
            Paragraph("Embeddings", table_cell_style),
            Paragraph("EMBEDDING_MODEL", table_code_style),
            Paragraph("Required", table_cell_style),
            Paragraph("text-embedding-3-large", table_code_style),
            Paragraph("3072-dimensional embedding model matching the Chroma collection.", table_cell_style),
        ],
        [
            Paragraph("Embeddings", table_cell_style),
            Paragraph("EMBEDDING_DIMENSION", table_code_style),
            Paragraph("Required", table_cell_style),
            Paragraph("3072", table_code_style),
            Paragraph("Vector dimension specification for cosine distance similarity.", table_cell_style),
        ],
        # Chroma Cloud
        [
            Paragraph("Chroma Cloud", table_cell_style),
            Paragraph("CHROMA_CLOUD_API_KEY", table_code_style),
            Paragraph("CRITICAL", table_cell_style),
            Paragraph("ck-CLbEQVRCf58NnFX92...", table_code_style),
            Paragraph("API key connecting backend directly to hosted Chroma Cloud vector DB.", table_cell_style),
        ],
        [
            Paragraph("Chroma Cloud", table_cell_style),
            Paragraph("CHROMA_TENANT", table_code_style),
            Paragraph("Required", table_cell_style),
            Paragraph("da86004b-97e5-490c-ac...", table_code_style),
            Paragraph("Chroma Cloud account tenant identifier (UUID format).", table_cell_style),
        ],
        [
            Paragraph("Chroma Cloud", table_cell_style),
            Paragraph("CHROMA_DATABASE", table_code_style),
            Paragraph("Required", table_cell_style),
            Paragraph("StarkBots", table_code_style),
            Paragraph("Chroma Cloud database holding Infosys knowledge chunks.", table_cell_style),
        ],
        [
            Paragraph("Chroma Cloud", table_cell_style),
            Paragraph("CHROMA_COLLECTION_NAME", table_code_style),
            Paragraph("Required", table_cell_style),
            Paragraph("InfyOne", table_code_style),
            Paragraph("Target collection containing the 100 Infosys knowledge chunks.", table_cell_style),
        ],
        [
            Paragraph("Chroma Cloud", table_cell_style),
            Paragraph("CHROMA_DISTANCE_METRIC", table_code_style),
            Paragraph("Required", table_cell_style),
            Paragraph("cosine", table_code_style),
            Paragraph("Distance metric used for HNSW vector space similarity search.", table_cell_style),
        ],
        # Redis (Optional)
        [
            Paragraph("Redis Cache", table_cell_style),
            Paragraph("REDIS_HOST", table_code_style),
            Paragraph("Optional", table_cell_style),
            Paragraph("redis-14271.c330...redislabs.com", table_code_style),
            Paragraph("Hosted Redis endpoint for cross-restart distributed caching.", table_cell_style),
        ],
        [
            Paragraph("Redis Cache", table_cell_style),
            Paragraph("REDIS_PORT", table_code_style),
            Paragraph("Optional", table_cell_style),
            Paragraph("14271", table_code_style),
            Paragraph("Redis Cloud service network port number.", table_cell_style),
        ],
        [
            Paragraph("Redis Cache", table_cell_style),
            Paragraph("REDIS_PASSWORD", table_code_style),
            Paragraph("Optional", table_cell_style),
            Paragraph("IZ3rUdsemHJsoiM3V9...", table_code_style),
            Paragraph("Redis authentication password credential.", table_cell_style),
        ],
        # Logging & Runtime
        [
            Paragraph("Logging", table_cell_style),
            Paragraph("LOG_LEVEL", table_code_style),
            Paragraph("Optional", table_cell_style),
            Paragraph("INFO", table_code_style),
            Paragraph("Terminal log verbosity level (INFO recommended in production).", table_cell_style),
        ],
        [
            Paragraph("Logging", table_cell_style),
            Paragraph("LOG_FORMAT", table_code_style),
            Paragraph("Optional", table_cell_style),
            Paragraph("json", table_code_style),
            Paragraph("Formats output as single-line JSON records for Render log viewer.", table_cell_style),
        ],
    ]

    col_widths = [65, 115, 50, 140, 170]
    table = Table(raw_table_data, colWidths=col_widths, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), primary_color),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dddddd")),
        ("BOX", (0, 0), (-1, -1), 1, primary_color),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, light_bg]),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))

    story.append(table)
    story.append(Spacer(1, 10))

    # Section 4: Post-Deployment Verification
    story.append(Paragraph("4. Post-Deployment Verification Checklist", heading2_style))
    story.append(Paragraph("Once Render deploys your service, test these 3 endpoints to verify operation:", body_style))
    story.append(Paragraph("<b>1. Health Check:</b> <font name='Courier'>GET https://&lt;your-app&gt;.onrender.com/api/health</font> &rarr; returns <font name='Courier'>{\"status\": \"healthy\"}</font>.", bullet_style))
    story.append(Paragraph("<b>2. Web Interface:</b> Visit <font name='Courier'>https://&lt;your-app&gt;.onrender.com/</font> &rarr; loads interactive InfyOne chatbot UI.", bullet_style))
    story.append(Paragraph("<b>3. Chat Execution:</b> Send <font name='Courier'>POST /api/chat</font> with payload <font name='Courier'>{\"message\": \"Who is the next Infosys CEO?\"}</font> &rarr; returns Ashiss Kumar Dash + sources.", bullet_style))

    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#bbbbbb"), spaceAfter=8))
    story.append(Paragraph(
        "<i>Render Blueprint File: A pre-configured <font name='Courier'>render.yaml</font> has also been created in your repository root for 1-click automated Blueprint deployment.</i>",
        subtitle_style
    ))

    doc.build(story)
    print(f"PDF successfully generated at: {os.path.abspath(PDF_FILENAME)}")

if __name__ == "__main__":
    generate_pdf()

