import io
import datetime
import tempfile
import numpy as np
from PIL import Image

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer,
    Image as RLImage, Table, TableStyle, HRFlowable,
)


def build_pdf(
    patient:     dict,
    operator:    str,
    pred_name:   str,
    confidence:  float,
    probs:       list,
    orig_pil:    Image.Image,
    seg_pil:     Image.Image,
    report_text: str,
    class_names: list,
    stage_info:  dict,
) -> bytes:
    """
    Génère le rapport PDF complet et retourne les bytes du fichier.

    Contenu :
        - En-tête + numéro de référence
        - Avertissement IA obligatoire
        - Informations patient
        - Résultats IA (classification + distribution)
        - Images (originale + segmentation côte à côte)
        - Rapport clinique généré par LLM
        - Pied de page traçabilité
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm,  bottomMargin=2*cm,
    )

    # ── Couleurs ──────────────────────────────────────────
    NAVY     = colors.HexColor("#1a2744")
    BLUE     = colors.HexColor("#2563eb")
    GRAY     = colors.HexColor("#6b7280")
    LGRAY    = colors.HexColor("#e2e8f0")
    AMBER    = colors.HexColor("#f59e0b")
    AMBER_BG = colors.HexColor("#fffbeb")
    LBLUE_BG = colors.HexColor("#eff6ff")
    DARK     = colors.HexColor("#374151")

    # ── Styles typographiques ─────────────────────────────
    H1 = ParagraphStyle("H1", fontSize=15, fontName="Helvetica-Bold",
                         textColor=NAVY, spaceAfter=4)
    H2 = ParagraphStyle("H2", fontSize=9.5, fontName="Helvetica-Bold",
                         textColor=BLUE, spaceBefore=12, spaceAfter=4)
    BODY = ParagraphStyle("BODY", fontSize=9, fontName="Helvetica",
                           textColor=DARK, leading=14, spaceAfter=3,
                           alignment=TA_JUSTIFY)
    SM = ParagraphStyle("SM", fontSize=7.5, fontName="Helvetica",
                          textColor=GRAY, leading=11)
    WARN = ParagraphStyle("WARN", fontSize=8.5, fontName="Helvetica-Bold",
                           textColor=colors.HexColor("#92400e"), leading=12)

    story    = []
    date_str = datetime.datetime.now().strftime("%d/%m/%Y — %H:%M")
    ref      = f"UA-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"

    # ── En-tête ───────────────────────────────────────────
    hd = Table([[
        Paragraph(
            "<b>ULCER ANALYZER</b><br/>"
            "<font size='8' color='#6b7280'>Outil d'aide au diagnostic — Usage clinique</font>",
            H1
        ),
        Paragraph(
            f"<font size='8' color='#6b7280'>"
            f"Date : {date_str}<br/>Opérateur : {operator}<br/>Réf. : {ref}"
            f"</font>",
            SM
        ),
    ]], colWidths=[12*cm, 5*cm])
    hd.setStyle(TableStyle([
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ("LINEBELOW",     (0, 0), (-1,  0), 1.5, BLUE),
        ("BOTTOMPADDING", (0, 0), (-1,  0), 8),
    ]))
    story += [hd, Spacer(1, .35*cm)]

    # ── Avertissement ─────────────────────────────────────
    wa = Table([[Paragraph(
        "AVERTISSEMENT : Ce rapport est généré par un système d'intelligence artificielle "
        "à titre d'aide au diagnostic uniquement. Il ne remplace en aucun cas l'évaluation "
        "clinique d'un professionnel de santé qualifié. Toute décision thérapeutique doit "
        "être validée par un médecin.",
        WARN
    )]], colWidths=[17*cm])
    wa.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), AMBER_BG),
        ("BOX",           (0, 0), (-1, -1), 1, AMBER),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 10),
        ("TOPPADDING",    (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    story += [wa, Spacer(1, .4*cm)]

    # ── Informations patient ──────────────────────────────
    story.append(Paragraph("INFORMATIONS PATIENT", H2))
    pt = Table([
        ["Nom / Prénom",        patient["nom"]],
        ["Âge",                 f"{patient['age']} ans"],
        ["Sexe",                patient["sexe"]],
        ["Localisation",        patient["localisation"]],
        ["Observation clinique", patient["observation"]],
    ], colWidths=[5*cm, 12*cm])
    pt.setStyle(TableStyle([
        ("FONTNAME",       (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE",       (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1),
         [colors.HexColor("#f8fafc"), colors.white]),
        ("TOPPADDING",     (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING",  (0, 0), (-1, -1), 5),
        ("LEFTPADDING",    (0, 0), (-1, -1), 8),
        ("BOX",            (0, 0), (-1, -1), .5, LGRAY),
        ("INNERGRID",      (0, 0), (-1, -1), .5, LGRAY),
    ]))
    story += [pt, Spacer(1, .4*cm)]

    # ── Résultats IA ──────────────────────────────────────
    story.append(Paragraph("RÉSULTATS DE L'ANALYSE IA", H2))
    desc = stage_info[pred_name][3]
    rt = Table([
        ["Classification", f"{pred_name.replace('_', ' ')} — {desc}"],
        ["Confiance",       f"{confidence*100:.1f}%"],
    ], colWidths=[5*cm, 12*cm])
    rt.setStyle(TableStyle([
        ("FONTNAME",       (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE",       (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [LBLUE_BG, colors.white]),
        ("TOPPADDING",     (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING",  (0, 0), (-1, -1), 5),
        ("LEFTPADDING",    (0, 0), (-1, -1), 8),
        ("BOX",            (0, 0), (-1, -1), .5, colors.HexColor("#bfdbfe")),
        ("INNERGRID",      (0, 0), (-1, -1), .5, colors.HexColor("#bfdbfe")),
    ]))
    story += [rt, Spacer(1, .25*cm)]

    # Distribution des probabilités
    story.append(Paragraph("Distribution des probabilités :", BODY))
    pb = Table(
        [["Classe", "Probabilité"]] + [
            [class_names[i].replace("_", " "), f"{probs[i]*100:.1f}%"]
            for i in np.argsort(probs)[::-1]
        ],
        colWidths=[8*cm, 9*cm],
    )
    pb.setStyle(TableStyle([
        ("FONTNAME",       (0, 0), (-1,  0), "Helvetica-Bold"),
        ("FONTSIZE",       (0, 0), (-1, -1), 8.5),
        ("BACKGROUND",     (0, 0), (-1,  0), NAVY),
        ("TEXTCOLOR",      (0, 0), (-1,  0), colors.white),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.HexColor("#f8fafc"), colors.white]),
        ("TOPPADDING",     (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",  (0, 0), (-1, -1), 4),
        ("LEFTPADDING",    (0, 0), (-1, -1), 8),
        ("BOX",            (0, 0), (-1, -1), .5, LGRAY),
        ("INNERGRID",      (0, 0), (-1, -1), .5, LGRAY),
    ]))
    story += [pb, Spacer(1, .4*cm)]

    # ── Images ────────────────────────────────────────────
    story.append(Paragraph("IMAGERIE", H2))

    def _to_rl_image(pil_img, w_cm, h_cm):
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        pil_img.resize((400, 400), Image.LANCZOS).save(tmp.name)
        return RLImage(tmp.name, width=w_cm*cm, height=h_cm*cm)

    img_table = Table([
        [_to_rl_image(orig_pil, 7.5, 7.5), _to_rl_image(seg_pil, 7.5, 7.5)],
        [Paragraph("<i>Image originale</i>", SM),
         Paragraph("<i>Segmentation (overlay heatmap)</i>", SM)],
    ], colWidths=[8.5*cm, 8.5*cm])
    img_table.setStyle(TableStyle([
        ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story += [img_table, Spacer(1, .4*cm)]

    # ── Rapport LLM ───────────────────────────────────────
    story.append(Paragraph("RAPPORT CLINIQUE GÉNÉRÉ PAR IA", H2))
    for line in report_text.split("\n"):
        if line.strip():
            story.append(Paragraph(line.strip(), BODY))

    # ── Pied de page ──────────────────────────────────────
    story += [
        Spacer(1, .4*cm),
        HRFlowable(width="100%", thickness=.5, color=LGRAY),
        Spacer(1, .2*cm),
        Paragraph(
            f"Document généré par Ulcer Analyzer — {date_str} — "
            f"Opérateur : {operator} — Réf. : {ref}<br/>"
            "Document confidentiel réservé au personnel soignant autorisé.",
            SM,
        ),
    ]

    doc.build(story)
    return buf.getvalue()