import datetime
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import torchvision.models as models
import segmentation_models_pytorch as smp

from llm    import generate_report_text
from report import build_pdf

# =======================================================
# CONFIG PAGE
# =======================================================

st.set_page_config(
    page_title="Ulcer Analyzer — Outil clinique",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =======================================================
# CSS — Thème médical
# =======================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family:'IBM Plex Sans',sans-serif; font-size:14px; }
.stApp { background:#f4f6f9; color:#1a1f2e; }

.topbar {
    background:#1a2744; padding:0.85rem 2rem;
    display:flex; align-items:center; justify-content:space-between;
    margin:-1rem -1rem 1.8rem -1rem;
    border-bottom:3px solid #2563eb;
}
.topbar-title { color:white; font-size:1.05rem; font-weight:600; }
.topbar-right  { color:#93c5fd; font-size:0.78rem; font-family:'IBM Plex Mono',monospace; }
.topbar-badge  {
    background:#2563eb; color:white; font-size:0.7rem;
    padding:0.18rem 0.65rem; border-radius:20px; font-weight:500;
    letter-spacing:0.4px; margin-left:0.8rem;
}
.card {
    background:white; border:1px solid #e2e8f0; border-radius:8px;
    padding:1.2rem 1.4rem; margin-bottom:0.9rem;
    box-shadow:0 1px 3px rgba(0,0,0,0.05);
}
.card-title {
    font-size:0.68rem; text-transform:uppercase; letter-spacing:1.6px;
    color:#94a3b8; margin-bottom:0.45rem; font-weight:500;
}
.section-title {
    font-size:0.8rem; font-weight:600; color:#1a2744;
    text-transform:uppercase; letter-spacing:1px;
    margin-bottom:0.7rem; padding-bottom:0.35rem;
    border-bottom:2px solid #e2e8f0;
}
.warning-box {
    background:#fffbeb; border:1px solid #f59e0b;
    border-left:4px solid #f59e0b; border-radius:6px;
    padding:0.75rem 1rem; font-size:0.81rem; color:#92400e;
    margin-bottom:1.2rem;
}
.stage-badge {
    display:inline-block; padding:0.3rem 0.9rem; border-radius:4px;
    font-size:0.86rem; font-weight:600;
}
.conf-track { background:#e2e8f0; border-radius:4px; height:7px; overflow:hidden; margin-top:0.45rem; }
.conf-fill  { height:100%; border-radius:4px; }
.prob-row {
    display:flex; align-items:center; gap:0.75rem;
    padding:0.32rem 0; border-bottom:1px solid #f1f5f9; font-size:0.82rem;
}
.prob-row:last-child { border-bottom:none; }
.prob-name { width:125px; color:#475569; }
.prob-bar-bg   { flex:1; background:#f1f5f9; border-radius:3px; height:5px; }
.prob-bar-fill { height:5px; border-radius:3px; }
.prob-pct {
    width:38px; text-align:right; color:#64748b;
    font-family:'IBM Plex Mono',monospace; font-size:0.76rem;
}
#MainMenu, footer, header { visibility:hidden; }
.block-container { padding-top:0; padding-bottom:2rem; max-width:1200px; }
</style>
""", unsafe_allow_html=True)

# =======================================================
# CONSTANTES
# =======================================================

CLASS_NAMES = ["Invalid","Sdti","Stage_I","Stage_II","Stage_III","Stage_IV","Unstageable"]
IMG_SIZE    = 224
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEG_THRESH  = 0.30

# (text_color, bg_color, accent_color, description)
STAGE_INFO = {
    "Invalid":     ("#475569", "#f1f5f9", "#64748b", "Image non médicale / hors contexte"),
    "Sdti":        ("#92400e", "#fffbeb", "#f59e0b", "Lésion tissu profond suspectée"),
    "Stage_I":     ("#065f46", "#ecfdf5", "#10b981", "Rougeur persistante"),
    "Stage_II":    ("#1e40af", "#eff6ff", "#3b82f6", "Perte partielle d'épaisseur"),
    "Stage_III":   ("#7c2d12", "#fff7ed", "#f97316", "Perte totale d'épaisseur"),
    "Stage_IV":    ("#7f1d1d", "#fef2f2", "#ef4444", "Lésion sévère (muscle/os)"),
    "Unstageable": ("#4c1d95", "#f5f3ff", "#8b5cf6", "Profondeur indéterminée"),
}

# Opérateurs autorisés — remplacer par BDD/LDAP en production
OPERATORS = {
    "DEMO":   "Utilisateur Démo",
}

# =======================================================
# MODÈLES (mêmes architectures que models.py)
# =======================================================

class Classifier(nn.Module):
    def __init__(self, num_classes=7, dropout=0.4):
        super().__init__()
        self.model = models.resnet18(weights=None)
        in_f = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_f, 256),
            nn.ReLU(), 
            nn.Dropout(p=dropout/2), 
            nn.Linear(256, num_classes)
        )
    def forward(self, x): return self.model(x)

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = smp.Unet(
            encoder_name="resnet34", encoder_weights=None,
            in_channels=3, classes=1, activation=None
        )
    def forward(self, x): return self.model(x)

@st.cache_resource
def load_models():
    clf = Classifier(num_classes=7).to(DEVICE)
    seg = UNet().to(DEVICE)
    for model, path in [
        (clf, "../Checkpoints/best_classifier_model.pth"),
        (seg, "../Checkpoints/best_segmentation_model.pth"),
    ]:
        try:
            model.load_state_dict(torch.load(path, map_location=DEVICE))
        except FileNotFoundError:
            pass
    clf.eval()
    seg.eval()
    return clf, seg

# =======================================================
# UTILS
# =======================================================

preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def apply_heatmap(img_np, prob_map, alpha=0.45):
    import matplotlib.cm as cm
    heatmap = cm.hot(prob_map)[:, :, :3]
    mask    = prob_map[:, :, None] > SEG_THRESH
    return np.clip(img_np * (1 - alpha*mask) + heatmap * alpha*mask, 0, 1)

def conf_color(c):
    return "#10b981" if c >= 0.75 else ("#f59e0b" if c >= 0.50 else "#ef4444")

@torch.no_grad()
def run_inference(pil_img, clf_model, seg_model):
    t        = preprocess(pil_img).unsqueeze(0).to(DEVICE)
    probs    = F.softmax(clf_model(t), dim=1)[0].cpu().numpy()
    prob_map = torch.sigmoid(seg_model(t)).squeeze().cpu().numpy()
    return int(np.argmax(probs)), probs, prob_map

# =======================================================
# SESSION STATE
# =======================================================

for key, val in [("authenticated", False), ("operator_code", ""), ("operator_name", "")]:
    if key not in st.session_state:
        st.session_state[key] = val

# =======================================================
# PAGE LOGIN
# =======================================================

if not st.session_state.authenticated:
    _, col, _ = st.columns([1, 1.2, 1])
    with col:
        st.markdown("""
        <div style="background:white;border:1px solid #e2e8f0;border-radius:12px;
                    padding:2.5rem;box-shadow:0 4px 20px rgba(0,0,0,0.07);margin-top:3rem;">
            <div style="text-align:center;font-size:2.4rem;margin-bottom:.4rem;"></div>
            <div style="text-align:center;font-size:1.15rem;font-weight:600;color:#1a2744;">
                Ulcer Analyzer
            </div>
            <div style="text-align:center;font-size:0.78rem;color:#94a3b8;margin-bottom:1.6rem;">
                Outil clinique d'aide au diagnostic<br>Identifiez-vous pour accéder
            </div>
        </div>
        """, unsafe_allow_html=True)

        code = st.text_input("Code opérateur", placeholder="ex : INF001", max_chars=10)
        if st.button("Se connecter", use_container_width=True, type="primary"):
            c = code.strip().upper()
            if c in OPERATORS:
                st.session_state.authenticated = True
                st.session_state.operator_code = c
                st.session_state.operator_name = OPERATORS[c]
                st.rerun()
            else:
                st.error("Code non reconnu. Contactez l'administration.")
        st.caption("Code démo : **DEMO**")
    st.stop()

# =======================================================
# PAGE PRINCIPALE
# =======================================================

clf_model, seg_model = load_models()

# Top bar
st.markdown(f"""
<div class="topbar">
    <div>
        <span class="topbar-title"> Ulcer Analyzer</span>
        <span class="topbar-badge">Outil clinique</span>
    </div>
    <div class="topbar-right">
         {st.session_state.operator_name} &nbsp;·&nbsp;
        {st.session_state.operator_code} &nbsp;·&nbsp;
        {datetime.datetime.now().strftime("%d/%m/%Y %H:%M")}
    </div>
</div>
""", unsafe_allow_html=True)

# Avertissement permanent
st.markdown("""
<div class="warning-box">
    <strong>⚠️ Avertissement :</strong> Ce système est un <strong>outil d'aide au diagnostic</strong>
    basé sur l'intelligence artificielle. Les résultats sont indicatifs et ne remplacent pas
    l'évaluation clinique d'un professionnel de santé. Toute décision thérapeutique doit être
    validée par un médecin.
</div>
""", unsafe_allow_html=True)

# == Layout principal =================================
col_left, col_right = st.columns([1, 1.5], gap="large")

with col_left:
    st.markdown('<div class="section-title">Image à analyser</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("", type=["jpg","jpeg","png"], label_visibility="collapsed")

    if uploaded:
        pil_img = Image.open(uploaded).convert("RGB")
        st.image(pil_img, use_container_width=True)
    else:
        st.markdown("""
        <div style="border:1.5px dashed #cbd5e1;border-radius:8px;padding:2.5rem 1rem;
                    text-align:center;color:#94a3b8;font-size:0.84rem;background:white;">
            Déposez une image ici<br>
            <span style="font-size:0.74rem;">JPG · JPEG · PNG</span>
        </div>""", unsafe_allow_html=True)

    # Renseignements patient
    st.markdown('<div class="section-title" style="margin-top:1.1rem;">Renseignements patient</div>',
                unsafe_allow_html=True)
    p_nom  = st.text_input("Nom / Prénom", placeholder="Nom Prénom")
    c1, c2 = st.columns(2)
    with c1: p_age  = st.number_input("Âge", min_value=0, max_value=120, value=65)
    with c2: p_sexe = st.selectbox("Sexe", ["Masculin", "Féminin", "Autre"])
    p_loc  = st.text_input("Localisation de la plaie", placeholder="ex : talon gauche")
    p_obs  = st.text_area("Observation clinique", placeholder="Aspect de la plaie…", height=85)

    patient = {
        "nom": p_nom or "Non renseigné", "age": p_age,
        "sexe": p_sexe, "localisation": p_loc or "Non renseigné",
        "observation": p_obs or "Non renseigné",
    }

    analyze_btn = st.button(
        "Analyser", use_container_width=True,
        type="primary", disabled=(uploaded is None)
    )

with col_right:
    st.markdown('<div class="section-title">Résultats</div>', unsafe_allow_html=True)

    # Lancer l'analyse
    if uploaded and analyze_btn:
        with st.spinner("Analyse en cours…"):
            pred_idx, probs, prob_map = run_inference(pil_img, clf_model, seg_model)
        st.session_state.results = dict(
            pred_idx=pred_idx, probs=probs, prob_map=prob_map,
            pil_img=pil_img, patient=patient,
        )

    if "results" in st.session_state:
        r          = st.session_state.results
        pred_idx   = r["pred_idx"]
        probs      = r["probs"]
        prob_map   = r["prob_map"]
        pil_img    = r["pil_img"]
        patient    = r["patient"]
        pred_name  = CLASS_NAMES[pred_idx]
        tc, bg, ac, desc = STAGE_INFO[pred_name]
        confidence = float(probs[pred_idx])

        # Résultat classification
        st.markdown(f"""
        <div class="card">
            <div class="card-title">Classification détectée</div>
            <span class="stage-badge" style="background:{bg};color:{tc};">
                {pred_name.replace('_', ' ')}
            </span>
            <div style="font-size:0.86rem;color:#475569;margin-top:.4rem;">{desc}</div>
            <div class="conf-track">
                <div class="conf-fill"
                     style="width:{confidence*100:.0f}%;background:{conf_color(confidence)};"></div>
            </div>
            <div style="font-size:0.76rem;color:{conf_color(confidence)};
                        margin-top:.3rem;font-weight:500;">
                Confiance : {confidence*100:.1f}%
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Segmentation overlay
        img_np = np.array(pil_img.resize((IMG_SIZE, IMG_SIZE))).astype(np.float32) / 255.
        pm_r   = np.array(
            Image.fromarray((prob_map * 255).astype(np.uint8))
            .resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        ).astype(np.float32) / 255.
        overlay     = apply_heatmap(img_np, pm_r)
        overlay_pil = Image.fromarray((overlay * 255).astype(np.uint8))
        st.image(overlay_pil,
                 caption="Zones d'intérêt détectées par segmentation",
                 use_container_width=True)

        # Distribution des probabilités
        st.markdown('<div class="card-title" style="margin-top:.7rem;">Distribution</div>',
                    unsafe_allow_html=True)
        rows = ""
        for i in np.argsort(probs)[::-1]:
            w   = f"{probs[i]*100:.0f}%"
            col = ac if i == pred_idx else "#cbd5e1"
            rows += f"""
            <div class="prob-row">
                <span class="prob-name">{CLASS_NAMES[i].replace('_', ' ')}</span>
                <div class="prob-bar-bg">
                    <div class="prob-bar-fill" style="width:{w};background:{col};"></div>
                </div>
                <span class="prob-pct">{probs[i]*100:.1f}%</span>
            </div>"""
        st.markdown(f'<div class="card">{rows}</div>', unsafe_allow_html=True)

        # Bouton rapport
        st.markdown("---")
        if st.button("Générer le rapport PDF", use_container_width=True):
            with st.spinner("Génération du rapport via LLM local…"):
                report_text = generate_report_text(
                    patient=patient,
                    operator=st.session_state.operator_name,
                    pred_name=pred_name,
                    confidence=confidence,
                    probs=probs,
                    class_names=CLASS_NAMES,
                    stage_info=STAGE_INFO,
                )
                pdf_bytes = build_pdf(
                    patient=patient,
                    operator=st.session_state.operator_name,
                    pred_name=pred_name,
                    confidence=confidence,
                    probs=probs,
                    orig_pil=pil_img,
                    seg_pil=overlay_pil,
                    report_text=report_text,
                    class_names=CLASS_NAMES,
                    stage_info=STAGE_INFO,
                )

            fname = (
                f"rapport_{patient['nom'].replace(' ', '_')}_"
                f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            )
            st.download_button(
                label="Télécharger le rapport",
                data=pdf_bytes,
                file_name=fname,
                mime="application/pdf",
                use_container_width=True,
            )
            with st.expander("Aperçu du texte généré par le LLM"):
                st.text(report_text)

    else:
        st.markdown("""
        <div style="padding:4rem 1rem;text-align:center;color:#94a3b8;font-size:0.86rem;">
            Chargez une image et cliquez sur <b>Analyser</b>.
        </div>""", unsafe_allow_html=True)

# Sidebar déconnexion
with st.sidebar:
    st.markdown(f"**Connecté**  \n{st.session_state.operator_name}")
    st.caption(f"Code : `{st.session_state.operator_code}`")
    st.markdown("---")
    if st.button("Se déconnecter"):
        st.session_state.authenticated = False
        st.session_state.pop("results", None)
        st.rerun()