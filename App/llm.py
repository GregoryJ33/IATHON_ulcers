import datetime
import numpy as np
import ollama

OLLAMA_MODEL = "llama3.1"

def generate_report_text(
    patient:     dict,
    operator:    str,
    pred_name:   str,
    confidence:  float,
    probs:       list,
    class_names: list,
    stage_info:  dict,
) -> str:
    """
    Génère le texte clinique du rapport via Ollama.
    Retourne une chaîne de caractères (rapport ou message d'erreur clair).
    """
    desc      = stage_info[pred_name][3]
    date_str  = datetime.datetime.now().strftime("%d/%m/%Y à %H:%M")
    probs_txt = "\n".join(
        f"  - {class_names[i].replace('_', ' ')}: {probs[i]*100:.1f}%"
        for i in np.argsort(probs)[::-1]
    )

    prompt = f"""
    Tu es un assistant médical expert en plaies chroniques et ulcères de pression.

    CONSIGNES DE RÉPONSE :
    1. Rédige un rapport clinique structuré, sobre et professionnel en français.
    2. Base-toi uniquement sur les informations fournies ci-dessous, n'invente rien.
    3. Adapte les recommandations au stade d'ulcère identifié.
    4. Si la confiance IA est inférieure à 70%, ajoute obligatoirement une note de vigilance.
    5. Termine TOUJOURS par la mention légale indiquant que ce rapport est généré par IA.

    CONTEXTE CLINIQUE :
    - Date d'analyse : {date_str}
    - Opérateur : {operator}
    - Patient : {patient['nom']}, {patient['age']} ans, {patient['sexe']}
    - Localisation de la plaie : {patient['localisation']}
    - Observation clinique déclarée : {patient['observation']}

    RÉSULTATS DE L'ANALYSE IA :
    - Classification retenue : {pred_name.replace('_', ' ')} — {desc}
    - Confiance du modèle : {confidence*100:.1f}%
    - Distribution complète des probabilités :
{probs_txt}

    STRUCTURE ATTENDUE DU RAPPORT :
    1. Résumé clinique (2-3 phrases décrivant l'état de la plaie selon le stade)
    2. Recommandations de soins (3-4 points concis et actionnables)
    3. Note de vigilance (si confiance < 70% ou stade ambigu)
    4. Mention légale obligatoire
    """

    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Tu es un assistant médical clinique précis et professionnel. "
                        "Réponds toujours en français. "
                        "Ne génère que le contenu du rapport, sans introduction ni conclusion superflue."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            stream=False,
            options={
                "temperature": 0.2,    # très bas pour des réponses reproductibles et fiables
                "num_predict": 700,
            },
        )
        return response["message"]["content"].strip()

    except ollama.ResponseError as e:
        if "404" in str(e) or "not found" in str(e).lower():
            return (
                f"[Modèle '{OLLAMA_MODEL}' non installé]\n\n"
                f"Installez-le avec :\n    ollama pull {OLLAMA_MODEL}\n\n"
                "Alternatives légères :\n"
                "    ollama pull phi3\n"
                "    ollama pull llama3.1\n"
                "    ollama pull mistral"
            )
        return f"[Erreur Ollama : {e}]"

    except Exception as e:
        if "Connection refused" in str(e) or "connect" in str(e).lower():
            return (
                "[Ollama non disponible — le service n'est pas lancé]\n\n"
                "Pour activer la génération :\n"
                "  1. Installez Ollama : https://ollama.com\n"
                f"  2. ollama pull {OLLAMA_MODEL}\n"
                "  3. ollama serve"
            )
        return f"[Erreur inattendue : {e}]"