"""
detecteur de maladie a partir des symptomes par ML
"""
import os
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup

URL_BASE       = "https://www.nhs.uk"
URL_CONDITIONS = "https://www.nhs.uk/conditions/"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-GB,en;q=0.9",
}

SYMPTOMES_CIBLES = [
    "fever","cough","fatigue","headache","sore_throat",
    "shortness_of_breath","nausea","vomiting","diarrhea",
    "chest_pain","abdominal_pain","back_pain","joint_pain",
    "muscle_pain","rash","itching","swelling","dizziness",
    "confusion","loss_of_appetite","weight_loss","weight_gain",
    "insomnia","anxiety","depression","blurred_vision",
    "runny_nose","sneezing","wheezing","constipation",
    "bloating","jaundice","bruising","bleeding","hair_loss",
    "dry_skin","sweating","chills","numbness","weakness",
    "tremor","seizure","memory_loss","difficulty_swallowing",
    "frequent_urination","painful_urination","blood_in_urine",
    "palpitations","fainting","loss_of_smell",
]

MOTS_CLES = {
    "fever":                ["fever","high temperature","pyrexia"],
    "cough":                ["cough","coughing"],
    "fatigue":              ["fatigue","tiredness","exhaustion","lethargy"],
    "headache":             ["headache","head pain","migraine"],
    "sore_throat":          ["sore throat","throat pain","pharyngitis"],
    "shortness_of_breath":  ["shortness of breath","breathlessness","difficulty breathing"],
    "nausea":               ["nausea","feeling sick","queasy"],
    "vomiting":             ["vomiting","being sick","throwing up"],
    "diarrhea":             ["diarrhoea","diarrhea","loose stools"],
    "chest_pain":           ["chest pain","chest tightness","chest discomfort"],
    "abdominal_pain":       ["abdominal pain","stomach pain","tummy pain"],
    "back_pain":            ["back pain","backache","lower back"],
    "joint_pain":           ["joint pain","arthralgia","aching joints"],
    "muscle_pain":          ["muscle pain","myalgia","muscle ache"],
    "rash":                 ["rash","skin rash","hives","urticaria"],
    "itching":              ["itching","itchy","pruritus"],
    "swelling":             ["swelling","swollen","oedema","edema"],
    "dizziness":            ["dizziness","dizzy","vertigo","lightheaded"],
    "confusion":            ["confusion","confused","disorientation"],
    "loss_of_appetite":     ["loss of appetite","reduced appetite","not hungry"],
    "weight_loss":          ["weight loss","losing weight"],
    "weight_gain":          ["weight gain","gaining weight"],
    "insomnia":             ["insomnia","difficulty sleeping","sleep problems"],
    "anxiety":              ["anxiety","anxious","panic attack"],
    "depression":           ["depression","low mood","feeling depressed"],
    "blurred_vision":       ["blurred vision","blurry vision","vision problems"],
    "runny_nose":           ["runny nose","nasal discharge","rhinorrhoea"],
    "sneezing":             ["sneezing","sneeze"],
    "wheezing":             ["wheezing","wheeze"],
    "constipation":         ["constipation","difficulty passing stools"],
    "bloating":             ["bloating","bloated"],
    "jaundice":             ["jaundice","yellowing","yellow skin"],
    "bruising":             ["bruising","bruise","easy bruising"],
    "bleeding":             ["bleeding","blood loss","haemorrhage"],
    "hair_loss":            ["hair loss","alopecia","losing hair"],
    "dry_skin":             ["dry skin","skin dryness","flaky skin"],
    "sweating":             ["sweating","night sweats","hyperhidrosis"],
    "chills":               ["chills","shivering","rigors"],
    "numbness":             ["numbness","tingling","pins and needles"],
    "weakness":             ["weakness","muscle weakness"],
    "tremor":               ["tremor","trembling","shaking"],
    "seizure":              ["seizure","fit","convulsion"],
    "memory_loss":          ["memory loss","forgetfulness"],
    "difficulty_swallowing":["difficulty swallowing","dysphagia"],
    "frequent_urination":   ["frequent urination","urinating more often"],
    "painful_urination":    ["painful urination","burning urination","dysuria"],
    "blood_in_urine":       ["blood in urine","haematuria"],
    "palpitations":         ["palpitations","heart racing","rapid heartbeat"],
    "fainting":             ["fainting","faint","blackout","syncope"],
    "loss_of_smell":        ["loss of smell","anosmia","can't smell"],
}

LABELS_FR = {
    "fever":"Fièvre","cough":"Toux","fatigue":"Fatigue",
    "headache":"Maux de tête","sore_throat":"Mal de gorge",
    "shortness_of_breath":"Essoufflement","nausea":"Nausées",
    "vomiting":"Vomissements","diarrhea":"Diarrhée",
    "chest_pain":"Douleur thoracique","abdominal_pain":"Douleur abdominale",
    "back_pain":"Douleur dorsale","joint_pain":"Douleur articulaire",
    "muscle_pain":"Douleur musculaire","rash":"Éruption cutanée",
    "itching":"Démangeaisons","swelling":"Gonflement",
    "dizziness":"Vertiges","confusion":"Confusion",
    "loss_of_appetite":"Perte d'appétit","weight_loss":"Perte de poids",
    "weight_gain":"Prise de poids","insomnia":"Insomnie",
    "anxiety":"Anxiété","depression":"Dépression",
    "blurred_vision":"Vision floue","runny_nose":"Nez qui coule",
    "sneezing":"Éternuements","wheezing":"Sifflement respiratoire",
    "constipation":"Constipation","bloating":"Ballonnements",
    "jaundice":"Jaunisse","bruising":"Ecchymoses",
    "bleeding":"Saignements","hair_loss":"Chute de cheveux",
    "dry_skin":"Peau sèche","sweating":"Transpiration excessive",
    "chills":"Frissons","numbness":"Engourdissement",
    "weakness":"Faiblesse","tremor":"Tremblements",
    "seizure":"Convulsions","memory_loss":"Perte de mémoire",
    "difficulty_swallowing":"Difficulté à avaler",
    "frequent_urination":"Urination fréquente",
    "painful_urination":"Urination douloureuse",
    "blood_in_urine":"Sang dans les urines",
    "palpitations":"Palpitations","fainting":"Évanouissement",
    "loss_of_smell":"Perte d'odorat",
}


def recuperer_liste_maladies(url_base: str, max_maladies: int = 60) -> list:
    print(f"[scraper] Récupération des maladies depuis : {url_base}")
    response = requests.get(url_base, headers=HEADERS, timeout=15)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    maladies = []
    vus = set()
    for lien in soup.find_all("a", href=True):
        href = lien["href"]
        nom  = lien.get_text(strip=True)
        if (href.startswith("/conditions/") and
            href != "/conditions/" and
            len(href.split("/")) == 4 and
            nom and len(nom) > 2 and
            "see " not in nom.lower()):
            url_complete = URL_BASE + href
            if url_complete not in vus:
                vus.add(url_complete)
                maladies.append({"nom": nom, "url": url_complete})

    maladies = maladies[:max_maladies]
    print(f"[scraper] {len(maladies)} maladies trouvées")
    return maladies


def scraper_symptomes(url: str, nom: str) -> dict:
    ligne = {s: 0 for s in SYMPTOMES_CIBLES}
    ligne["maladie"] = nom
    ligne["label"]   = 1
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        texte = _section_symptoms(soup)
        if not texte:
            main = soup.find("main") or soup.find("div", {"id": "maincontent"})
            texte = main.get_text(separator=" ").lower() if main else ""
        for symptome, mots in MOTS_CLES.items():
            for mot in mots:
                if mot.lower() in texte.lower():
                    ligne[symptome] = 1
                    break
    except Exception as e:
        print(f"[scraper] ⚠️ {nom} : {e}")
    return ligne


def _section_symptoms(soup):
    for balise in ["h2","h3","h4"]:
        for titre in soup.find_all(balise):
            if "symptom" in titre.get_text(strip=True).lower():
                texte = ""
                for frere in titre.find_next_siblings():
                    if frere.name in ["h2","h3","h4"]:
                        break
                    texte += frere.get_text(separator=" ")
                return texte
    return ""


def generer_sains(nb: int) -> list:
    import random
    random.seed(42)
    sains = []
    legers = ["runny_nose","sneezing","fatigue","headache"]
    for _ in range(nb):
        ligne = {s: 0 for s in SYMPTOMES_CIBLES}
        ligne["maladie"] = "Sain"
        ligne["label"]   = 0
        nb_s = random.randint(0, 1)
        for s in random.sample(legers, min(nb_s, len(legers))):
            ligne[s] = 1
        sains.append(ligne)
    return sains


def construire_dataset(url: str, max_maladies: int = 100,
                       callback=None) -> pd.DataFrame:
    maladies = recuperer_liste_maladies(url, max_maladies)
    lignes = []
    for i, m in enumerate(maladies):
        if callback:
            callback(i+1, len(maladies), m["nom"])
        ligne = scraper_symptomes(m["url"], m["nom"])
        lignes.append(ligne)
        time.sleep(0.4)

    # Supprimer les maladies sans symptômes
    lignes = [l for l in lignes if sum(l[s] for s in SYMPTOMES_CIBLES) > 0]

    sains = generer_sains(len(lignes))
    df = pd.DataFrame(lignes + sains)

    cols = ["maladie"] + SYMPTOMES_CIBLES + ["label"]
    df = df[[c for c in cols if c in df.columns]]
    df[SYMPTOMES_CIBLES] = df[SYMPTOMES_CIBLES].fillna(0).astype(int)

    os.makedirs("data/raw", exist_ok=True)
    df.to_csv("data/raw/symptoms_dataset.csv", index=False)
    print(f"[scraper] ✅ Dataset : {len(df)} lignes")
    return df
