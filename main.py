from scraper_nhs import construire_dataset, URL_CONDITIONS
from model import entrainer

print("=== SCRAPING ===")
df = construire_dataset(URL_CONDITIONS, max_maladies=100)

print("=== ENTRAINEMENT ===")
resultats = entrainer(df)

print(resultats)python maim