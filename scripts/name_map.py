"""Mappa nomi squadre tra The Odds API e football-data.co.uk."""
ODDS_TO_FD = {
    "AC Milan": "Milan",
    "Inter Milan": "Inter",
    "Atalanta BC": "Atalanta",
    "AS Roma": "Roma",
    "Hellas Verona": "Verona",
    "Genoa": "Genoa", "Frosinone": "Frosinone", "Lazio": "Lazio",
    "Cagliari": "Cagliari", "Lecce": "Lecce", "Monza": "Monza",
    "Napoli": "Napoli", "Bologna": "Bologna", "Sassuolo": "Sassuolo",
    "Juventus": "Juventus", "Torino": "Torino", "Como": "Como",
    "Parma": "Parma", "Udinese": "Udinese", "Venezia": "Venezia",
    "Fiorentina": "Fiorentina",
}


def to_fd(name: str) -> str:
    return ODDS_TO_FD.get(name, name)
