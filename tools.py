import requests
import os

API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")

def get_stock_price(company_name: str):
    """Usa Alpha Vantage para buscar ticker y obtener precio."""
    # 1. Buscar el símbolo
    search_url = f"https://www.alphavantage.co/query?function=SYMBOL_SEARCH&keywords={company_name}&apikey={API_KEY}"
    search_data = requests.get(search_url).json()
    
    matches = search_data.get("bestMatches", [])
    if not matches:
        return f"No se encontró ninguna empresa llamada '{company_name}'."
    
    ticker = matches[0]["1. symbol"]
    name = matches[0]["2. name"]

    # 2. Obtener el precio
    quote_url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={ticker}&apikey={API_KEY}"
    quote_data = requests.get(quote_url).json()
    price = quote_data["Global Quote"]["05. price"]

    return f"El precio actual de {name} ({ticker}) es {float(price):.2f} USD."

# print(get_stock_price("Apple inc"))
