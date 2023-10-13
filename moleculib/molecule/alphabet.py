from mendeleev.fetch import fetch_table

elements = fetch_table("elements").assign(
            symbol=lambda df: df.symbol.str.upper()
        )

#TODO: add deuterium and tritium using the isotopes table