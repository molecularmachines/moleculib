from mendeleev.fetch import fetch_table

elements = fetch_table("elements").assign(
            symbol=lambda df: df.symbol.str.upper()
        )