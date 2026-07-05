from shiny import App, ui, render, reactive
import pandas as pd
import joblib

modelo = joblib.load("modelo.pkl")

print(modelo.classes_)

# =========================
# UI MELHORADA
# =========================

app_ui = ui.page_fluid(

    ui.panel_title("🌱 Sistema Inteligente de Irrigação"),

    ui.h4("Parâmetros da Lavoura"),

    ui.layout_columns(

        ui.input_select("solo", "Tipo de Solo", {
            "Clay":"Clay",
            "Loamy":"Loamy",
            "Sandy":"Sandy",
            "Silt":"Silt"
        }),

        ui.input_select("cultura", "Tipo de Cultura", {
            "Cotton":"Cotton",
            "Maize":"Maize",
            "Potato":"Potato",
            "Rice":"Rice",
            "Sugarcane":"Sugarcane",
            "Wheat":"Wheat"
        }),

        ui.input_select("irrigacao", "Forma de Irrigação", {
            "Canal":"Canal",
            "Drip":"Drip",
            "Rainfed":"Rainfed",
            "Sprinkler":"Sprinkler"
        }),

        ui.input_select("agua", "Fonte de Água", {
            "Groundwater":"Groundwater",
            "Rainwater":"Rainwater",
            "Reservoir":"Reservoir",
            "River":"River"
        }),

        ui.input_numeric("umidade", "Umidade do Solo (%)", 30),
        ui.input_numeric("chuva", "Precipitação (mm)", 100),
        ui.input_numeric("temperatura", "Temperatura (°C)", 25),

    ),

    ui.br(),

    ui.input_action_button("prever", "🔍 Prever Irrigação", class_="btn-primary"),

    ui.hr(),

    ui.h3("Resultado"),

    ui.output_text("resultado")
)

# =========================
# SERVER LIMPO
# =========================

def server(input, output, session):

    @output
    @render.text
    @reactive.event(input.prever)
    def resultado():

        novo = pd.DataFrame({
            "Soil_Type":[input.solo()],
            "Crop_Type":[input.cultura()],
            "Irrigation_Type":[input.irrigacao()],
            "Water_Source":[input.agua()],
            "Soil_Moisture":[input.umidade()],
            "Rainfall_mm":[input.chuva()],
            "Temperature_C":[input.temperatura()]
        })

        pred = modelo.predict(novo)[0]
        probs = modelo.predict_proba(novo)[0]

        resultado_probs = dict(zip(modelo.classes_, probs))
        pred = max(resultado_probs, key=resultado_probs.get)

        # probabilidade máxima
        confianca = max(probs)

        resultado_probs = dict(zip(modelo.classes_, probs))

        texto_probs = (
        f"📊 Probabilidades:\n"
        f"High: {resultado_probs.get('High', 0):.1%}\n"
        f"Medium: {resultado_probs.get('Medium', 0):.1%}\n"
        f"Low: {resultado_probs.get('Low', 0):.1%}\n"
        )


        if pred == "Low":
            classe = "🟢 BAIXA necessidade de irrigação"
        elif pred == "Medium":
            classe = "🟠 MÉDIA necessidade de irrigação"
        else:
            classe = "🔴 ALTA necessidade de irrigação"

        return f"{classe}\nConfiança: {confianca:.1%}\n\n{texto_probs}"


app = App(app_ui, server)