import requests
import os
import json
import google.generativeai as genai
from groq import Groq  # <--- Needed for DeepSeek R1
from xhtml2pdf import pisa

# --- CONFIGURATION ---
SERPER_API_KEY = "0f0678396df38b459ec2446b7680784f92d33837"
GEMINI_API_KEY = "AIzaSyDR_e5kOoAsdWvgSkYURpjWk6wLMZ_GGhI"  # Put your actual key here
GROQ_API_KEY = "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"  # Put your actual key here


# --- 1. REASONING LAYER (DeepSeek R1) ---

def decide_repair_strategy(car_info, part_name, damage_type, severity, api_key):
    """
    Uses DeepSeek R1 to analyze damage and decide on Repair vs Replace.
    Returns a dictionary with the decision, reasoning, and search query.
    """
    print("🧠 DeepSeek R1: Analyzing damage logic...")

    client = Groq(api_key=api_key)

    prompt = f"""
    You are a Senior Automotive Adjuster. 
    Analyze the following case to decide between REPAIR or REPLACE.

    Vehicle: {car_info}
    Part: {part_name}
    Damage Type: {damage_type}
    Severity: {severity}

    Logic Rules:
    1. Safety parts (glass, airbags, seatbelts) are ALWAYS replace.
    2. Severe structural damage (frame) is replace.
    3. Minor cosmetic damage (dents, scratches) on metal/plastic is repair.
    4. Complex electronics (headlights with cracked lens) are usually replace.

    Task:
    1. Make a decision: "Repair" or "Replace".
    2. Write a short reasoning (1 sentence).
    3. Generate a highly optimized Google Search Query for Egypt.
       - If Replace: Search for "{part_name} price {car_info} Egypt"
       - If Repair: Search for "{part_name} repair cost {car_info} Egypt" or "body shop prices"

    OUTPUT FORMAT:
    Return ONLY a raw JSON object (no markdown, no ```json tags).
    {{
        "decision": "Replace",
        "reasoning": "Headlight lenses cannot be safely resealed after shattering.",
        "search_query": "Toyota Corolla 2022 headlight assembly price Egypt"
    }}
    """

    try:
        # We use the deepseek-r1 model hosted on Groq
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1  # Low temp for logical consistency
        )

        content = response.choices[0].message.content

        # Cleanup: Remove <think> tags if DeepSeek includes them in the output
        if "</think>" in content:
            content = content.split("</think>")[-1].strip()

        # Cleanup: Remove markdown code blocks if present
        content = content.replace("```json", "").replace("```", "").strip()

        return json.loads(content)

    except Exception as e:
        print(f"❌ Error in Reasoning Layer: {e}")
        # Fallback default
        return {
            "decision": "Check Manual",
            "reasoning": "AI Failed",
            "search_query": f"{car_info} {part_name} price Egypt"
        }


# --- 2. SEARCH LAYER ---

def search_web(query, api_key):
    print(f"🔍 Searching web for: {query}...")
    url = "https://google.serper.dev/search"
    payload = {"q": query, "num": 8}  # Reduced num slightly for speed
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"❌ Error during web search: {e}")
        return None


# --- 3. REPORT LAYER (Gemini) ---

def generate_final_report(strategy_data, search_results, api_key):
    """
    Uses Gemini to synthesize the DeepSeek decision + Google Results into a PDF.
    """
    print("🤖 Gemini 2.5: Writing final report...")

    try:
        genai.configure(api_key=api_key)
        # Note: 'gemini-1.5-flash' is the correct stable model name
        model = genai.GenerativeModel('gemini-2.5-flash')

        prompt = f"""
        You are an expert auto repair estimator.

        CASE DETAILS:
        - Decision: {strategy_data['decision']}
        - Logic: {strategy_data['reasoning']}

        MARKET DATA (Search Results):
        {search_results}

        Task:
        Generate a professional HTML repair estimate report.

        Structure:
        1. **Assessment**: State clearly that the recommendation is to {strategy_data['decision']} and why.
        2. **Cost Analysis**: Extract prices from search results. Convert to EGP. Calculate Min/Max/Avg.
        3. **Recommendation**: Final advice for the user.

        Styling Requirements:
        - Use standard HTML5 with internal CSS.
        - Font: Helvetica or Arial.
        - Use a color scheme of Navy Blue and White.
        - Render the "Decision" (Repair/Replace) in a large, colored badge (Red for Replace, Green for Repair).
        - RETURN ONLY RAW HTML.
        """

        response = model.generate_content(prompt)
        return response.text

    except Exception as e:
        print(f"❌ Error during Gemini generation: {e}")
        return None


# --- 4. PDF CONVERSION ---

def convert_html_to_pdf(source_html, output_filename):
    print(f"⚙️ Converting Report to PDF...")
    try:
        with open(output_filename, "wb") as result_file:
            pisa_status = pisa.CreatePDF(source_html, dest=result_file)

        if not pisa_status.err:
            print(f"✅ Success! PDF saved as: {os.path.abspath(output_filename)}")
        else:
            print("❌ Error during PDF conversion.")
    except Exception as e:
        print(f"❌ File Error: {e}")


# --- MAIN ORCHESTRATOR ---

def main():
    # --- INPUTS (These would come from your Streamlit App or YOLO model) ---
    car_info = "Toyota Corolla 2022"
    part_name = "Front Bumper"
    damage_type = "Deep Dent with Paint Scratch"
    severity = "Moderate"
    # Try changing severity to "Severe/Cracked" to see the logic change!

    print(f"🚗 Processing Case: {car_info} | {part_name} | {severity}")

    # STEP 1: DeepSeek decides Strategy & Query
    strategy = decide_repair_strategy(
        car_info, part_name, damage_type, severity, GROQ_API_KEY
    )
    print(f"👉 Strategy: {strategy['decision']} ({strategy['reasoning']})")

    # STEP 2: Search using the OPTIMIZED query
    results = search_web(strategy['search_query'], SERPER_API_KEY)
    if not results: return

    # STEP 3: Gemini writes the report
    html_content = generate_final_report(strategy, results, GEMINI_API_KEY)
    if not html_content: return

    # Cleanup HTML markdown
    if "```html" in html_content:
        html_content = html_content.split("```html")[1].split("```")[0]
    elif "```" in html_content:
        html_content = html_content.split("```")[1].split("```")[0]

    # STEP 4: Convert to PDF
    convert_html_to_pdf(html_content, "smart_damage_report.pdf")


if __name__ == "__main__":
    main()